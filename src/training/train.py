# Vedaz Qwen Fine-Tuning Training Script
"""
Fine-tune Qwen models using QLoRA with the SFTTrainer from TRL.

Usage:
    python src/training/train.py
    python src/training/train.py --config configs/training_config.yaml
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
from datetime import datetime

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict) -> BitsAndBytesConfig | None:
    """Create BitsAndBytesConfig for QLoRA if enabled."""
    quant_cfg = config.get("quantization", {})
    if not quant_cfg.get("enabled", False):
        return None

    compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def setup_lora(config: dict) -> LoraConfig:
    """Create LoRA configuration."""
    lora_cfg = config.get("lora", {})
    return LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        bias="none",
    )


def load_model_and_tokenizer(config: dict, bnb_config: BitsAndBytesConfig | None):
    """Load the base model and tokenizer."""
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-7B-Instruct")
    torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    attn_impl = model_cfg.get("attn_implementation", "sdpa")

    print(f"\n[1/4] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[2/4] Loading model: {model_name}")
    print(f"       Dtype: {torch_dtype}, Attention: {attn_impl}")
    if bnb_config:
        print(f"       Quantization: 4-bit NF4 (QLoRA)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training if quantized
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # Disable cache for training (incompatible with gradient checkpointing)
    model.config.use_cache = False

    return model, tokenizer


def load_data(config: dict):
    """Load training and validation datasets."""
    dataset_cfg = config.get("dataset", {})
    train_path = dataset_cfg.get("path", "data/train.jsonl")
    val_path = dataset_cfg.get("val_path", "data/val.jsonl")

    print(f"\n[3/4] Loading datasets")
    print(f"       Train: {train_path}")
    print(f"       Val:   {val_path}")

    data_files = {"train": train_path}
    if os.path.exists(val_path):
        data_files["validation"] = val_path

    dataset = load_dataset("json", data_files=data_files)

    print(f"       Train samples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"       Val samples:   {len(dataset['validation'])}")

    return dataset


def create_training_args(config: dict) -> TrainingArguments:
    """Create TrainingArguments from config."""
    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        train_cfg.get("output_dir", "outputs/checkpoints"),
        f"run_{timestamp}"
    )

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        eval_steps=train_cfg.get("eval_steps", 100),
        eval_strategy="steps" if config.get("dataset", {}).get("val_path") else "no",
        save_total_limit=train_cfg.get("save_total_limit", 3),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        report_to=train_cfg.get("report_to", "tensorboard"),
        logging_dir=os.path.join("logs", "runs", f"run_{timestamp}"),
        seed=train_cfg.get("seed", 42),
    )


def format_chat_messages(example, tokenizer):
    """Apply the chat template to format messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with QLoRA")
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to training config YAML"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print("=" * 60)
    print("  Vedaz Qwen Fine-Tuning")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model:  {config['model']['name']}")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU:    {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected! Training will be extremely slow.")
        print("         Consider using Google Colab or a cloud GPU instance.")

    # Setup components
    bnb_config = setup_quantization(config)
    lora_config = setup_lora(config)
    model, tokenizer = load_model_and_tokenizer(config, bnb_config)

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"\n       Trainable params: {trainable_params:,} / {total_params:,}")
    print(f"       Trainable %:     {100 * trainable_params / total_params:.2f}%")

    # Load data
    dataset = load_data(config)

    # Apply chat template
    print(f"\n[4/4] Formatting with chat template")
    formatted_train = dataset["train"].map(
        lambda ex: format_chat_messages(ex, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    formatted_val = None
    if "validation" in dataset:
        formatted_val = dataset["validation"].map(
            lambda ex: format_chat_messages(ex, tokenizer),
            remove_columns=dataset["validation"].column_names,
        )

    # Create training arguments
    training_args = create_training_args(config)
    print(f"\n       Output: {training_args.output_dir}")
    print(f"       Epochs: {training_args.num_train_epochs}")
    print(f"       Batch:  {training_args.per_device_train_batch_size}")
    print(f"       Grad Accum: {training_args.gradient_accumulation_steps}")
    print(f"       Effective Batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"       LR: {training_args.learning_rate}")
    
    dataset_cfg = config.get("dataset", {})
    max_seq_length = dataset_cfg.get("max_seq_length", 2048)
    print(f"       Max Seq Len: {max_seq_length}")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        processing_class=tokenizer,
        max_seq_length=max_seq_length,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Train
    print(f"\n{'='*60}")
    print("  Starting Training...")
    print(f"{'='*60}\n")

    train_result = trainer.train()

    # Save final model
    print(f"\nSaving final adapter to: {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"{'='*60}")
    print(f"  Adapter saved to: {training_args.output_dir}")
    print(f"  TensorBoard logs: {training_args.logging_dir}")
    print(f"\n  Next steps:")
    print(f"    1. Merge adapter:  python src/training/merge_model.py --adapter_path {training_args.output_dir}")
    print(f"    2. Run inference:  python src/inference/run_inference.py")
    print(f"    3. View logs:      tensorboard --logdir logs/runs")


if __name__ == "__main__":
    main()
