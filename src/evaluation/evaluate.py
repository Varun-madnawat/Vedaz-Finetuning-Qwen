# Evaluation Script for Vedaz Fine-Tuned Model
"""
Evaluate the fine-tuned model on the validation set.
Computes perplexity and generates sample responses for manual review.

Usage:
    python src/evaluation/evaluate.py --adapter_path outputs/checkpoints/run_XXXXX
    python src/evaluation/evaluate.py --model_path outputs/merged_model
"""

import json
import argparse
import torch
import yaml
from pathlib import Path
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_for_eval(args):
    """Load model for evaluation."""
    train_config = load_config("configs/training_config.yaml")
    base_model_name = train_config["model"]["name"]
    torch_dtype = getattr(torch, train_config["model"].get("torch_dtype", "bfloat16"))

    if args.adapter_path:
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    elif args.model_path:
        print(f"Loading merged model: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        raise ValueError("Provide either --adapter_path or --model_path")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, dataset, max_samples: int = None):
    """Compute perplexity on the validation set."""
    import math

    total_loss = 0.0
    total_tokens = 0
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    print(f"\nComputing perplexity on {len(samples)} samples...")

    for i, example in enumerate(samples):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

        if (i + 1) % 10 == 0:
            current_ppl = math.exp(total_loss / total_tokens)
            print(f"  Processed {i+1}/{len(samples)} - Running PPL: {current_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def generate_sample_responses(model, tokenizer, dataset, num_samples: int = 5):
    """Generate sample responses for manual review."""
    results = []
    samples = dataset.select(range(min(num_samples, len(dataset))))

    print(f"\nGenerating {len(samples)} sample responses...")

    for i, example in enumerate(samples):
        messages = example["messages"]

        # Find first user message and use everything up to it as context
        context = []
        first_user_msg = None
        for msg in messages:
            if msg["role"] == "user" and first_user_msg is None:
                first_user_msg = msg["content"]
                context.append(msg)
                break
            context.append(msg)

        if not first_user_msg:
            continue

        # Generate response
        text = tokenizer.apply_chat_template(
            context, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Get expected response
        expected = None
        for msg in messages:
            if msg["role"] == "assistant":
                expected = msg["content"]
                break

        results.append({
            "user_query": first_user_msg,
            "expected": expected,
            "generated": generated,
        })

        print(f"\n--- Sample {i+1} ---")
        print(f"User: {first_user_msg[:100]}...")
        print(f"Expected: {(expected or 'N/A')[:100]}...")
        print(f"Generated: {generated[:100]}...")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Vedaz model")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5, help="Number of sample responses")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    model, tokenizer = load_model_for_eval(args)

    # Load validation data
    val_path = "data/val.jsonl"
    if not Path(val_path).exists():
        print(f"Error: {val_path} not found. Run prepare_data.py first.")
        return

    val_dataset = load_dataset("json", data_files=val_path, split="train")
    print(f"Validation samples: {len(val_dataset)}")

    # Compute perplexity
    ppl = compute_perplexity(model, tokenizer, val_dataset)
    print(f"\n{'='*50}")
    print(f"  Validation Perplexity: {ppl:.2f}")
    print(f"{'='*50}")

    # Generate sample responses
    samples = generate_sample_responses(model, tokenizer, val_dataset, args.num_samples)

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/eval_results_{timestamp}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results = {
        "perplexity": ppl,
        "num_val_samples": len(val_dataset),
        "sample_responses": samples,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
