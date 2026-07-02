# Merge LoRA Adapter with Base Model
"""
Merges the trained LoRA adapter weights back into the base model,
creating a standalone model ready for deployment or upload to Hugging Face.

Usage:
    python src/training/merge_model.py --adapter_path outputs/checkpoints/run_XXXXX
    python src/training/merge_model.py --adapter_path outputs/checkpoints/run_XXXXX --output_dir outputs/merged_model
"""

import argparse
import torch
import yaml
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter_path", type=str, required=True,
        help="Path to the trained LoRA adapter (checkpoint directory)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/merged_model",
        help="Directory to save the merged model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to training config (to get base model name)"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push merged model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_repo", type=str, default=None,
        help="Hugging Face Hub repo name (e.g., 'username/model-name')"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["name"]
    torch_dtype = getattr(torch, config["model"].get("torch_dtype", "bfloat16"))

    print("=" * 60)
    print("  Merging LoRA Adapter")
    print("=" * 60)
    print(f"  Base model:   {model_name}")
    print(f"  Adapter:      {args.adapter_path}")
    print(f"  Output:       {args.output_dir}")

    # Load base model (without quantization for merging)
    print(f"\n[1/4] Loading base model (full precision)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Load and merge adapter
    print(f"[3/4] Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = model.merge_and_unload()

    # Save merged model
    print(f"[4/4] Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n[OK] Merged model saved to: {args.output_dir}")

    # Push to hub if requested
    if args.push_to_hub and args.hub_repo:
        print(f"\nPushing to Hugging Face Hub: {args.hub_repo}")
        merged_model.push_to_hub(args.hub_repo, safe_serialization=True)
        tokenizer.push_to_hub(args.hub_repo)
        print(f"[OK] Pushed to: https://huggingface.co/{args.hub_repo}")


if __name__ == "__main__":
    main()
