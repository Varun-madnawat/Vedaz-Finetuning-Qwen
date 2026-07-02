# Inference Script for Vedaz Fine-Tuned Qwen Model
"""
Run interactive or batch inference with the fine-tuned model.
Supports both merged models and base model + LoRA adapter.

Usage:
    Interactive: python src/inference/run_inference.py
    With adapter: python src/inference/run_inference.py --adapter_path outputs/checkpoints/run_XXXXX
    Single query: python src/inference/run_inference.py --query "Mera naam Rahul hai..."
"""

import argparse
import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


SYSTEM_PROMPT = (
    "You are Vedaz's AI Vedic astrologer. You give compassionate, balanced, "
    "non-fatalistic guidance based on Vedic astrology. You never predict death, "
    "illness, or guaranteed misfortune. You never give exact dates or guaranteed "
    "outcomes. Remedies are always suggested as supportive spiritual practices, "
    "not guarantees. In moments of extreme emotional distress, you prioritize "
    "user safety by providing professional helpline resources."
)


def load_model(args, inference_cfg: dict):
    """Load model — either merged or base + adapter."""
    torch_dtype = getattr(torch, inference_cfg.get("torch_dtype", "bfloat16"))

    if args.adapter_path:
        # Load base model + LoRA adapter
        print(f"Loading base model + adapter...")
        print(f"  Adapter: {args.adapter_path}")

        # Get base model name from training config
        train_config = load_config("configs/training_config.yaml")
        base_model_name = train_config["model"]["name"]

        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
    else:
        # Load merged model
        model_path = inference_cfg.get("model_path", "outputs/merged_model")
        print(f"Loading merged model: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    model.eval()
    return model, tokenizer


def generate_response(
    model, tokenizer, messages: list[dict], inference_cfg: dict
) -> str:
    """Generate a response for the given messages."""
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=inference_cfg.get("max_new_tokens", 512),
            temperature=inference_cfg.get("temperature", 0.7),
            top_p=inference_cfg.get("top_p", 0.9),
            top_k=inference_cfg.get("top_k", 50),
            repetition_penalty=inference_cfg.get("repetition_penalty", 1.1),
            do_sample=inference_cfg.get("do_sample", True),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (exclude input)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def interactive_chat(model, tokenizer, inference_cfg: dict):
    """Run an interactive chat session."""
    print("\n" + "=" * 60)
    print("  Vedaz AI Vedic Astrologer - Interactive Chat")
    print("=" * 60)
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'reset' to start a new conversation.")
    print("=" * 60 + "\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break
        if user_input.lower() == "reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("\n[Conversation reset]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nVedaz: ", end="", flush=True)
        response = generate_response(model, tokenizer, messages, inference_cfg)
        print(response)
        print()

        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Vedaz model")
    parser.add_argument(
        "--config", type=str, default="configs/inference_config.yaml",
        help="Path to inference config YAML"
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="Path to LoRA adapter (if not using merged model)"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query to answer (non-interactive mode)"
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None,
        help="Override the default system prompt"
    )
    args = parser.parse_args()

    # Load config
    inference_cfg = load_config(args.config).get("inference", {})

    # Load model
    model, tokenizer = load_model(args, inference_cfg)

    system = args.system_prompt or SYSTEM_PROMPT

    if args.query:
        # Single query mode
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": args.query},
        ]
        response = generate_response(model, tokenizer, messages, inference_cfg)
        print(f"\nVedaz: {response}")
    else:
        # Interactive mode
        interactive_chat(model, tokenizer, inference_cfg)


if __name__ == "__main__":
    main()
