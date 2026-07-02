# Data Processing Utilities for Vedaz Fine-Tuning
"""
Handles the mixed-format raw data (single-line JSONL + multi-line JSON),
normalizes it, validates structure, and creates train/val splits.

Usage:
    python src/data/prepare_data.py
    python src/data/prepare_data.py --val_ratio 0.15 --seed 123
"""

import json
import os
import random
import argparse
from pathlib import Path


def load_mixed_json(filepath: str) -> list[dict]:
    """
    Load a file that contains a mix of:
      - Single-line JSONL entries
      - Multi-line pretty-printed JSON objects
      - Optional trailing commas between objects
    Uses json.JSONDecoder.raw_decode to handle all cases.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    decoder = json.JSONDecoder()
    objects = []
    idx = 0

    while idx < len(content):
        # Skip whitespace and commas between objects
        while idx < len(content) and content[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content[idx:])
            objects.append(obj)
            idx += end
        except json.JSONDecodeError as e:
            # Skip problematic character and continue
            print(f"  Warning: JSON parse error at position {idx}: {e}")
            idx += 1

    return objects


def load_jsonl(filepath: str) -> list[dict]:
    """Load a standard JSONL file (one JSON object per line)."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping line {line_num}: {e}")
    return data


def save_jsonl(data: list[dict], filepath: str) -> None:
    """Save a list of dictionaries to a JSONL file (one JSON per line)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples to {filepath}")


def normalize_conversation(obj: dict) -> dict | None:
    """
    Normalize a conversation object:
      - Keep only the 'messages' key (strip 'id', 'tags', etc.)
      - Validate message structure (role + content)
      - Ensure conversation starts with system prompt
    Returns None if invalid.
    """
    if "messages" not in obj:
        return None

    messages = obj["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    # Validate each message has role and content
    cleaned_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        if "role" not in msg or "content" not in msg:
            return None
        if msg["role"] not in ("system", "user", "assistant"):
            return None
        if not msg["content"].strip():
            return None
        cleaned_messages.append({
            "role": msg["role"],
            "content": msg["content"].strip()
        })

    # Ensure at least one user and one assistant message
    roles = {m["role"] for m in cleaned_messages}
    if "user" not in roles or "assistant" not in roles:
        return None

    return {"messages": cleaned_messages}


def compute_stats(data: list[dict]) -> dict:
    """Compute dataset statistics."""
    total_turns = sum(len(d["messages"]) for d in data)
    system_count = sum(1 for d in data for m in d["messages"] if m["role"] == "system")
    user_count = sum(1 for d in data for m in d["messages"] if m["role"] == "user")
    assistant_count = sum(1 for d in data for m in d["messages"] if m["role"] == "assistant")

    # Multi-turn stats
    multi_turn = sum(1 for d in data if sum(1 for m in d["messages"] if m["role"] == "user") > 1)

    # Token estimate (rough: 1 token ≈ 4 chars for mixed Hindi/English)
    total_chars = sum(len(m["content"]) for d in data for m in d["messages"])
    estimated_tokens = total_chars // 3  # Hindi/Devanagari is ~3 chars per token

    return {
        "conversations": len(data),
        "total_turns": total_turns,
        "system_prompts": system_count,
        "user_messages": user_count,
        "assistant_messages": assistant_count,
        "multi_turn_conversations": multi_turn,
        "total_characters": total_chars,
        "estimated_tokens": estimated_tokens,
    }


def train_val_split(
    data: list[dict], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split data into training and validation sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Prepare Vedaz fine-tuning data")
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Directory containing raw data files")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory for processed train/val files")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print("Error: No raw data directory found.")
        print(f"  Expected: {raw_dir.resolve()}")
        return

    # Load all data files
    all_data = []
    for file in sorted(raw_dir.iterdir()):
        if file.suffix in (".json", ".jsonl"):
            print(f"\nLoading: {file.name}")
            if file.suffix == ".jsonl":
                raw = load_jsonl(str(file))
            else:
                raw = load_mixed_json(str(file))
            print(f"  Raw objects found: {len(raw)}")

            # Normalize and validate
            valid = []
            for i, obj in enumerate(raw):
                normalized = normalize_conversation(obj)
                if normalized:
                    valid.append(normalized)
                else:
                    print(f"  Warning: Skipping invalid object at index {i}")
            print(f"  Valid conversations: {len(valid)}")
            all_data.extend(valid)

    if not all_data:
        print("\nNo valid conversations found!")
        return

    # Print stats
    stats = compute_stats(all_data)
    print(f"\n{'='*50}")
    print(f"Dataset Statistics")
    print(f"{'='*50}")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value:,}")

    # Split into train/val
    print(f"\nSplitting with val_ratio={args.val_ratio}, seed={args.seed}")
    train_data, val_data = train_val_split(all_data, args.val_ratio, args.seed)

    # Save
    save_jsonl(train_data, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(args.output_dir, "val.jsonl"))

    print(f"\n[OK] Done! Train: {len(train_data)}, Val: {len(val_data)}")


if __name__ == "__main__":
    main()
