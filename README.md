# Vedaz Qwen Fine-Tuning

Fine-tune Qwen 2.5 models using QLoRA for the **Vedaz AI Vedic Astrologer** — a compassionate, non-fatalistic astrology assistant that speaks Hindi, Hinglish, and English.

## Project Structure

```
Vedaz-Finetuning-Quen/
├── configs/
│   ├── training_config.yaml      # Model, LoRA, training hyperparameters
│   └── inference_config.yaml     # Inference generation settings
├── data/
│   ├── raw/                      # Raw data files (JSON/JSONL)
│   ├── train.jsonl               # Processed training data (generated)
│   └── val.jsonl                 # Processed validation data (generated)
├── src/
│   ├── data/prepare_data.py      # Data loading, validation, train/val split
│   ├── training/train.py         # QLoRA fine-tuning with SFTTrainer
│   ├── training/merge_model.py   # Merge LoRA adapter into base model
│   ├── inference/run_inference.py # Interactive & batch inference
│   └── evaluation/evaluate.py    # Perplexity & sample response evaluation
├── outputs/                      # Checkpoints & merged models
├── logs/                         # TensorBoard training logs
├── notebooks/                    # Jupyter experimentation
├── scripts/                      # Shell/batch scripts
├── requirements.txt              # Frozen dependencies
└── .gitignore
```

## Quick Start

### 1. Setup
```bash
python -m venv venv
.\venv\Scripts\Activate      # Windows
pip install -r requirements.txt
```

### 2. Prepare Data
Place your raw data in `data/raw/` and run:
```bash
python src/data/prepare_data.py
```

### 3. Train
```bash
python src/training/train.py --config configs/training_config.yaml
```

### 4. Merge Adapter
```bash
python src/training/merge_model.py --adapter_path outputs/checkpoints/run_XXXXX
```

### 5. Inference
```bash
python src/inference/run_inference.py
```

### 6. Evaluate
```bash
python src/evaluation/evaluate.py --adapter_path outputs/checkpoints/run_XXXXX
```

## Configuration

All hyperparameters are controlled via `configs/training_config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Model | Qwen/Qwen2.5-7B-Instruct | Base model from HuggingFace |
| LoRA r | 64 | LoRA rank |
| LoRA alpha | 128 | LoRA scaling factor |
| Quantization | 4-bit NF4 | QLoRA quantization |
| Batch size | 2 | Per-device batch size |
| Grad accumulation | 8 | Effective batch = 16 |
| Learning rate | 2e-4 | Peak learning rate |
| Epochs | 3 | Training epochs |
| Max seq length | 2048 | Maximum sequence length |

## Data Format

Training data uses the OpenAI chat format:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Hardware Requirements

- **Minimum**: 1x GPU with 16 GB VRAM (e.g., RTX 4080, T4)
- **Recommended**: 1x GPU with 24 GB VRAM (e.g., RTX 4090, A10G)
- For the 7B model with 4-bit QLoRA, ~12-16 GB VRAM is typically needed.