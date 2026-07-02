# Hosting a Fine-Tuned Qwen Model on a VPS Using vLLM

## Objective
Deploy a fine-tuned Qwen model on a VPS using **vLLM** to expose an OpenAI-compatible inference API.

## 1. Provision the VPS
- Ubuntu 22.04 LTS
- NVIDIA GPU (recommended 16 GB+ VRAM)
- CUDA Toolkit and NVIDIA drivers installed
- SSH access configured

## 2. Prepare the Environment

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git

python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install torch transformers vllm
```

## 4. Upload the Fine-Tuned Model

Example directory structure:

```text
models/
└── qwen-finetuned/
```

If LoRA adapters are used, merge them with the base model before deployment or load them according to your serving strategy.

## 5. Launch the vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model models/qwen-finetuned \
    --host 0.0.0.0 \
    --port 8000
```

The server exposes an OpenAI-compatible API.

## 6. Test the Deployment

```bash
curl http://SERVER_IP:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model":"qwen-finetuned",
  "messages":[
    {"role":"user","content":"Can you analyze my career prospects?"}
  ]
}'
```

## 7. Production Recommendations

- Configure Nginx as a reverse proxy.
- Enable HTTPS using Let's Encrypt.
- Run the API server as a systemd service.
- Monitor GPU utilization with `nvidia-smi`.
- Store logs for monitoring and debugging.

## Workflow Summary

1. Provision VPS
2. Install Python and vLLM
3. Upload fine-tuned model
4. Start vLLM API server
5. Test inference endpoint
6. Deploy behind Nginx for production
