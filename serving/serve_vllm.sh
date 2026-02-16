#!/bin/bash

# Script to serve the fine-tuned model using vLLM

# Paths - updated for new structure
MODEL_PATH="${MODEL_PATH:-../../models/qwen_lora_sft}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "Starting vLLM server..."
echo "Model path: $MODEL_PATH"
echo "Port: $PORT"
echo "Host: $HOST"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    --tensor-parallel-size 1 \
    --trust-remote-code
