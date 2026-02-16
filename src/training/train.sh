#!/bin/bash

# Script to run Llama Factory training

# Configuration - updated paths for new structure
CONFIG_FILE="${CONFIG_FILE:-../../config/qwen_lora_sft.yaml}"
DATASET_DIR="${DATASET_DIR:-../../data/processed}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

echo "Starting training with Llama Factory..."
echo "Config: $CONFIG_FILE"
echo "Dataset directory: $DATASET_DIR"

# Run Llama Factory training
llamafactory-cli train "$CONFIG_FILE"

echo "Training completed!"
