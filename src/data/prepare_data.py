#!/usr/bin/env python3
"""
Script to format raw data for Llama Factory training.
Converts raw data into the format expected by Llama Factory.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import random

# Paths - updated for new structure
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(input_file: str) -> List[Dict]:
    """
    Load raw data from JSON file.
    
    Args:
        input_file: Path to input JSON file
        
    Returns:
        List of data samples
    """
    input_path = RAW_DATA_DIR / input_file
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def format_for_llama_factory(sample: Dict) -> Dict:
    """
    Format a single sample for Llama Factory.
    
    Args:
        sample: Raw sample with instruction, input, output fields
        
    Returns:
        Formatted sample for Llama Factory
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    
    # Format according to Llama Factory's expected format
    # Adjust based on your template requirements
    formatted = {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }
    
    return formatted


def split_train_val(data: List[Dict], val_ratio: float = 0.1) -> tuple:
    """
    Split data into training and validation sets.
    
    Args:
        data: List of data samples
        val_ratio: Ratio of validation data
        
    Returns:
        Tuple of (train_data, val_data)
    """
    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data


def prepare_data(
    input_file: str = "synthetic_data.json",
    output_train: str = "train.json",
    output_val: str = "val.json",
    val_ratio: float = 0.1
):
    """
    Prepare data for training by formatting and splitting.
    
    Args:
        input_file: Input JSON file in raw directory
        output_train: Output training JSON file
        output_val: Output validation JSON file
        val_ratio: Ratio of validation data
    """
    print(f"Loading data from {input_file}...")
    raw_data = load_raw_data(input_file)
    
    print(f"Formatting {len(raw_data)} samples...")
    formatted_data = [format_for_llama_factory(sample) for sample in raw_data]
    
    print(f"Splitting into train/val (ratio: {1-val_ratio}/{val_ratio})...")
    train_data, val_data = split_train_val(formatted_data, val_ratio)
    
    # Save training data
    train_path = PROCESSED_DATA_DIR / output_train
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_data)} training samples to {train_path}")
    
    # Save validation data
    val_path = PROCESSED_DATA_DIR / output_val
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(val_data)} validation samples to {val_path}")
    
    return train_path, val_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for Llama Factory training")
    parser.add_argument("--input", type=str, default="synthetic_data.json", help="Input filename")
    parser.add_argument("--train_output", type=str, default="train.json", help="Training output filename")
    parser.add_argument("--val_output", type=str, default="val.json", help="Validation output filename")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    
    args = parser.parse_args()
    
    prepare_data(
        input_file=args.input,
        output_train=args.train_output,
        output_val=args.val_output,
        val_ratio=args.val_ratio
    )
