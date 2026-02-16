#!/usr/bin/env python3
"""
Script to generate synthetic training data using OpenAI API.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths - updated for new structure
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_training_sample(prompt_template: str) -> dict:
    """
    Generate a single training sample using OpenAI.
    
    Args:
        prompt_template: Template for generating the sample
        
    Returns:
        Dictionary with instruction, input, and output fields
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates high-quality training data for fine-tuning language models."},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        
        # Parse the response (adjust based on your prompt template)
        # This is a placeholder - customize based on your needs
        return {
            "instruction": "Your instruction here",
            "input": "",
            "output": content
        }
    except Exception as e:
        print(f"Error generating sample: {e}")
        return None


def generate_dataset(num_samples: int = 100, output_file: str = "synthetic_data.json"):
    """
    Generate a dataset of synthetic training samples.
    
    Args:
        num_samples: Number of samples to generate
        output_file: Name of the output JSON file
    """
    samples = []
    
    print(f"Generating {num_samples} training samples...")
    
    for i in tqdm(range(num_samples)):
        # Customize this prompt template based on your use case
        prompt_template = f"""
        Generate a training example for fine-tuning a language model.
        Create a realistic instruction-output pair related to news analysis or general knowledge.
        Format: instruction, input (optional), output.
        """
        
        sample = generate_training_sample(prompt_template)
        if sample:
            samples.append(sample)
    
    # Save to JSON file
    output_path = RAW_DATA_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(samples)} samples and saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="synthetic_data.json", help="Output filename")
    
    args = parser.parse_args()
    
    generate_dataset(num_samples=args.num_samples, output_file=args.output)
