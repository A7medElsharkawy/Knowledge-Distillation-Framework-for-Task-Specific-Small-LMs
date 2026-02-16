#!/usr/bin/env python3
"""
Script to test the fine-tuned model locally.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load environment variables
load_dotenv()

# Default paths - updated for new structure
BASE_MODEL = os.getenv("BASE_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "qwen_lora_sft"


def load_model_and_tokenizer(base_model: str, adapter_path: Path):
    """
    Load base model and LoRA adapter.
    
    Args:
        base_model: Path or name of the base model
        adapter_path: Path to the LoRA adapter
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path.exists():
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()  # Merge adapter for inference
    else:
        print(f"Warning: Adapter path not found: {adapter_path}")
        print("Using base model without fine-tuning.")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_length: int = 512):
    """
    Generate a response from the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        
    Returns:
        Generated text
    """
    # Format prompt according to Qwen template
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def interactive_inference():
    """
    Run interactive inference session.
    """
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL, MODEL_PATH)
    
    print("\nModel loaded! Type 'quit' or 'exit' to end the session.")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
            
            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nModel: ", end="", flush=True)
            response = generate_response(model, tokenizer, prompt)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def test_sample_prompts():
    """
    Test the model with sample prompts.
    """
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL, MODEL_PATH)
    
    test_prompts = [
        "What is machine learning?",
        "Explain the concept of fine-tuning in NLP.",
        "Write a short summary of recent AI developments."
    ]
    
    print("\nTesting model with sample prompts:")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        response = generate_response(model, tokenizer, prompt)
        print(f"Response: {response}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the fine-tuned model")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "test"],
        default="interactive",
        help="Inference mode: interactive or test"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_inference()
    else:
        test_sample_prompts()
