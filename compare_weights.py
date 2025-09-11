#!/usr/bin/env python3
"""
Short script to compare weights between a local model and a Hugging Face model.
Returns the number of different weights.
"""

import torch
from transformers import AutoModelForImageTextToText
import argparse
import sys

def compare_model_weights(local_path, hf_model_name):
    """Compare weights between local and HF models."""
    try:
        # Load local model
        print(f"Loading local model from {local_path}...")
        local_model = AutoModelForImageTextToText.from_pretrained(local_path, dtype=torch.bfloat16)
        
        # Load HF model
        print(f"Loading HF model {hf_model_name}...")
        hf_model = AutoModelForImageTextToText.from_pretrained(hf_model_name, dtype=torch.bfloat16)
        
        # Compare weights
        different_weights = 0
        total_weights = 0
        different_weight_names = []
        
        for (name1, param1), (name2, param2) in zip(local_model.named_parameters(), hf_model.named_parameters()):
            if name1 != name2:
                print(f"Warning: Parameter names don't match: {name1} vs {name2}")
                continue
                
            total_weights += 1
            
            # Check if shapes match
            if param1.shape != param2.shape:
                print(f"Shape mismatch for {name1}: {param1.shape} vs {param2.shape}")
                different_weights += 1
                different_weight_names.append(f"{name1} (shape mismatch)")
                continue
            
            # Check if values are different (with small tolerance for floating point)
            if not torch.allclose(param1, param2, atol=1e-6):
                different_weights += 1
                different_weight_names.append(name1)
        
        print(f"\nComparison complete:")
        print(f"Total parameters compared: {total_weights}")
        print(f"Different weights: {different_weights}")
        print(f"Identical weights: {total_weights - different_weights}")
        
        if different_weight_names:
            print(f"\nDifferent weight names:")
            for name in different_weight_names:
                print(f"  - {name}")
        
        # Check language_model.lm_head.weight existence and differences
        print(f"\nlanguage_model.lm_head.weight analysis:")
        
        local_has_lm_head = hasattr(local_model, 'language_model') and hasattr(local_model.language_model, 'lm_head')
        hf_has_lm_head = hasattr(hf_model, 'language_model') and hasattr(hf_model.language_model, 'lm_head')
        
        if local_has_lm_head and hf_has_lm_head:
            local_lm_head = local_model.language_model.lm_head.weight
            hf_lm_head = hf_model.language_model.lm_head.weight
            
            print(f"✓ language_model.lm_head.weight exists in both models")
            print(f"  Local shape: {local_lm_head.shape}")
            print(f"  HF shape: {hf_lm_head.shape}")
            
            if local_lm_head.shape == hf_lm_head.shape:
                if torch.allclose(local_lm_head, hf_lm_head, atol=1e-6):
                    print(f"  → Weights are IDENTICAL")
                else:
                    print(f"  → Weights are DIFFERENT")
            else:
                print(f"  → Shapes are DIFFERENT")
        elif local_has_lm_head:
            print(f"✓ language_model.lm_head.weight exists in LOCAL model only")
        elif hf_has_lm_head:
            print(f"✓ language_model.lm_head.weight exists in HF model only")
        else:
            print(f"✗ language_model.lm_head.weight not found in either model")
        
        return different_weights
        
    except Exception as e:
        print(f"Error: {e}")
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model weights")
    parser.add_argument("--local_path", default="./merged_model", help="Path to local model folder")
    parser.add_argument("--hf_model", default="lightonai/Mistral-Small-3.1-24B-Instruct-2503", help="Hugging Face model name")
    
    args = parser.parse_args()
    
    different_count = compare_model_weights(args.local_path, args.hf_model)
    sys.exit(0 if different_count >= 0 else 1)
