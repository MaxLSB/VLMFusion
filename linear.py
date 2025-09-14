import argparse
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)
import torch
from tqdm import tqdm


def load_model_config(model_type):
    """Load model configuration from JSON file."""
    config_path = os.path.join(
        os.path.dirname(__file__), "models", f"{model_type}.json"
    )
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_models(vlm_repo, llm_repo, device):
    """Load VLM and LLM models and tokenizers with model-specific configurations."""
    model_type = detect_model_type(vlm_repo)
    print(f"Loading {model_type.upper()} models...")
    print(f"Device: {device}")

    # Load VLM model
    vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_repo)
    vlm_processor = AutoProcessor.from_pretrained(vlm_repo, trust_remote_code=True)
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        vlm_repo, dtype="bfloat16", trust_remote_code=True
    ).to(device)

    # Load LLM model
    llm_model = AutoModelForCausalLM.from_pretrained(llm_repo, dtype="bfloat16").to(
        device
    )

    return vlm_model, vlm_tokenizer, vlm_processor, llm_model, model_type


def detect_model_type(vlm_repo):
    """Detect model type by attempting to load with different AutoModel classes."""
    print("Detecting model type...")

    # Try LFM2 first (most common)
    try:
        AutoModelForImageTextToText.from_pretrained(
            vlm_repo, dtype="bfloat16", trust_remote_code=True
        )
        return "lfm2"
    except Exception:
        pass

    # Try Mistral
    try:
        AutoModelForCausalLM.from_pretrained(
            vlm_repo, dtype="bfloat16", trust_remote_code=True
        )
        return "mistral"
    except Exception:
        pass

    raise ValueError(
        f"Could not detect model type for {vlm_repo}. Supported models: LFM2 (AutoModelForImageTextToText) or Mistral (AutoModelForCausalLM)"
    )


def merge_models(vlm_model, llm_model, model_type, alpha=0.5):
    """Merge VLM and LLM models using linear combination (alpha * VLM + (1-alpha) * LLM)."""
    config = load_model_config(model_type)
    print(f"Merging models (α={alpha})...")

    # Collect LLM weights for merging
    llm_weights = {
        name: param
        for name, param in llm_model.named_parameters()
        if name.startswith(config["llm_prefix"])
        or (config["include_lm_head"] and name == "lm_head.weight")
    }

    # Collect VLM parameters to merge
    vlm_params_to_merge = [
        (name, param)
        for name, param in vlm_model.named_parameters()
        if name.startswith(config["vlm_prefix"])
        or (config["include_lm_head"] and name == "lm_head.weight")
    ]

    mismatches = 0
    with torch.no_grad():
        for name, param in tqdm(vlm_params_to_merge, desc="Merging weights"):
            # Convert VLM parameter name to LLM parameter name
            llm_name = (
                name.replace(config["vlm_prefix"], config["llm_prefix"])
                if name.startswith(config["vlm_prefix"])
                else name
            )

            if llm_name in llm_weights:
                llm_param = llm_weights[llm_name]
                if param.shape == llm_param.shape:
                    # Ensure both parameters are on the same device and dtype
                    llm_param = llm_param.to(device=param.device, dtype=param.dtype)
                    # Linear combination: alpha * VLM + (1-alpha) * LLM
                    param.data = alpha * param.data + (1 - alpha) * llm_param.data
                else:
                    mismatches += 1
            else:
                mismatches += 1

    if mismatches > 0:
        print(f"Warning: {mismatches} parameters could not be merged")
    else:
        print("✓ All parameters merged successfully")

    return


def save_model(vlm_model, vlm_tokenizer, vlm_processor, output_path):
    """Save the merged model, tokenizer, and processor to the specified path."""
    print("Saving model...")
    os.makedirs(output_path, exist_ok=True)
    vlm_model.save_pretrained(output_path, safe_serialization=True)
    vlm_tokenizer.save_pretrained(output_path)
    vlm_processor.save_pretrained(output_path)
    print(f"✓ Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge VLM and LLM models (supports Mistral and LFM2)"
    )
    parser.add_argument(
        "--vlm_repo", type=str, required=True, help="VLM model repository"
    )
    parser.add_argument(
        "--llm_repo", type=str, required=True, help="LLM model repository"
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA/GPU (default: CPU)"
    )
    parser.add_argument(
        "--output_path", type=str, default="./merged_model", help="Output folder path"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha value for merging (0.0=100% LLM, 1.0=100% VLM, default: 0.5)",
    )

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("Alpha must be between 0.0 and 1.0")

    # Determine device
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device == "cpu":
        print("Warning: CUDA requested but not available, falling back to CPU")

    print(f"Device: {device} | α: {args.alpha}")
    print("-" * 50)

    vlm_model, vlm_tokenizer, vlm_processor, llm_model, model_type = load_models(
        args.vlm_repo, args.llm_repo, device
    )

    print(f"Model: {model_type.upper()}")

    merge_models(vlm_model, llm_model, model_type, args.alpha)

    save_model(vlm_model, vlm_tokenizer, vlm_processor, args.output_path)

    print("✓ Merge completed successfully!")


if __name__ == "__main__":
    main()
