import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
import torch
from tqdm import tqdm


def load_models(vlm_repo, llm_repo):
    """Load VLM and LLM models and tokenizers."""
    print("Loading models...")
    vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_repo)
    vlm_model = AutoModelForImageTextToText.from_pretrained(vlm_repo, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_repo)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_repo, dtype=torch.bfloat16, device_map="auto")
    
    return vlm_model, vlm_tokenizer, llm_model, llm_tokenizer


def merge_models(vlm_model, llm_model, alpha=0.5):
    """Merge VLM and LLM models using linear combination (alpha * VLM + (1-alpha) * LLM)."""
    print(f"Merging models with alpha={alpha} (VLM: {alpha*100:.0f}%, LLM: {(1-alpha)*100:.0f}%)...")
    
    llm_weights = {}
    for name, param in llm_model.named_parameters():
        if name.startswith("model.") or name == "lm_head.weight":
            llm_weights[name] = param

    vlm_params_to_merge = []
    for name, param in vlm_model.named_parameters():
        if name.startswith("model.language_model.") or name == "lm_head.weight":
            vlm_params_to_merge.append((name, param))

    mismatches = 0
    mismatch_details = []
    with torch.no_grad():
        for name, param in tqdm(vlm_params_to_merge, desc="Merging weights"):
            if name.startswith("model.language_model."):
                llm_name = name.replace("model.language_model.", "model.")
            else:  # lm_head.weight
                llm_name = name
            if llm_name in llm_weights:
                llm_param = llm_weights[llm_name]
                if param.shape == llm_param.shape:
                    llm_param = llm_param.to(device=param.device, dtype=param.dtype)
                    param.data = alpha * param.data + (1 - alpha) * llm_param.data
                else:
                    mismatches += 1
                    mismatch_details.append(f"Shape mismatch: VLM {name} {param.shape} vs LLM {llm_name} {llm_param.shape}")
            else:
                mismatches += 1
                mismatch_details.append(f"Missing in LLM: {name} -> {llm_name}")
    
    if mismatches > 0:
        print(f"Warning: {mismatches} parameters were not merged due to mismatches.")
        print("Mismatch details:")
        for detail in mismatch_details:
            print(f"  - {detail}")
    
    return mismatches


def save_model(vlm_model, vlm_tokenizer, output_path):
    """Save the merged model and tokenizer to the specified path."""
    print("Saving model...")
    os.makedirs(output_path, exist_ok=True)
    vlm_model.save_pretrained(output_path, safe_serialization=True)
    vlm_tokenizer.save_pretrained(output_path)


def push_to_huggingface(output_path, hf_repo_id, private=True):
    """Push the merged model to Hugging Face Hub."""
    from huggingface_hub import HfApi
    
    print(f"Pushing to Hugging Face: {hf_repo_id}")
    
    api = HfApi()
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        private=private,
        exist_ok=True
    )
    api.upload_folder(
        folder_path=output_path,
        repo_id=hf_repo_id,
        repo_type="model"
    )

def main():
    parser = argparse.ArgumentParser(description="Merge VLM and LLM models")
    parser.add_argument("--vlm_repo", type=str, required=True, help="VLM model repository")
    parser.add_argument("--llm_repo", type=str, required=True, help="LLM model repository")
    parser.add_argument("--output_path", type=str, default="./merged_model", help="Output folder path")
    parser.add_argument("--hf_repo_id", type=str, help="Hugging Face repository ID to push merged model")
    parser.add_argument("--public", action="store_true", help="Make Hugging Face repository public (default: private)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for merging (0.0=100% LLM, 1.0=100% VLM, default: 0.5)")
    
    args = parser.parse_args()
    
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("Alpha must be between 0.0 and 1.0")
    
    vlm_model, vlm_tokenizer, llm_model, llm_tokenizer = load_models(args.vlm_repo, args.llm_repo)
    
    mismatches = merge_models(vlm_model, llm_model, args.alpha)
    
    save_model(vlm_model, vlm_tokenizer, args.output_path)
    
    if args.hf_repo_id:
        push_to_huggingface(args.output_path, args.hf_repo_id, private=not args.public)
    
    print("Done!")


if __name__ == "__main__":
    main()
