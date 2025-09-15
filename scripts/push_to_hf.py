from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import create_repo

# Define your repo and local folder
repo_id = "lightonai/Mistral-VLM-French-0.9"
folder_path = "merged_model"

# Create repo if it doesn't exist
create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

# Load the VLM components from the local checkpoint
tokenizer = AutoTokenizer.from_pretrained(folder_path)
processor = AutoProcessor.from_pretrained(folder_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    folder_path, torch_dtype="bfloat16", trust_remote_code=True
)

# Push all components to the Hub
tokenizer.push_to_hub(repo_id)
processor.push_to_hub(repo_id)
model.push_to_hub(repo_id)
