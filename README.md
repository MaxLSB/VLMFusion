# VLMFusion

A tool for merging Vision Language Models (VLM) and Large Language Models (LLM), if they share the same language model backbone, using linear combination.

## Setup

Create a virtual environment using `uv`:

```bash
uv venv myenv
source myenv/bin/activate
uv pip install -r requirements.txt
```

## Merging

The `linear.py` script merges VLM and LLM models using linear combination. The output model is a VLM. Currently supports:

- **Mistral models**
- **LFM2 models**

Basic examples:

```bash
source myenv/bin/activate
python linear.py --vlm_repo "mistralai/Mistral-Small-3.1-24B-Instruct-2503" --llm_repo "MaxLSB/Mistral-Small-24B-Instruct-merged-checkpoint-3648" --output_path "./merged_model" --alpha 0.7
```

_Note: Alpha is the merging coefficient for the VLM, so 0.0=100% LLM, 1.0=100% VLM (default: 0.5). Use `--cuda` for GPU acceleration._
