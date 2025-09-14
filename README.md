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

- **Mistral models**: Auto-detects from repository name containing "mistral"
- **LFM2 models**: Auto-detects from repository name containing "lfm2"

/!\ The merged model is saved locally. It won't include specific (potentially needed) files like `preprocessor_config.json`,  `processor_config.json`, `tekken.json` or `params.json`.

Basic examples:

```bash
# Mistral models (CPU)
source myenv/bin/activate
python linear.py --vlm_repo "mistralai/Mistral-Small-3.1-24B-Instruct-2503" --llm_repo "MaxLSB/Mistral-Small-24B-Instruct-merged-checkpoint-3648" --output_path "./merged_model" --alpha 0.7

# LFM2 models (GPU)
python linear.py --vlm_repo "LiquidAI/LFM2-VL-450M" --llm_repo "kurakurai/Luth-LFM2-350M" --cuda --alpha 0.5
```

_Note: Alpha is the merging coefficient for the VLM, so 0.0=100% LLM, 1.0=100% VLM (default: 0.5). Use `--cuda` for GPU acceleration._