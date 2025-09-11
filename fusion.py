#!/usr/bin/env python3
"""
VLM-LLM Fusion Tool using mergekit

This script merges a Vision-Language Model (VLM) with a Large Language Model (LLM)
while preserving the vision components (vision tower and projector) of the VLM
and using mergekit to merge only the language model components.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import mergekit
from mergekit.merge import MergeOptions
from mergekit.config import MergeConfiguration


class VLMFusion:
    """
    A class to handle the fusion of VLM and LLM models using mergekit.
    """

    def __init__(self, vlm_path: str, llm_path: str, output_path: str):
        """
        Initialize the VLM-LLM fusion process.

        Args:
            vlm_path: Path to the VLM model
            llm_path: Path to the LLM model
            output_path: Path where the fused model will be saved
        """
        self.vlm_path = vlm_path
        self.llm_path = llm_path
        self.output_path = output_path
        self.temp_dir = None

    def identify_model_components(self, model_path: str) -> Dict[str, List[str]]:
        """
        Identify the different components of a model (vision, language, etc.).

        Args:
            model_path: Path to the model

        Returns:
            Dictionary mapping component types to lists of layer names
        """
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True
        )

        components = {
            "vision_tower": [],
            "projector": [],
            "language_model": [],
            "other": [],
        }

        # Get all parameter names
        param_names = list(model.state_dict().keys())

        # Common patterns for different components
        vision_patterns = ["vision", "visual", "image", "clip", "vit"]
        projector_patterns = ["projector", "proj", "mm_projector", "vision_proj"]
        language_patterns = ["lm_head", "embed", "transformer", "model", "layers"]

        for param_name in param_names:
            param_lower = param_name.lower()

            # Check for vision components
            if any(pattern in param_lower for pattern in vision_patterns):
                components["vision_tower"].append(param_name)
            # Check for projector components
            elif any(pattern in param_lower for pattern in projector_patterns):
                components["projector"].append(param_name)
            # Check for language model components
            elif any(pattern in param_lower for pattern in language_patterns):
                components["language_model"].append(param_name)
            else:
                components["other"].append(param_name)

        print(f"Identified components for {model_path}:")
        for comp_type, layers in components.items():
            if layers:
                print(f"  {comp_type}: {len(layers)} layers")

        return components

    def create_merge_config(
        self, vlm_components: Dict, llm_components: Dict, merge_ratio: float = 0.5
    ) -> MergeConfiguration:
        """
        Create a mergekit configuration for merging VLM and LLM language components.

        Args:
            vlm_components: VLM component mapping
            llm_components: LLM component mapping
            merge_ratio: Ratio for linear merge (0.0 = all VLM, 1.0 = all LLM)

        Returns:
            MergeConfiguration object
        """
        # Create temporary directory for intermediate models
        self.temp_dir = tempfile.mkdtemp(prefix="vlm_fusion_")

        # Save VLM language model components separately
        vlm_lm_path = os.path.join(self.temp_dir, "vlm_lm")
        self._extract_language_model(
            self.vlm_path, vlm_lm_path, vlm_components["language_model"]
        )

        # Create merge configuration
        merge_config = {
            "models": [
                {"model": vlm_lm_path, "parameters": {"weight": 1.0 - merge_ratio}},
                {"model": self.llm_path, "parameters": {"weight": merge_ratio}},
            ],
            "merge_method": "linear",
            "base_model": vlm_lm_path,
            "parameters": {"normalize": True},
        }

        return MergeConfiguration.model_validate(merge_config)

    def _extract_language_model(
        self, source_path: str, target_path: str, language_layers: List[str]
    ):
        """
        Extract only the language model components from a VLM.

        Args:
            source_path: Path to source VLM
            target_path: Path to save extracted language model
            language_layers: List of language model layer names
        """
        print(f"Extracting language model components from {source_path}...")

        # Load the full model
        config = AutoConfig.from_pretrained(source_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            source_path, torch_dtype=torch.float16, trust_remote_code=True
        )

        # Create a new model with only language components
        state_dict = model.state_dict()
        language_state_dict = {
            k: v for k, v in state_dict.items() if k in language_layers
        }

        # Save the language model components
        os.makedirs(target_path, exist_ok=True)
        torch.save(language_state_dict, os.path.join(target_path, "pytorch_model.bin"))
        config.save_pretrained(target_path)

        print(f"Language model components saved to {target_path}")

    def merge_models(self, merge_ratio: float = 0.5) -> str:
        """
        Perform the actual merge of VLM and LLM models.

        Args:
            merge_ratio: Ratio for linear merge (0.0 = all VLM, 1.0 = all LLM)

        Returns:
            Path to the merged model
        """
        print("Starting VLM-LLM fusion process...")

        # Identify components
        print("Analyzing VLM components...")
        vlm_components = self.identify_model_components(self.vlm_path)

        print("Analyzing LLM components...")
        llm_components = self.identify_model_components(self.llm_path)

        # Create merge configuration
        print("Creating merge configuration...")
        merge_config = self.create_merge_config(
            vlm_components, llm_components, merge_ratio
        )

        # Create temporary output directory
        temp_output = os.path.join(self.temp_dir, "merged_lm")

        # Perform the merge
        print("Merging language model components...")
        merge_options = MergeOptions(
            allow_crimes=True,
            transformers_cache_dir=None,
            lora_merge_cache=None,
            cuda=False,
            low_cpu_memory=False,
            out_shard_size=5_000_000_000,
            write_safetensors=False,
            random_seed=42,
            trust_remote_code=True,
        )

        try:
            mergekit.merge(merge_config, temp_output, merge_options)
            print(f"Language model merge completed: {temp_output}")
        except Exception as e:
            print(f"Error during merge: {e}")
            raise

        # Now combine with VLM vision components
        print("Combining merged language model with VLM vision components...")
        self._combine_with_vision_components(temp_output, vlm_components)

        # Clean up temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

        print(f"VLM-LLM fusion completed! Output saved to: {self.output_path}")
        return self.output_path

    def _combine_with_vision_components(
        self, merged_lm_path: str, vlm_components: Dict
    ):
        """
        Combine the merged language model with the original VLM vision components.

        Args:
            merged_lm_path: Path to the merged language model
            vlm_components: VLM component mapping
        """
        # Load VLM and merged language model
        vlm_model = AutoModel.from_pretrained(
            self.vlm_path, torch_dtype=torch.float16, trust_remote_code=True
        )
        merged_lm = AutoModel.from_pretrained(
            merged_lm_path, torch_dtype=torch.float16, trust_remote_code=True
        )

        # Get state dictionaries
        vlm_state = vlm_model.state_dict()
        merged_lm_state = merged_lm.state_dict()

        # Create final state dictionary
        final_state = {}

        # Add vision components from VLM
        for layer_name in vlm_components["vision_tower"] + vlm_components["projector"]:
            if layer_name in vlm_state:
                final_state[layer_name] = vlm_state[layer_name]
                print(f"Added vision component: {layer_name}")

        # Add merged language model components
        for layer_name, tensor in merged_lm_state.items():
            final_state[layer_name] = tensor
            print(f"Added merged language component: {layer_name}")

        # Add any other VLM components that weren't categorized
        for layer_name in vlm_components["other"]:
            if layer_name in vlm_state and layer_name not in final_state:
                final_state[layer_name] = vlm_state[layer_name]
                print(f"Added other component: {layer_name}")

        # Save the final model
        os.makedirs(self.output_path, exist_ok=True)

        # Save state dict
        torch.save(final_state, os.path.join(self.output_path, "pytorch_model.bin"))

        # Copy config and tokenizer from VLM (as it has the complete architecture)
        vlm_config = AutoConfig.from_pretrained(self.vlm_path, trust_remote_code=True)
        vlm_config.save_pretrained(self.output_path)

        # Copy tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.vlm_path, trust_remote_code=True
            )
            tokenizer.save_pretrained(self.output_path)
        except:
            print("Warning: Could not copy tokenizer from VLM")

        print(f"Final fused model saved to: {self.output_path}")


def main():
    """Main function to run the VLM-LLM fusion tool."""
    parser = argparse.ArgumentParser(
        description="Fuse VLM and LLM models using mergekit"
    )
    parser.add_argument("--vlm", required=True, help="Path to the VLM model")
    parser.add_argument("--llm", required=True, help="Path to the LLM model")
    parser.add_argument(
        "--output", required=True, help="Output path for the fused model"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Merge ratio (0.0 = all VLM, 1.0 = all LLM)",
    )

    args = parser.parse_args()

    # Validate inputs
    def is_valid_model_path(model_path: str) -> bool:
        """Check if model path is valid (local path or Hugging Face model ID)."""
        if os.path.exists(model_path):
            return True
        # Check if it's a valid Hugging Face model ID by trying to load config
        try:
            AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            return True
        except:
            return False

    if not is_valid_model_path(args.vlm):
        print(f"Error: VLM model path is invalid: {args.vlm}")
        return 1

    if not is_valid_model_path(args.llm):
        print(f"Error: LLM model path is invalid: {args.llm}")
        return 1

    if not 0.0 <= args.ratio <= 1.0:
        print("Error: Merge ratio must be between 0.0 and 1.0")
        return 1

    # Create fusion instance and run
    try:
        fusion = VLMFusion(args.vlm, args.llm, args.output)
        result_path = fusion.merge_models(args.ratio)
        print(f"\n✅ Fusion completed successfully!")
        print(f"📁 Output model: {result_path}")
        return 0
    except Exception as e:
        print(f"\n❌ Fusion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
