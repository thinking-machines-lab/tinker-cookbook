"""Merge Tinker adapter weights to a HuggingFace model, and save the new model to a given path.

Please refer to the following documentation for how to download a Tinker sampler adapter weights: https://tinker-docs.thinkingmachines.ai/download-weights

Usage:
python merge_tinker_adapter_to_hf_model.py --hf-model <name_or_path_to_hf_model> --tinker-adapter-path <local_path_to_tinker_adapter_weights> --output-path <output_path_to_save_merged_model>

NOTE: This script is a thin CLI wrapper around tinker_cookbook.weights.build_hf_model().
For programmatic use, prefer importing from tinker_cookbook.weights directly.
"""

import argparse

from tinker_cookbook.weights import build_hf_model


def main():
    parser = argparse.ArgumentParser(
        description="Merge Tinker LoRA adapter weights into a HuggingFace model."
    )
    parser.add_argument(
        "--tinker-adapter-path", type=str, required=True, help="Path to the Tinker adapter"
    )
    parser.add_argument(
        "--hf-model", type=str, required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save the merged model"
    )
    args = parser.parse_args()

    build_hf_model(
        base_model=args.hf_model,
        adapter_path=args.tinker_adapter_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
