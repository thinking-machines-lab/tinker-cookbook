"""Weight lifecycle utilities for Tinker training.

Provides functions for downloading, building, and publishing trained model
weights. Two build paths are available:

- :func:`build_hf_model` — merge LoRA adapter into a full HF model
- :func:`build_lora_adapter` — convert to PEFT format for serving with
  vLLM / SGLang (no merging, lightweight output)

Each function takes local paths as input/output, making them composable
and independently testable.

Example — merged model::

    from tinker_cookbook import weights

    adapter_dir = weights.download(
        tinker_path="tinker://run-id/sampler_weights/final",
        output_dir="./adapter",
    )
    weights.build_hf_model(
        base_model="Qwen/Qwen3.5-35B-A3B",
        adapter_path=adapter_dir,
        output_path="./model",
    )
    weights.publish_to_hf_hub(model_path="./model", repo_id="user/my-finetuned-model")

Example — PEFT adapter for serving::

    weights.build_lora_adapter(
        base_model="Qwen/Qwen3-8B",
        adapter_path=adapter_dir,
        output_path="./peft_adapter",
    )
    # Then serve with: vllm serve Qwen/Qwen3-8B --lora-modules my_adapter=./peft_adapter
"""

from tinker_cookbook.weights._adapter import build_lora_adapter
from tinker_cookbook.weights._download import download
from tinker_cookbook.weights._export import build_hf_model
from tinker_cookbook.weights._model_card import ModelCardConfig, generate_model_card
from tinker_cookbook.weights._publish import publish_to_hf_hub

__all__ = [
    "ModelCardConfig",
    "build_hf_model",
    "build_lora_adapter",
    "download",
    "generate_model_card",
    "publish_to_hf_hub",
]
