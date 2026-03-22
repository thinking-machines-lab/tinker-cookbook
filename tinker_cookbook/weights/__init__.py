"""Weight lifecycle utilities for Tinker training.

Provides functions for downloading, building, and publishing trained model
weights. The typical workflow is:

    download → build → publish

Each function takes local paths as input/output, making them composable
and independently testable.

Example::

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
"""

from tinker_cookbook.weights._download import download
from tinker_cookbook.weights._export import build_hf_model
from tinker_cookbook.weights._publish import publish_to_hf_hub

__all__ = [
    "download",
    "build_hf_model",
    "publish_to_hf_hub",
]


def build_lora_adapter(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """Convert a Tinker LoRA adapter to standard LoRA format.

    The output can be loaded directly by vLLM (``--lora-modules``),
    SGLang, or any framework supporting LoRA adapters without merging
    into base model weights.

    Args:
        base_model: HuggingFace model name or local path. Needed to
            resolve model-specific weight naming conventions.
        adapter_path: Local path to the Tinker adapter directory
            (must contain ``adapter_model.safetensors`` and
            ``adapter_config.json``).
        output_path: Directory where the standard LoRA adapter will
            be saved.
    """
    raise NotImplementedError(
        "build_lora_adapter is not yet implemented. "
        "Use build_hf_model to merge the adapter into a full HF model instead."
    )
