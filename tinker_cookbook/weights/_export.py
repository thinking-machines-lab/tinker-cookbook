"""Build deployable model artifacts from Tinker weights."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from tinker_cookbook.weights._merge import merge_adapter_weights

logger = logging.getLogger(__name__)


def build_hf_model(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """Build a complete HuggingFace model from Tinker LoRA adapter weights.

    Merges the LoRA adapter into the base model and saves the result as a
    standard HuggingFace model directory, compatible with vLLM, SGLang, TGI,
    or any HuggingFace-compatible inference framework.

    Args:
        base_model: HuggingFace model name (e.g. ``"Qwen/Qwen3.5-35B-A3B"``)
            or local path to a saved HuggingFace model.
        adapter_path: Local path to the Tinker adapter directory. Must contain
            ``adapter_model.safetensors`` and ``adapter_config.json``.
        output_path: Directory where the merged model will be saved. Must not
            already exist.

    Raises:
        FileNotFoundError: If adapter files are missing.
        FileExistsError: If output_path already exists.
        KeyError: If adapter config is malformed.
        ValueError: If tensor shapes are incompatible during merge.
    """
    # Validate adapter exists before loading the (potentially huge) base model
    adapter_weights, adapter_config = _load_adapter_weights(Path(adapter_path))

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=False)

    try:
        logger.info("Loading base model: %s", base_model)
        hf_model = _load_model(base_model)

        logger.info("Merging adapter weights")
        merge_adapter_weights(hf_model, adapter_weights, adapter_config)

        logger.info("Saving merged model to: %s", out)
        hf_model.save_pretrained(out)

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.save_pretrained(out)

        try:
            processor = AutoProcessor.from_pretrained(base_model)
            processor.save_pretrained(out)
        except Exception:
            logger.debug(
                "No processor found for %s (this is expected for text-only models)", base_model
            )

        logger.info("Done — merged model saved to %s", out)
    except Exception:
        # Clean up partial output so the user can retry without manual deletion
        if out.exists():
            shutil.rmtree(out)
        raise


def _load_model(model_path: str) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_path)
    auto_cls = AutoModelForImageTextToText if "vision_config" in config else AutoModelForCausalLM
    kwargs: dict = {"dtype": torch.bfloat16}
    try:
        import accelerate  # noqa: F401

        kwargs["device_map"] = "auto"
    except ImportError:
        pass
    return auto_cls.from_pretrained(model_path, **kwargs)


def _load_adapter_weights(adapter_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    adapter_dir = adapter_dir.expanduser().resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {safetensors_path}")

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")

    weights = load_file(str(safetensors_path), device=device)
    with open(config_path) as f:
        config = json.load(f)
    return weights, config
