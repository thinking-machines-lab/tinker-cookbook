"""Build deployable model artifacts from Tinker weights."""

from __future__ import annotations

import json
import logging
import os

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
    """
    os.makedirs(output_path, exist_ok=False)

    logger.info("Loading base model: %s", base_model)
    hf_model = _load_model(base_model)

    logger.info("Loading adapter weights from: %s", adapter_path)
    adapter_weights, adapter_config = _load_adapter_weights(adapter_path)

    logger.info("Merging adapter weights")
    merge_adapter_weights(hf_model, adapter_weights, adapter_config)

    logger.info("Saving merged model to: %s", output_path)
    hf_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)

    try:
        processor = AutoProcessor.from_pretrained(base_model)
        processor.save_pretrained(output_path)
    except Exception:
        pass

    logger.info("Done — merged model saved to %s", output_path)


def _load_model(model_path: str) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_path)
    auto_cls = AutoModelForImageTextToText if "vision_config" in config else AutoModelForCausalLM
    kwargs: dict = {"dtype": torch.bfloat16}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return auto_cls.from_pretrained(model_path, **kwargs)


def _load_adapter_weights(adapter_path: str) -> tuple[dict[str, torch.Tensor], dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = load_file(
        os.path.expanduser(os.path.join(adapter_path, "adapter_model.safetensors")),
        device=device,
    )
    config_path = os.path.expanduser(os.path.join(adapter_path, "adapter_config.json"))
    with open(config_path) as f:
        config = json.load(f)
    return weights, config
