"""Full-model export strategy.

Loads the entire base model into memory, merges LoRA adapter weights in-place,
and saves via ``model.save_pretrained()``. This is the original merge behavior
and serves as the fallback when shard-by-shard processing isn't suitable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PretrainedConfig,
    PreTrainedModel,
)

from tinker_cookbook.weights._artifacts import load_adapter_weights
from tinker_cookbook.weights._export import (
    cleanup_on_failure,
    is_multimodal,
    save_tokenizer_and_processor,
)
from tinker_cookbook.weights._merge import merge_adapter_weights

logger = logging.getLogger(__name__)


def build_full(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    dtype: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> None:
    """Merge by loading the entire base model into memory."""
    # Validate adapter exists before loading the (potentially huge) base model
    adapter_weights, adapter_config = load_adapter_weights(Path(adapter_path))

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=False)

    try:
        logger.info("Loading base model: %s (dtype=%s)", base_model, dtype)
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        hf_model = _load_model(
            config, base_model, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
        )

        logger.info("Merging adapter weights")
        merge_adapter_weights(hf_model, adapter_weights, adapter_config)

        logger.info("Saving merged model to: %s", out)
        hf_model.save_pretrained(out)

        save_tokenizer_and_processor(base_model, out, is_multimodal(config), trust_remote_code)

        logger.info("Done — merged model saved to %s", out)
    except Exception:
        cleanup_on_failure(out)
        raise


def _load_model(
    config: PretrainedConfig,
    model_path: str,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> PreTrainedModel:
    auto_cls = AutoModelForImageTextToText if is_multimodal(config) else AutoModelForCausalLM
    return auto_cls.from_pretrained(
        model_path, dtype=torch_dtype, trust_remote_code=trust_remote_code
    )
