"""Build deployable model artifacts from Tinker weights."""

from __future__ import annotations

import json
import logging
import os
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
    PretrainedConfig,
    PreTrainedModel,
)

from tinker_cookbook.weights._deepseek import (
    is_deepseek_config as _is_deepseek_config,
)
from tinker_cookbook.weights._deepseek import (
    save_deepseek_model as _save_deepseek_model,
)
from tinker_cookbook.weights._merge import merge_adapter_weights

logger = logging.getLogger(__name__)

# Map user-facing dtype strings to torch dtypes.
_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_trust_remote_code(trust_remote_code: bool | None) -> bool:
    """Resolve trust_remote_code from parameter or environment variable.

    Priority: explicit parameter > HF_TRUST_REMOTE_CODE env var > False.
    """
    if trust_remote_code is not None:
        return trust_remote_code
    env_val = os.environ.get("HF_TRUST_REMOTE_CODE", "").lower()
    return env_val in ("1", "true", "yes")


def build_hf_model(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    dtype: str = "bfloat16",
    trust_remote_code: bool | None = None,
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
        dtype: Data type for loading the base model. One of ``"bfloat16"``
            (default), ``"float16"``, or ``"float32"``. Use ``"float32"``
            for maximum precision during merge.
        trust_remote_code: Whether to trust remote code when loading HF
            models. Required for some newer model architectures (e.g.
            Qwen3.5). If ``None`` (default), falls back to the
            ``HF_TRUST_REMOTE_CODE`` environment variable, then ``False``.

    Raises:
        FileNotFoundError: If adapter files are missing.
        FileExistsError: If output_path already exists.
        KeyError: If adapter config is malformed.
        ValueError: If tensor shapes are incompatible during merge, or
            if ``dtype`` is not a recognized value.
    """
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {dtype!r}. Choose from: {list(_DTYPE_MAP.keys())}")
    torch_dtype = _DTYPE_MAP[dtype]
    resolved_trust = _resolve_trust_remote_code(trust_remote_code)
    out = Path(output_path)
    preserve_partial_output = _is_deepseek_model_path(base_model)
    out.mkdir(parents=True, exist_ok=preserve_partial_output)
    is_multimodal = False

    try:
        if preserve_partial_output:
            logger.info("Detected DeepSeek model; using DeepSeek-specific save path")
            logger.info("Saving merged model to: %s", out)
            _save_deepseek_model(
                base_model=base_model,
                adapter_path=adapter_path,
                output_path=out,
            )
        else:
            # Validate adapter exists before loading the (potentially huge) base model
            adapter_weights, adapter_config = _load_adapter_weights(Path(adapter_path))

            logger.info("Loading base model: %s (dtype=%s)", base_model, dtype)
            config = AutoConfig.from_pretrained(base_model, trust_remote_code=resolved_trust)
            is_multimodal = _is_multimodal(config)
            hf_model = _load_model(
                config, base_model, torch_dtype=torch_dtype, trust_remote_code=resolved_trust
            )

            logger.info("Merging adapter weights")
            merge_adapter_weights(hf_model, adapter_weights, adapter_config)

            logger.info("Saving merged model to: %s", out)
            hf_model.save_pretrained(out)

        logger.info("Saving tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=resolved_trust)
        tokenizer.save_pretrained(out)

        if is_multimodal:
            try:
                logger.info("Saving processor")
                processor = AutoProcessor.from_pretrained(
                    base_model, trust_remote_code=resolved_trust
                )
                processor.save_pretrained(out)
            except (OSError, ValueError) as e:
                logger.warning(
                    "Could not load processor for vision model %s: %s. "
                    "You may need to copy the processor files manually.",
                    base_model,
                    e,
                )

        logger.info("Done — merged model saved to %s", out)
    except Exception:
        if out.exists() and preserve_partial_output:
            logger.warning(
                "DeepSeek export failed; preserving partial output at %s for inspection or resume",
                out,
            )
        # Clean up partial output so the user can retry without manual deletion
        elif out.exists():
            try:
                shutil.rmtree(out)
            except OSError:
                logger.warning("Failed to clean up partial output at %s", out)
        raise


def _is_multimodal(config: PretrainedConfig) -> bool:
    """Check if a model config indicates a multimodal (e.g. vision-language) model.

    Checks for known multimodal config patterns. This is more robust than
    checking a single key, and should be extended as Tinker adds support
    for new multimodal model types.
    """
    multimodal_config_keys = ("vision_config", "audio_config", "speech_config")
    return any(
        hasattr(config, key) and getattr(config, key) is not None for key in multimodal_config_keys
    )


def _load_model(
    config: PretrainedConfig,
    model_path: str,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> PreTrainedModel:
    auto_cls = AutoModelForImageTextToText if _is_multimodal(config) else AutoModelForCausalLM
    return auto_cls.from_pretrained(
        model_path, dtype=torch_dtype, trust_remote_code=trust_remote_code
    )


def _is_deepseek_model_path(model_path: str) -> bool:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return _is_deepseek_config(config)


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
