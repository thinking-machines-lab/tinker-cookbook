"""Build deployable model artifacts from Tinker weights.

Provides :func:`build_hf_model`, the main entry point for merging a Tinker
LoRA adapter into a HuggingFace model. Supports multiple merge strategies:

- ``"full"`` — loads the entire base model into memory (original behavior)
- ``"shard"`` — processes one safetensors shard at a time (low memory)
- ``"auto"`` (default) — uses shard-by-shard

Model-specific export strategies (e.g. DeepSeek FP8) live in their own
submodules and are dispatched automatically based on ``config.json``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
)

logger = logging.getLogger(__name__)

# Map user-facing dtype strings to torch dtypes.
_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_VALID_STRATEGIES = {"auto", "shard", "full"}
_VALID_QUANTIZE = {"experts-fp8"}
_VALID_SERVING_FORMATS = {"vllm"}


def build_hf_model(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    dtype: str = "bfloat16",
    trust_remote_code: bool | None = None,
    merge_strategy: str = "auto",
    dequantize: bool = False,
    quantize: str | None = None,
    serving_format: str | None = None,
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
            for maximum precision during merge. Only used by
            ``merge_strategy="full"``; the shard strategy preserves the
            on-disk dtype of each tensor.
        trust_remote_code: Whether to trust remote code when loading HF
            models. Required for some newer model architectures (e.g.
            Qwen3.5). If ``None`` (default), falls back to the
            ``HF_TRUST_REMOTE_CODE`` environment variable, then ``False``.
        merge_strategy: Controls how the merge is performed. ``"auto"``
            (default) uses shard-by-shard processing for lower peak memory.
            ``"shard"`` forces shard-by-shard (fails if shards can't be
            resolved). ``"full"`` forces full-model loading (original
            behavior, higher memory but simpler).
        dequantize: If ``True``, dequantize quantized base model weights
            before merging. Not yet implemented for the standard merge path,
            but used internally by the quantized export path for models with
            native FP8 weights (e.g. DeepSeek V3.1).
        quantize: Output quantization method. Currently supported:
            ``"experts-fp8"`` — quantize routed expert weights to FP8 with
            blockwise scaling. Requires ``serving_format`` to be set.
            ``None`` (default) — no quantization.
        serving_format: Serving framework format for quantization metadata.
            Currently supported: ``"vllm"`` — write compressed-tensors
            config for vLLM. Required when ``quantize`` is set.
            ``None`` (default) — no serving-specific metadata.

    Raises:
        FileNotFoundError: If adapter files are missing.
        FileExistsError: If output_path already exists.
        KeyError: If adapter config is malformed.
        ValueError: If tensor shapes are incompatible during merge, or
            if ``dtype``, ``merge_strategy``, ``quantize``, or
            ``serving_format`` is not a recognized value, or if
            ``quantize`` and ``serving_format`` are not both set/unset.
        NotImplementedError: If ``dequantize=True`` on the standard merge path.
    """
    # --- Validate quantize / serving_format ---
    if quantize is not None and quantize not in _VALID_QUANTIZE:
        raise ValueError(
            f"Unsupported quantize={quantize!r}. Choose from: {sorted(_VALID_QUANTIZE)}"
        )
    if serving_format is not None and serving_format not in _VALID_SERVING_FORMATS:
        raise ValueError(
            f"Unsupported serving_format={serving_format!r}. "
            f"Choose from: {sorted(_VALID_SERVING_FORMATS)}"
        )
    if quantize is not None and serving_format is None:
        raise ValueError(
            f"quantize={quantize!r} requires serving_format to be set "
            f"(e.g. serving_format='vllm') to write scale metadata."
        )
    if serving_format is not None and quantize is None:
        raise ValueError(
            f"serving_format={serving_format!r} requires quantize to be set "
            f"(e.g. quantize='experts-fp8'). Serving format without quantization is meaningless."
        )
    if quantize == "experts-fp8" and dtype != "bfloat16":
        raise ValueError(f"quantize='experts-fp8' requires dtype='bfloat16', got dtype={dtype!r}.")

    # --- Validate standard params ---
    if dequantize and quantize is None:
        raise NotImplementedError(
            "dequantize is not yet supported for the standard merge path. "
            "Use quantize='experts-fp8' for models with native FP8 weights."
        )
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {dtype!r}. Choose from: {list(_DTYPE_MAP.keys())}")
    if merge_strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unsupported merge_strategy {merge_strategy!r}. "
            f"Choose from: {sorted(_VALID_STRATEGIES)}"
        )

    resolved_trust = resolve_trust_remote_code(trust_remote_code)

    # Load model config for model-family detection (lightweight, no weight download).
    config_dict = load_config_dict(base_model)

    # --- Quantized export path ---
    if quantize is not None:
        from tinker_cookbook.weights._artifacts import resolve_model_dir
        from tinker_cookbook.weights._export._quantized import build_quantized

        model_dir = resolve_model_dir(base_model)
        build_quantized(
            base_model=base_model,
            adapter_path=adapter_path,
            output_path=output_path,
            trust_remote_code=resolved_trust,
            model_dir=model_dir,
            config_dict=config_dict,
            serving_format=serving_format,  # type: ignore[arg-type]  # validated non-None above
        )
        return

    # --- Standard merge path ---
    strategy = _resolve_strategy(merge_strategy)

    if strategy == "full":
        from tinker_cookbook.weights._export._full import build_full

        build_full(
            base_model=base_model,
            adapter_path=adapter_path,
            output_path=output_path,
            dtype=dtype,
            torch_dtype=_DTYPE_MAP[dtype],
            trust_remote_code=resolved_trust,
            config_dict=config_dict,
        )
    else:
        if dtype != "bfloat16":
            logger.warning(
                "dtype=%r only applies to merge_strategy='full'. "
                "The shard strategy preserves each tensor's on-disk dtype. "
                "Pass merge_strategy='full' to control output precision.",
                dtype,
            )

        from tinker_cookbook.weights._artifacts import resolve_model_dir
        from tinker_cookbook.weights._export._shard import build_sharded

        model_dir = resolve_model_dir(base_model)
        build_sharded(
            base_model=base_model,
            adapter_path=adapter_path,
            output_path=output_path,
            trust_remote_code=resolved_trust,
            model_dir=model_dir,
            config_dict=config_dict,
        )


def _resolve_strategy(merge_strategy: str) -> str:
    """Resolve ``"auto"`` to a concrete strategy."""
    if merge_strategy == "auto":
        return "shard"
    return merge_strategy


# ---------------------------------------------------------------------------
# Shared helpers used by export strategy modules
# ---------------------------------------------------------------------------


def resolve_trust_remote_code(trust_remote_code: bool | None) -> bool:
    """Resolve trust_remote_code from parameter or environment variable.

    Priority: explicit parameter > HF_TRUST_REMOTE_CODE env var > False.
    """
    if trust_remote_code is not None:
        return trust_remote_code
    env_val = os.environ.get("HF_TRUST_REMOTE_CODE", "").lower()
    return env_val in ("1", "true", "yes")


def load_config_dict(model_dir_or_name: str | Path) -> dict:
    """Load config.json as a raw dict from a local directory or HF model name.

    For local directories, reads config.json directly. For HF model names
    (not a local directory), falls back to ``AutoConfig.from_pretrained``.

    Raises:
        FileNotFoundError: If ``model_dir_or_name`` is a local directory
            that doesn't contain ``config.json``.
    """
    model_dir = (
        Path(model_dir_or_name) if not isinstance(model_dir_or_name, Path) else model_dir_or_name
    )
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    # If it's a local directory without config.json, fail explicitly
    if model_dir.is_dir():
        raise FileNotFoundError(
            f"No config.json found in {model_dir}. "
            f"Ensure this is a valid HuggingFace model directory."
        )
    # Fall back to HF config loading for remote model names
    config = AutoConfig.from_pretrained(str(model_dir_or_name))
    return config.to_dict()


def is_multimodal(config: PretrainedConfig) -> bool:
    """Check if a model config indicates a multimodal (e.g. vision-language) model."""
    multimodal_config_keys = ("vision_config", "audio_config", "speech_config")
    return any(
        hasattr(config, key) and getattr(config, key) is not None for key in multimodal_config_keys
    )


def is_multimodal_from_dict(config_dict: dict) -> bool:
    """Check if a raw config dict indicates a multimodal model."""
    multimodal_keys = ("vision_config", "audio_config", "speech_config")
    return any(config_dict.get(key) is not None for key in multimodal_keys)


def save_tokenizer_and_processor(
    base_model: str,
    output_path: Path,
    multimodal: bool,
    trust_remote_code: bool,
) -> None:
    """Save tokenizer and optional processor to the output directory."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_path)

    if multimodal:
        try:
            processor = AutoProcessor.from_pretrained(
                base_model, trust_remote_code=trust_remote_code
            )
            processor.save_pretrained(output_path)
        except (OSError, ValueError) as e:
            logger.warning(
                "Could not load processor for vision model %s: %s. "
                "You may need to copy the processor files manually.",
                base_model,
                e,
            )


def cleanup_on_failure(out: Path) -> None:
    """Clean up partial output so the user can retry without manual deletion."""
    try:
        if out.exists():
            shutil.rmtree(out)
    except OSError:
        logger.warning("Failed to clean up partial output at %s", out)
