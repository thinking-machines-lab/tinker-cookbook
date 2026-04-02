"""Quantized export strategy.

Merges LoRA adapters shard-by-shard and quantizes routed expert weights to FP8.
Produces output compatible with vLLM's compressed-tensors format.

Currently supports DeepSeek V3/V3.1 models. The infrastructure (FP8 math, vLLM
config generation, resume support) is reusable for future model families.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tinker_cookbook.exceptions import WeightsMergeError
from tinker_cookbook.weights._artifacts import (
    copy_artifact_file,
    copy_model_code_files,
    get_model_state_shapes,
    get_shard_files,
    load_adapter_weights,
)
from tinker_cookbook.weights._export import (
    is_multimodal_from_dict,
    save_tokenizer_and_processor,
)
from tinker_cookbook.weights._merge import (
    apply_merge_op,
    detect_merge_profile,
    plan_merge_ops,
    validate_merge_op_shapes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek detection
# ---------------------------------------------------------------------------

_DEEPSEEK_MODEL_TYPES = frozenset({"deepseek_v3"})


def is_deepseek_config(config_dict: dict) -> bool:
    """Check if config describes a DeepSeek model family."""
    return config_dict.get("model_type") in _DEEPSEEK_MODEL_TYPES


# ---------------------------------------------------------------------------
# FP8 blockwise quantization
# ---------------------------------------------------------------------------

# DeepSeek V3/V3.1 native FP8 block size
_FP8_BLOCK_SIZE = 128


def _get_fp8_max() -> float:
    """Get max representable value in float8_e4m3fn, with fallback for older PyTorch."""
    try:
        return float(torch.finfo(torch.float8_e4m3fn).max)
    except TypeError:
        return 448.0


_FP8_MAX = _get_fp8_max()


def quantize_blockwise(
    tensor: torch.Tensor,
    block_size: tuple[int, int] = (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to FP8 using blockwise scaling.

    Divides the tensor into blocks, computes a per-block scale factor, and
    quantizes each block to float8_e4m3fn.

    Args:
        tensor: 2D float tensor to quantize.
        block_size: (row_block, col_block) sizes. Tensor is padded if dimensions
            are not evenly divisible.

    Returns:
        Tuple of (quantized_fp8, scale_inv) where:
        - quantized_fp8: float8_e4m3fn tensor, same shape as input
        - scale_inv: float32 tensor of shape (ceil(rows/row_block), ceil(cols/col_block))
    """
    assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D"
    rows, cols = tensor.shape
    rb, cb = block_size

    # Pad to block boundaries
    pad_rows = (rb - rows % rb) % rb
    pad_cols = (cb - cols % cb) % cb
    if pad_rows > 0 or pad_cols > 0:
        padded = torch.zeros(
            rows + pad_rows, cols + pad_cols, dtype=tensor.dtype, device=tensor.device
        )
        padded[:rows, :cols] = tensor
    else:
        padded = tensor

    # Reshape into blocks
    pr, pc = padded.shape
    blocks = padded.reshape(pr // rb, rb, pc // cb, cb).permute(0, 2, 1, 3)

    # Per-block max for scale computation
    block_max = blocks.abs().reshape(blocks.shape[0], blocks.shape[1], -1).max(dim=-1).values
    # Avoid division by zero
    block_max = block_max.clamp(min=1e-12)

    scale = block_max / _FP8_MAX
    scale_inv = scale  # scale_inv[i,j] = max_val / FP8_MAX

    # Quantize: scale each block, clamp, cast
    inv_scale = 1.0 / scale.unsqueeze(-1).unsqueeze(-1)  # broadcast over block dims
    scaled_blocks = blocks.float() * inv_scale
    clamped = scaled_blocks.clamp(-_FP8_MAX, _FP8_MAX)

    # Reshape back to padded shape
    quantized_padded = clamped.permute(0, 2, 1, 3).reshape(pr, pc)

    # Trim padding
    quantized = quantized_padded[:rows, :cols].to(torch.float8_e4m3fn)

    return quantized, scale_inv.float()


def dequantize_blockwise(
    quantized: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to float using blockwise scales.

    Args:
        quantized: float8_e4m3fn tensor.
        scale_inv: float32 scale tensor from :func:`quantize_blockwise`.
        block_size: Must match the block_size used during quantization.
        dtype: Output dtype.

    Returns:
        Dequantized tensor in the requested dtype.
    """
    assert quantized.ndim == 2, f"Expected 2D tensor, got {quantized.ndim}D"
    rows, cols = quantized.shape
    rb, cb = block_size

    # Pad to block boundaries
    pad_rows = (rb - rows % rb) % rb
    pad_cols = (cb - cols % cb) % cb
    if pad_rows > 0 or pad_cols > 0:
        padded = torch.zeros(
            rows + pad_rows, cols + pad_cols, dtype=torch.float32, device=quantized.device
        )
        padded[:rows, :cols] = quantized.float()
    else:
        padded = quantized.float()

    # Reshape into blocks
    pr, pc = padded.shape
    blocks = padded.reshape(pr // rb, rb, pc // cb, cb).permute(0, 2, 1, 3)

    # Multiply by scale
    blocks = blocks * scale_inv.unsqueeze(-1).unsqueeze(-1)

    # Reshape back
    result = blocks.permute(0, 2, 1, 3).reshape(pr, pc)
    return result[:rows, :cols].to(dtype)


# ---------------------------------------------------------------------------
# Weight classification
# ---------------------------------------------------------------------------

# Pattern for routed expert weights in DeepSeek models
# e.g. "model.layers.3.mlp.experts.42.gate_proj.weight"
_ROUTED_EXPERT_PATTERN = ".mlp.experts."
_SHARED_EXPERT_PATTERN = ".mlp.shared_experts."


def _is_routed_expert_weight(key: str) -> bool:
    """Check if a weight key belongs to a routed (non-shared) expert."""
    return _ROUTED_EXPERT_PATTERN in key and _SHARED_EXPERT_PATTERN not in key


# ---------------------------------------------------------------------------
# Keys to skip in DeepSeek checkpoints
# ---------------------------------------------------------------------------

# DeepSeek has some keys that should not be part of the merge:
# - Layer 61 is a placeholder/unused layer in some checkpoints
# - rotary_emb inverse frequency is derived, not a trained parameter
_SKIP_SUFFIXES = (".rotary_emb.inv_freq",)
_SKIP_LAYER_INDICES = frozenset({61})


def _should_skip_checkpoint_key(key: str) -> bool:
    """Check if a checkpoint key should be excluded from merge planning."""
    if any(key.endswith(s) for s in _SKIP_SUFFIXES):
        return True
    # Check for layer 61 (DeepSeek-specific)
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
                if layer_idx in _SKIP_LAYER_INDICES:
                    return True
            except ValueError:
                pass
    return False


# ---------------------------------------------------------------------------
# Native FP8 checkpoint handling
# ---------------------------------------------------------------------------


def _has_native_fp8_quantization(config_dict: dict) -> bool:
    """Check if the model checkpoint uses native FP8 quantization.

    DeepSeek V3.1 checkpoints can ship with native FP8 weights and
    ``quantization_config.quant_method == "fp8"``. These need to be
    dequantized before re-quantizing with our own scales.
    """
    quant_config = config_dict.get("quantization_config")
    if quant_config is None:
        return False
    if isinstance(quant_config, dict):
        return quant_config.get("quant_method", "") == "fp8"
    return False


def _get_native_block_size(config_dict: dict) -> tuple[int, int]:
    """Get the FP8 block size from the model's native quantization config.

    Falls back to the standard DeepSeek block size (128, 128) if not specified.
    """
    quant_config = config_dict.get("quantization_config", {})
    if isinstance(quant_config, dict):
        block_size = quant_config.get("weight_block_size", [_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE])
        return (int(block_size[0]), int(block_size[1]))
    return (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE)


def _make_cross_shard_tensor_loader(
    model_dir: Path,
) -> Callable[[str], torch.Tensor]:
    """Create a loader that can fetch a single tensor from any shard by key name.

    Used when a weight tensor and its scale are in different shards. Reads
    the safetensors index to find which shard contains a given key, then
    uses ``safe_open`` to load only that one tensor — no full shard loading.

    This keeps peak memory at O(single tensor) rather than O(full shard),
    which matters for DeepSeek V3 where shards are ~4-5 GB each.
    """
    # Build key → shard mapping from index
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index_weight_map: dict[str, str] = json.load(f)["weight_map"]
    else:
        # Single shard — build map from the one file
        shard_files = sorted(model_dir.glob("*.safetensors"))
        index_weight_map = {}
        for sf_path in shard_files:
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    index_weight_map[key] = sf_path.name

    def load_tensor(name: str) -> torch.Tensor:
        if name not in index_weight_map:
            raise KeyError(f"Tensor {name!r} not found in any shard at {model_dir}")
        shard_name = index_weight_map[name]
        with safe_open(str(model_dir / shard_name), framework="pt") as f:
            return f.get_tensor(name)

    return load_tensor


# ---------------------------------------------------------------------------
# vLLM compressed-tensors config
# ---------------------------------------------------------------------------


def _weight_scale_key(weight_key: str) -> str:
    """Map a weight key to its compressed-tensors scale key.

    Uses ``.weight_scale`` (compressed-tensors convention), NOT
    ``.weight_scale_inv`` (DeepSeek native convention).
    """
    return weight_key.removesuffix(".weight") + ".weight_scale"


def _build_vllm_quantization_config(output_weight_map: dict[str, str]) -> dict:
    """Build compressed-tensors quantization config for vLLM.

    Produces a config dict that tells vLLM which layers are FP8-quantized
    (routed experts) and which to ignore (everything else). No library
    imports needed — the schema is fixed and well-known.

    The decision is binary for every ``.weight`` key in the output:

    * **Has a matching ``.weight_scale``** → quantized → omitted from
      ``ignore`` so vLLM loads it as FP8.
    * **No matching ``.weight_scale``** → not quantized → added to
      ``ignore`` so vLLM leaves it alone.

    Adding non-``nn.Linear`` modules (norms, embeddings) to ``ignore`` is
    harmless — vLLM only consults the ignore list for modules that match
    ``targets: ["Linear"]``, so extra entries are silently skipped.

    Args:
        output_weight_map: Mapping of weight key -> shard filename.

    Returns:
        Dict suitable for config.json's ``compression_config`` field.
    """
    # Determine which modules have been quantized (have .weight_scale)
    quantized_prefixes = {
        key.removesuffix(".weight_scale")
        for key in output_weight_map
        if key.endswith(".weight_scale")
    }

    # Build ignore list: every .weight module that was NOT quantized.
    # This is model-agnostic — no need to enumerate projection suffixes
    # per architecture.
    ignore: list[str] = []
    for key in sorted(output_weight_map):
        if not key.endswith(".weight"):
            continue
        prefix = key.removesuffix(".weight")
        if prefix not in quantized_prefixes:
            ignore.append(prefix)

    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "global_compression_ratio": None,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "block",
                    "block_structure": [_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE],
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "tensor",
                    "dynamic": True,
                },
            },
        },
        "ignore": ignore,
    }


_VLLM_COMPAT_QUANT_CONFIG_FIELDS = {
    "config_groups",
    "format",
    "global_compression_ratio",
    "ignore",
    "kv_cache_scheme",
    "quantization_status",
}
_VLLM_COMPAT_QUANT_SCHEME_FIELDS = {
    "format",
    "input_activations",
    "output_activations",
    "targets",
    "weights",
}
_VLLM_COMPAT_QUANT_ARGS_FIELDS = {
    "actorder",
    "block_structure",
    "dynamic",
    "group_size",
    "num_bits",
    "observer",
    "observer_kwargs",
    "strategy",
    "symmetric",
    "type",
}


def _serialize_for_vllm(config: dict) -> dict:
    """Serialize only the compressed-tensors fields the current vLLM path needs.

    Uses an allowlist so new compressed-tensors fields are omitted automatically
    instead of breaking older vLLM builds.
    """
    serialized: dict = {}
    for key, value in config.items():
        if key == "config_groups" and isinstance(value, dict):
            serialized[key] = {
                group_name: _serialize_vllm_scheme(group)
                for group_name, group in value.items()
                if isinstance(group, dict)
            }
            continue
        if key in _VLLM_COMPAT_QUANT_CONFIG_FIELDS:
            serialized[key] = value
    serialized["quant_method"] = "compressed-tensors"
    return serialized


def _serialize_vllm_scheme(group: dict) -> dict:
    """Serialize a single quantization scheme for vLLM compatibility."""
    serialized: dict = {}
    for key, value in group.items():
        if key in {"weights", "input_activations", "output_activations"} and isinstance(
            value, dict
        ):
            serialized[key] = {
                field: field_value
                for field, field_value in value.items()
                if field in _VLLM_COMPAT_QUANT_ARGS_FIELDS
            }
            continue
        if key in _VLLM_COMPAT_QUANT_SCHEME_FIELDS:
            serialized[key] = value
    return serialized


# ---------------------------------------------------------------------------
# Resume state management
# ---------------------------------------------------------------------------

_MERGE_STATE_FILE = "merge_state.json"


def _load_resume_state(output_path: Path) -> dict:
    """Load resume state from a previous incomplete run.

    Returns:
        Dict with keys: ``status``, ``completed_shards`` (list of filenames),
        ``total_shards``. Returns empty dict if no state file exists.
    """
    state_file = output_path / _MERGE_STATE_FILE
    if not state_file.exists():
        return {}
    with open(state_file) as f:
        state = json.load(f)

    # Validate: every completed shard file must exist
    completed = state.get("completed_shards", [])
    for shard_name in completed:
        if not (output_path / shard_name).exists():
            raise WeightsMergeError(
                f"Resume state references {shard_name!r} but file not found in {output_path}. "
                f"Delete {output_path} and restart."
            )
    return state


def _save_merge_state(
    output_path: Path,
    *,
    status: str,
    completed_shards: list[str],
    total_shards: int,
) -> None:
    """Save merge state atomically for resume support."""
    state = {
        "status": status,
        "completed_shards": completed_shards,
        "total_shards": total_shards,
    }
    tmp = output_path / f"{_MERGE_STATE_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(output_path / _MERGE_STATE_FILE)


def _save_shard_atomic(
    output_path: Path, shard_name: str, tensors: dict[str, torch.Tensor]
) -> None:
    """Save a shard file atomically (write to temp, then rename)."""
    tmp_name = f"{shard_name}.tmp"
    save_file(tensors, str(output_path / tmp_name))
    (output_path / tmp_name).rename(output_path / shard_name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_quantized(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    trust_remote_code: bool,
    model_dir: Path,
    config_dict: dict,
    serving_format: str,
) -> None:
    """Merge LoRA adapter and quantize routed experts to FP8.

    Processes one safetensors shard at a time:
    1. Load shard tensors
    2. Apply any LoRA merge ops targeting this shard
    3. Quantize routed expert weights to FP8 with blockwise scales
    4. Preserve dense/shared-expert weights in BF16
    5. Write output shard (preserving input shard layout)
    6. Track progress for resume support

    After all shards are processed:
    - Write safetensors index
    - Patch config.json with compressed-tensors metadata
    - Copy tokenizer, model code

    Args:
        base_model: Model name or path (for tokenizer loading).
        adapter_path: Path to adapter directory.
        output_path: Where to write the quantized model.
        trust_remote_code: Whether to trust remote code.
        model_dir: Resolved local model directory.
        config_dict: Parsed config.json dict.
        serving_format: Serving framework format (e.g. "vllm").
    """
    out = Path(output_path)

    # Check for resume
    resume_state = {}
    if out.exists():
        resume_state = _load_resume_state(out)
        if not resume_state:
            raise FileExistsError(f"Output path already exists: {out}")
        if resume_state.get("status") == "completed":
            logger.info("Output already complete at %s, skipping", out)
            return
        logger.info(
            "Resuming: %d/%d shards completed",
            len(resume_state.get("completed_shards", [])),
            resume_state.get("total_shards", "?"),
        )
    else:
        out.mkdir(parents=True, exist_ok=False)

    # 1. Load adapter
    adapter_weights, adapter_config = load_adapter_weights(Path(adapter_path))

    # 2. Read model metadata
    model_shapes = get_model_state_shapes(model_dir)
    model_state_keys = set(model_shapes.keys())

    # Pre-filter keys that DeepSeek checkpoints include but shouldn't be merged.
    # Also exclude .weight_scale_inv — these are native FP8 scales, not merge targets.
    filtered_keys = {
        k
        for k in model_state_keys
        if not _should_skip_checkpoint_key(k) and not k.endswith(".weight_scale_inv")
    }

    # 3. Detect merge profile and plan ops
    profile = detect_merge_profile(config_dict, model_state_keys)
    logger.info(
        "Detected merge profile: expert_layout=%s, language_model_prefix=%s",
        profile.expert_layout,
        profile.has_language_model_prefix,
    )

    merge_ops = plan_merge_ops(adapter_weights, adapter_config, filtered_keys, profile)
    total_ops = sum(len(ops) for ops in merge_ops.values())
    logger.info("Planned %d merge operations across %d target keys", total_ops, len(merge_ops))

    # Validate shapes against filtered keys
    filtered_shapes = {k: v for k, v in model_shapes.items() if k in filtered_keys}
    validate_merge_op_shapes(merge_ops, filtered_shapes)

    # 4. Set up native FP8 handling (cross-shard scale lookup)
    is_native_fp8 = _has_native_fp8_quantization(config_dict)
    native_block_size = _get_native_block_size(config_dict) if is_native_fp8 else None
    cross_shard_loader = _make_cross_shard_tensor_loader(model_dir) if is_native_fp8 else None
    if is_native_fp8:
        logger.info(
            "Native FP8 checkpoint detected (block_size=%s), will dequantize before re-quantize",
            native_block_size,
        )

    # 5. Process shards
    shard_files = get_shard_files(model_dir)
    completed_shards = set(resume_state.get("completed_shards", []))
    all_completed: list[str] = list(completed_shards)
    weight_map: dict[str, str] = {}

    # Rebuild weight map from already-completed shards
    for shard_name in completed_shards:
        shard_tensors = load_file(str(out / shard_name))
        for key in shard_tensors:
            weight_map[key] = shard_name
        # Pop merge ops for completed shard keys
        for key in shard_tensors:
            merge_ops.pop(key, None)
        del shard_tensors

    logger.info(
        "Processing %d input shard(s) (%d already completed)",
        len(shard_files),
        len(completed_shards),
    )
    ops_applied = 0

    for i, shard_file in enumerate(shard_files):
        # Determine output shard name (preserve input naming)
        out_shard_name = shard_file

        if out_shard_name in completed_shards:
            logger.info("Skipping completed shard %d/%d: %s", i + 1, len(shard_files), shard_file)
            continue

        logger.info("Processing shard %d/%d: %s", i + 1, len(shard_files), shard_file)
        tensors = load_file(str(model_dir / shard_file))
        output_tensors: dict[str, torch.Tensor] = {}

        for key in list(tensors.keys()):
            tensor = tensors[key]

            # Skip keys that shouldn't be in output
            if _should_skip_checkpoint_key(key):
                continue

            # Skip native scale_inv tensors (we generate new .weight_scale)
            if key.endswith(".weight_scale_inv"):
                continue

            # Step 1: Dequantize native FP8 weights BEFORE merge
            # Native FP8 checkpoints store weights in FP8 + scale_inv.
            # We must dequantize to BF16 first so the LoRA merge math works
            # correctly in float precision.
            if key.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn and is_native_fp8:
                scale_key = key.replace(".weight", ".weight_scale_inv")
                # Scale may be in this shard or a different one
                scale_inv = tensors.get(scale_key)
                if scale_inv is None and cross_shard_loader is not None:
                    scale_inv = cross_shard_loader(scale_key)
                if scale_inv is not None:
                    assert native_block_size is not None
                    tensor = dequantize_blockwise(tensor, scale_inv, block_size=native_block_size)
                else:
                    raise WeightsMergeError(
                        f"Native FP8 weight {key!r} has no .weight_scale_inv tensor "
                        f"in any shard. Cannot dequantize for merge."
                    )

            # Step 2: Apply LoRA merge ops (on dequantized BF16 tensors)
            ops_for_key = merge_ops.pop(key, [])
            if ops_for_key:
                temp = {key: tensor}
                for op in ops_for_key:
                    apply_merge_op(temp, op)
                    ops_applied += 1
                tensor = temp[key]

            # Step 3: Quantize routed experts to FP8, preserve everything else
            if _is_routed_expert_weight(key) and key.endswith(".weight"):
                fp8_tensor, scale = quantize_blockwise(tensor)
                output_tensors[key] = fp8_tensor
                output_tensors[_weight_scale_key(key)] = scale
            else:
                output_tensors[key] = tensor

            weight_map[key] = out_shard_name
            # Also track scale tensors in weight map
            scale_out_key = _weight_scale_key(key) if key.endswith(".weight") else None
            if (
                scale_out_key
                and scale_out_key in output_tensors
                and scale_out_key not in weight_map
            ):
                weight_map[scale_out_key] = out_shard_name

        del tensors

        # Save shard atomically
        _save_shard_atomic(out, out_shard_name, output_tensors)
        del output_tensors

        all_completed.append(out_shard_name)
        _save_merge_state(
            out,
            status="in_progress",
            completed_shards=all_completed,
            total_shards=len(shard_files),
        )

    # Verify all merge ops were consumed
    if merge_ops:
        unconsumed = list(merge_ops.keys())
        raise WeightsMergeError(
            f"Merge ops not applied — {len(unconsumed)} target keys not found in any shard: "
            f"{unconsumed[:5]}{'...' if len(unconsumed) > 5 else ''}"
        )

    logger.info("Applied %d/%d merge operations", ops_applied, total_ops)

    # 6. Write index
    shard_names = set(weight_map.values())
    index = {
        "metadata": {"total_size": _compute_total_size(out, shard_names)},
        "weight_map": dict(sorted(weight_map.items())),
    }
    index_path = out / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # 7. Copy config and patch with quantization metadata
    src_config = model_dir / "config.json"
    if src_config.exists():
        copy_artifact_file(src_config, out / "config.json")

    if serving_format == "vllm":
        quant_config = _build_vllm_quantization_config(weight_map)
        _patch_config_with_quantization(out, quant_config)

    # 7. Copy model code and tokenizer
    copy_model_code_files(model_dir, out)
    save_tokenizer_and_processor(base_model, out, is_multimodal_from_dict(config_dict))

    # 8. Mark complete
    _save_merge_state(
        out,
        status="completed",
        completed_shards=all_completed,
        total_shards=len(shard_files),
    )

    logger.info("Done — quantized model saved to %s", out)


_DTYPE_SIZES: dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "BOOL": 1,
}


def _compute_total_size(output_path: Path, shard_names: set[str]) -> int:
    """Compute total byte size of all tensors across output shards.

    Reads safetensors headers only (shape + dtype) without loading tensor data,
    matching the HuggingFace convention for ``model.safetensors.index.json``.
    """
    total = 0
    for name in shard_names:
        shard_path = output_path / name
        if not shard_path.exists():
            continue
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                shape = f.get_slice(key).get_shape()
                dtype_str = f.get_slice(key).get_dtype()
                numel = 1
                for dim in shape:
                    numel *= dim
                total += numel * _DTYPE_SIZES.get(dtype_str, 4)
    return total


def _patch_config_with_quantization(output_path: Path, quant_config: dict) -> None:
    """Patch config.json with compressed-tensors quantization metadata.

    Adds ``compression_config`` and removes ``quantization_config`` (which
    refers to the input model's native quantization, not our output).
    """
    config_path = output_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    config["compression_config"] = _serialize_for_vllm(quant_config)
    config.pop("quantization_config", None)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
