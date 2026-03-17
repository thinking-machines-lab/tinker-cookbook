"""Quantized export strategy.

Merges LoRA adapters shard-by-shard and quantizes routed expert weights to FP8.
Produces output compatible with vLLM's compressed-tensors format.

Currently supports DeepSeek V3/V3.1 models. The infrastructure (FP8 math, vLLM
config generation, resume support) is reusable for future model families.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tinker_cookbook.weights._artifacts import (
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

# Max representable value in float8_e4m3fn
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


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
# vLLM compressed-tensors config
# ---------------------------------------------------------------------------


def _build_vllm_quantization_config(output_weight_map: dict[str, str]) -> dict:
    """Build compressed-tensors quantization config for vLLM.

    Produces a config dict that tells vLLM which layers are FP8-quantized
    (routed experts) and which to ignore (everything else). No library
    imports needed — the schema is fixed and well-known.

    Args:
        output_weight_map: Mapping of weight key -> shard filename.

    Returns:
        Dict suitable for config.json's ``compression_config`` field.
    """
    # Build ignore list: all non-routed-expert weight modules
    ignore = sorted(
        {
            key.removesuffix(".weight")
            for key in output_weight_map
            if key.endswith(".weight") and not _is_routed_expert_weight(key)
        }
    )

    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "global_compression_ratio": None,
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "tensor",
                },
                "targets": ["Linear"],
            },
        },
        "ignore": ignore,
    }


def _serialize_for_vllm(config: dict) -> dict:
    """Strip unknown fields from quantization config for vLLM compatibility.

    vLLM's compressed-tensors loader expects a specific set of fields.
    This strips anything that might cause warnings or errors.
    """
    known_keys = {
        "quant_method",
        "format",
        "quantization_status",
        "global_compression_ratio",
        "config_groups",
        "ignore",
    }
    return {k: v for k, v in config.items() if k in known_keys}


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
            raise RuntimeError(
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

    # Pre-filter keys that DeepSeek checkpoints include but shouldn't be merged
    filtered_keys = {k for k in model_state_keys if not _should_skip_checkpoint_key(k)}

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

    # 4. Process shards
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

            # Apply merge ops
            ops_for_key = merge_ops.pop(key, [])
            if ops_for_key:
                # Put tensor in a temp dict for apply_merge_op
                temp = {key: tensor}
                for op in ops_for_key:
                    apply_merge_op(temp, op)
                    ops_applied += 1
                tensor = temp[key]

            # Handle native FP8 weight_scale_inv (dequantize before re-quantize)
            scale_key = key.replace(".weight", ".weight_scale_inv")
            if (
                key.endswith(".weight")
                and tensor.dtype == torch.float8_e4m3fn
                and scale_key in tensors
            ):
                tensor = dequantize_blockwise(tensor, tensors[scale_key])
                # Don't include the original scale_inv in output

            # Quantize routed experts to FP8
            if _is_routed_expert_weight(key) and key.endswith(".weight"):
                # Ensure we're working with a float tensor for quantization
                if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    fp8_tensor, scale = quantize_blockwise(tensor)
                    output_tensors[key] = fp8_tensor
                    output_tensors[key.replace(".weight", ".weight_scale_inv")] = scale
                else:
                    # Already FP8 from dequant+requant path above shouldn't hit this,
                    # but handle gracefully
                    output_tensors[key] = tensor
            elif key.endswith(".weight_scale_inv"):
                # Skip original scale tensors (we generate new ones during quantization)
                continue
            else:
                # Dense/shared expert/non-weight tensors: preserve as-is
                # Ensure BF16 for weight tensors
                if tensor.dtype == torch.float8_e4m3fn:
                    # Shouldn't happen for non-expert weights, but be safe
                    output_tensors[key] = tensor
                else:
                    output_tensors[key] = tensor

            weight_map[key] = out_shard_name
            # Also track scale tensors in weight map
            scale_out_key = key.replace(".weight", ".weight_scale_inv")
            if scale_out_key in output_tensors and scale_out_key not in weight_map:
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

    logger.info("Applied %d/%d merge operations", ops_applied, total_ops)

    # 5. Write index
    shard_names = set(weight_map.values())
    index = {
        "metadata": {"total_size": _compute_total_size(out, shard_names)},
        "weight_map": dict(sorted(weight_map.items())),
    }
    index_path = out / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # 6. Copy config and patch with quantization metadata
    src_config = model_dir / "config.json"
    if src_config.exists():
        shutil.copy2(src_config, out / "config.json")

    if serving_format == "vllm":
        quant_config = _build_vllm_quantization_config(weight_map)
        _patch_config_with_quantization(out, quant_config)

    # 7. Copy model code and tokenizer
    copy_model_code_files(model_dir, out)
    save_tokenizer_and_processor(
        base_model, out, is_multimodal_from_dict(config_dict), trust_remote_code
    )

    # 8. Mark complete
    _save_merge_state(
        out,
        status="completed",
        completed_shards=all_completed,
        total_shards=len(shard_files),
    )

    logger.info("Done — quantized model saved to %s", out)


def _compute_total_size(output_path: Path, shard_names: set[str]) -> int:
    """Compute total size of all tensors across output shards."""
    total = 0
    for name in shard_names:
        path = output_path / name
        if path.exists():
            total += path.stat().st_size
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
