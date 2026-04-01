"""Compressed-tensors INT4 shard processing: packed expert weight handling.

Provides the shard processing hooks used by the generic shard export
(:mod:`_shard`) when processing checkpoints with INT4 quantized expert
weights (e.g. Kimi K2, K2.5). This keeps all INT4 dequant/merge/requant
logic isolated so it cannot affect other model families.
"""

from __future__ import annotations

import logging

from tinker_cookbook.exceptions import WeightsMergeError
from tinker_cookbook.weights._merge import MergeOp, apply_merge_op
from tinker_cookbook.weights._merge_utils import (
    create_virtual_weight_keys,
    create_virtual_weight_shapes,
)

logger = logging.getLogger(__name__)

# Suffix conventions for compressed-tensors pack-quantized format.
_PACKED_SUFFIX = ".weight_packed"
_SCALE_SUFFIX = ".weight_scale"
_SHAPE_SUFFIX = ".weight_shape"


# ---------------------------------------------------------------------------
# Pre-planning: augment model state with virtual keys
# ---------------------------------------------------------------------------


def augment_model_state_for_planning(
    model_state_keys: set[str],
    model_shapes: dict[str, tuple[int, ...]],
) -> tuple[set[str], dict[str, tuple[int, ...]], dict[str, str]]:
    """Add virtual ``.weight`` entries for INT4 packed expert weights.

    Call this before :func:`plan_merge_ops` so the planner can target
    standard ``.weight`` key names even though the on-disk checkpoint
    uses ``.weight_packed``.

    Returns:
        Tuple of ``(augmented_keys, augmented_shapes, packed_map)`` where
        ``packed_map`` maps each virtual ``.weight`` key to its
        ``.weight_packed`` counterpart.
    """
    augmented_keys, packed_map = create_virtual_weight_keys(model_state_keys)
    augmented_shapes = create_virtual_weight_shapes(model_shapes, packed_map)
    if packed_map:
        logger.info(
            "Created %d virtual .weight keys for INT4 packed experts",
            len(packed_map),
        )
    return augmented_keys, augmented_shapes, packed_map


# ---------------------------------------------------------------------------
# Shard processing: dequant → merge → requant
# ---------------------------------------------------------------------------


def apply_packed_merge_ops(
    tensors: dict,
    packed_key: str,
    virtual_key: str,
    ops: list[MergeOp],
    group_size: int,
) -> int:
    """Dequantize a packed weight, apply merge ops, re-quantize.

    Modifies ``tensors`` in-place: the packed/scale/shape keys are updated
    with re-quantized values after the LoRA delta is applied.

    Args:
        tensors: Shard tensor dict (from safetensors load).
        packed_key: The actual key ending in ``.weight_packed``.
        virtual_key: The virtual ``.weight`` key that merge ops target.
        ops: Merge ops targeting this weight.
        group_size: INT4 group size for re-quantization.

    Returns:
        Number of merge ops applied.
    """
    from tinker_cookbook.weights._packed_int4 import dequantize_int4_group, quantize_int4_group

    base = packed_key.removesuffix(_PACKED_SUFFIX)
    scale_key = base + _SCALE_SUFFIX
    shape_key = base + _SHAPE_SUFFIX

    packed_tensor = tensors[packed_key]
    scale_tensor = tensors.get(scale_key)
    shape_tensor = tensors.get(shape_key)

    if scale_tensor is None:
        raise WeightsMergeError(
            f"Missing scale tensor {scale_key!r} for packed weight {packed_key!r}"
        )
    if shape_tensor is None:
        raise WeightsMergeError(
            f"Missing shape tensor {shape_key!r} for packed weight {packed_key!r}"
        )

    original_shape = tuple(shape_tensor.tolist())
    assert len(original_shape) == 2, f"Expected 2D shape, got {original_shape}"

    # Dequantize INT4 → bf16
    weight_bf16 = dequantize_int4_group(packed_tensor, scale_tensor, original_shape, group_size)

    # Apply merge ops on the dequantized weight
    temp = {virtual_key: weight_bf16}
    for op in ops:
        apply_merge_op(temp, op)
    weight_bf16 = temp[virtual_key]

    # Re-quantize bf16 → INT4
    new_packed, new_scale = quantize_int4_group(weight_bf16, group_size)

    # Update tensors in-place
    tensors[packed_key] = new_packed
    tensors[scale_key] = new_scale
    # weight_shape stays unchanged

    return len(ops)


def try_apply_packed_ops(
    key: str,
    tensors: dict,
    merge_ops: dict[str, list[MergeOp]],
    packed_map: dict[str, str],
    group_size: int,
) -> int:
    """Check if a shard key has pending packed merge ops and apply them.

    This is the main hook called by the generic shard loop. Returns 0 if
    the key is not a packed weight or has no pending ops.

    Args:
        key: Current shard tensor key.
        tensors: Full shard tensor dict (modified in-place).
        merge_ops: Mutable ops dict (consumed ops are popped).
        packed_map: Virtual-key → packed-key mapping.
        group_size: INT4 group size.

    Returns:
        Number of ops applied (0 if this key wasn't handled).
    """
    if not key.endswith(_PACKED_SUFFIX):
        return 0
    virtual_key = key.removesuffix(_PACKED_SUFFIX) + ".weight"
    ops = merge_ops.pop(virtual_key, [])
    if not ops:
        return 0
    return apply_packed_merge_ops(tensors, key, virtual_key, ops, group_size)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def get_int4_group_size(config_dict: dict) -> int:
    """Extract INT4 group size from compressed-tensors quantization config.

    Checks both top-level and nested ``text_config`` for the quantization
    config (Kimi K2.5 stores it under ``text_config``).
    """
    for config in [config_dict, config_dict.get("text_config", {})]:
        if not isinstance(config, dict):
            continue
        quant = config.get("quantization_config")
        if not isinstance(quant, dict):
            continue
        groups = quant.get("config_groups", {})
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            weights_cfg = group.get("weights", {})
            if not isinstance(weights_cfg, dict):
                continue
            gs = weights_cfg.get("group_size")
            if isinstance(gs, int) and gs > 0:
                return gs
    return 32  # Default fallback
