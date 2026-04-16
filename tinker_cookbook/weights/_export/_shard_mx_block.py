"""MX block-quantized shard processing (MXFP4, MXFP8).

Provides the shard processing hooks used by the shard engine
(:mod:`_shard_engine`) when processing checkpoints with MX block-quantized
expert weights (e.g. GPT-OSS with MXFP4). This keeps all MX block
dequant/merge/requant logic isolated so it cannot affect other model families.

The :class:`MXBlockShardHooks` class implements the
:class:`~._quant_format.ShardHooks` protocol.
"""

from __future__ import annotations

import logging

import torch

from tinker_cookbook.exceptions import WeightsMergeError
from tinker_cookbook.weights._merge import MergeOp, apply_merge_op

logger = logging.getLogger(__name__)

_BLOCKS_SUFFIX = "_blocks"
_SCALES_SUFFIX = "_scales"
_BIAS_SUFFIX = "_bias"


# ---------------------------------------------------------------------------
# Pre-planning: augment model state with virtual keys
# ---------------------------------------------------------------------------


def augment_model_state_for_planning(
    model_state_keys: set[str],
    model_shapes: dict[str, tuple[int, ...]],
) -> tuple[set[str], dict[str, tuple[int, ...]], dict[str, str]]:
    """Add virtual weight keys for MX block-quantized expert weights.

    For each ``_blocks`` key, creates a virtual key without the suffix
    so the merge planner can target standard weight names. Also infers
    the original unquantized shape from the blocks shape.

    Example:
        ``model.layers.0.mlp.experts.gate_up_proj_blocks`` (32, 5760, 90, 16)
        → virtual ``model.layers.0.mlp.experts.gate_up_proj`` (32, 5760, 2880)

    Returns:
        Tuple of ``(augmented_keys, augmented_shapes, blocks_map)`` where
        ``blocks_map`` maps each virtual key to its ``_blocks`` counterpart.
    """
    from tinker_cookbook.weights._mxfp4 import BLOCK_SIZE

    augmented_keys = set(model_state_keys)
    augmented_shapes = dict(model_shapes)
    blocks_map: dict[str, str] = {}

    for key in list(model_state_keys):
        if not key.endswith(_BLOCKS_SUFFIX):
            continue

        # Derive virtual key: strip _blocks suffix
        virtual_key = key.removesuffix(_BLOCKS_SUFFIX)

        # Infer original shape from blocks.
        # blocks: (..., out_dim, n_blocks, packed) where in_dim = n_blocks * BLOCK_SIZE
        # The unquantized model stores fused expert weights as (..., in_dim, out_dim)
        # (fused dim on last axis for interleaved slicing), so we swap.
        blocks_shape = model_shapes.get(key)
        if blocks_shape is None or len(blocks_shape) < 3:
            continue

        out_dim = blocks_shape[-3]
        n_blocks = blocks_shape[-2]
        in_dim = n_blocks * BLOCK_SIZE
        virtual_shape = blocks_shape[:-3] + (in_dim, out_dim)

        augmented_keys.add(virtual_key)
        augmented_shapes[virtual_key] = virtual_shape
        blocks_map[virtual_key] = key

    if blocks_map:
        logger.info(
            "Created %d virtual keys for MX block-quantized weights",
            len(blocks_map),
        )

    return augmented_keys, augmented_shapes, blocks_map


# ---------------------------------------------------------------------------
# Shard processing: dequant → merge → requant
# ---------------------------------------------------------------------------


def apply_mx_block_merge_ops(
    tensors: dict[str, torch.Tensor],
    blocks_key: str,
    virtual_key: str,
    ops: list[MergeOp],
    device: torch.device | None = None,
) -> int:
    """Dequantize MX block weights, apply merge ops, re-quantize.

    Modifies ``tensors`` in-place: the blocks and scales keys are updated
    with re-quantized values after the LoRA delta is applied.
    The bias key (if present) is left unchanged.

    Args:
        tensors: Shard tensor dict (from safetensors load).
        blocks_key: The actual key ending in ``_blocks``.
        virtual_key: The virtual key (without suffix) that merge ops target.
        ops: Merge ops targeting this weight.
        device: Device for dequant/requant math. If not CPU, tensors are
            moved to device for compute and back to CPU for writing.

    Returns:
        Number of merge ops applied.
    """
    from tinker_cookbook.weights._mxfp4 import BLOCK_SIZE, dequantize_mxfp4, quantize_mxfp4

    base = blocks_key.removesuffix(_BLOCKS_SUFFIX)
    scales_key = base + _SCALES_SUFFIX

    blocks_tensor = tensors[blocks_key]
    scales_tensor = tensors.get(scales_key)

    if scales_tensor is None:
        raise WeightsMergeError(f"Missing scales tensor {scales_key!r} for blocks {blocks_key!r}")

    # Move to compute device if specified
    use_gpu = device is not None and device.type != "cpu"
    if use_gpu:
        blocks_tensor = blocks_tensor.to(device)
        scales_tensor = scales_tensor.to(device)

    # Dequantize: blocks (..., out_dim, n_blocks, 16) -> (..., out_dim, in_dim)
    n_blocks = blocks_tensor.shape[-2]
    in_dim = n_blocks * BLOCK_SIZE
    dequant_shape = blocks_tensor.shape[:-2] + (in_dim,)
    weight_bf16 = dequantize_mxfp4(blocks_tensor, scales_tensor, dequant_shape)

    # Transpose to (..., in_dim, out_dim) for merge ops — the unquantized model
    # stores weights this way (fused dim on last axis for interleaved slicing).
    weight_bf16 = weight_bf16.transpose(-1, -2)

    # Apply merge ops (on device)
    temp = {virtual_key: weight_bf16}
    for op in ops:
        apply_merge_op(temp, op)
    weight_bf16 = temp[virtual_key]

    # Transpose back to (..., out_dim, in_dim) for re-quantization
    weight_bf16 = weight_bf16.transpose(-1, -2)

    # Re-quantize
    new_blocks, new_scales = quantize_mxfp4(weight_bf16)

    # Move back to CPU for writing
    if use_gpu:
        new_blocks = new_blocks.cpu()
        new_scales = new_scales.cpu()

    # Update tensors in-place
    tensors[blocks_key] = new_blocks
    tensors[scales_key] = new_scales

    return len(ops)


def try_apply_mx_block_ops(
    key: str,
    tensors: dict[str, torch.Tensor],
    merge_ops: dict[str, list[MergeOp]],
    blocks_map: dict[str, str],
    device: torch.device | None = None,
) -> int:
    """Check if a shard key has pending MX block merge ops and apply them.

    This is the main hook called by the shard loop. Returns 0 if
    the key is not a blocks weight or has no pending ops.

    Args:
        key: Current shard tensor key.
        tensors: Full shard tensor dict (modified in-place).
        merge_ops: Mutable ops dict (consumed ops are popped).
        blocks_map: Virtual-key → blocks-key mapping.
        device: Device for dequant/requant math.

    Returns:
        Number of ops applied (0 if this key wasn't handled).
    """
    if not key.endswith(_BLOCKS_SUFFIX):
        return 0
    virtual_key = key.removesuffix(_BLOCKS_SUFFIX)
    ops = merge_ops.pop(virtual_key, [])
    if not ops:
        return 0
    return apply_mx_block_merge_ops(tensors, key, virtual_key, ops, device=device)


# ---------------------------------------------------------------------------
# ShardHooks implementation
# ---------------------------------------------------------------------------


class MXBlockShardHooks:
    """Hooks for MX block-quantized weights (MXFP4, MXFP8).

    Implements the :class:`~._quant_format.ShardHooks` protocol.

    Args:
        config_dict: Parsed config.json dict.
        device: Device for dequant/requant math (``"cpu"``, ``"cuda"``, etc.).
    """

    def __init__(self, config_dict: dict, device: str = "cpu") -> None:
        self._blocks_map: dict[str, str] = {}
        self._device = torch.device(device)

    def augment_for_planning(
        self,
        model_state_keys: set[str],
        model_shapes: dict[str, tuple[int, ...]],
    ) -> tuple[set[str], dict[str, tuple[int, ...]]]:
        augmented_keys, augmented_shapes, self._blocks_map = augment_model_state_for_planning(
            model_state_keys, model_shapes
        )
        return augmented_keys, augmented_shapes

    def try_apply(
        self,
        key: str,
        tensors: dict[str, torch.Tensor],
        merge_ops: dict[str, list[MergeOp]],
    ) -> int:
        if not self._blocks_map:
            return 0
        return try_apply_mx_block_ops(
            key, tensors, merge_ops, self._blocks_map, device=self._device
        )
