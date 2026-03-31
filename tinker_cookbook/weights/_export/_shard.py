"""Shard-by-shard export strategy.

Processes one safetensors shard at a time, keeping peak memory proportional to
the largest shard rather than the full model. Produces output identical to the
full-model path.

Supports models with compressed-tensors INT4 quantized weights (e.g. Kimi
K2.5): packed expert weights are dequantized before LoRA merge, then
re-quantized to the same format in the output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from safetensors.torch import load_file

from tinker_cookbook.exceptions import WeightsMergeError
from tinker_cookbook.weights._artifacts import (
    ShardWriter,
    copy_artifact_file,
    copy_model_code_files,
    get_model_state_shapes,
    get_shard_files,
    load_adapter_weights,
)
from tinker_cookbook.weights._export import (
    cleanup_on_failure,
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

# Suffix conventions for compressed-tensors pack-quantized format.
_PACKED_SUFFIX = ".weight_packed"
_SCALE_SUFFIX = ".weight_scale"
_SHAPE_SUFFIX = ".weight_shape"


def build_sharded(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    trust_remote_code: bool,
    model_dir: Path,
    config_dict: dict,
) -> None:
    """Merge by processing one safetensors shard at a time.

    For models with compressed-tensors INT4 quantized weights (detected by
    the presence of ``.weight_packed`` keys), packed expert weights are
    transparently dequantized before LoRA merge and re-quantized in the
    output.

    Args:
        base_model: Original model name (used for tokenizer loading).
        adapter_path: Path to adapter directory.
        output_path: Where to write the merged model.
        trust_remote_code: Whether to trust remote code for HF loading.
        model_dir: Resolved local directory containing model files.
        config_dict: Parsed config.json dict (loaded by dispatcher).
    """
    # 0. Fail fast if output already exists (before any expensive work)
    out = Path(output_path)
    if out.exists():
        raise FileExistsError(f"Output path already exists: {out}")

    # 1. Load adapter (small — only LoRA matrices)
    adapter_weights, adapter_config = load_adapter_weights(Path(adapter_path))

    # 2. Read model state shapes from safetensors headers (no weight loading)
    model_shapes = get_model_state_shapes(model_dir)
    model_state_keys = set(model_shapes.keys())

    # 3. Detect model-specific merge profile from config + key names
    profile = detect_merge_profile(config_dict, model_state_keys)
    logger.info(
        "Detected merge profile: family=%s, expert_layout=%s, language_model_prefix=%s",
        profile.model_family,
        profile.expert_layout,
        profile.has_language_model_prefix,
    )

    # 4. Handle compressed-tensors INT4 packed weights (e.g. Kimi K2.5).
    #    Create virtual .weight keys so the planner can target them; the
    #    actual dequant/merge/requant happens during shard processing.
    packed_map: dict[str, str] = {}
    if profile.model_family == "kimi_k25":
        from tinker_cookbook.weights._merge_kimi_k25 import (
            create_virtual_weight_keys,
            create_virtual_weight_shapes,
        )

        model_state_keys, packed_map = create_virtual_weight_keys(model_state_keys)
        model_shapes = create_virtual_weight_shapes(model_shapes, packed_map)
        if packed_map:
            logger.info(
                "Created %d virtual .weight keys for INT4 packed experts",
                len(packed_map),
            )

    # 5. Plan all merge ops (validates keys before any heavy I/O)
    merge_ops = plan_merge_ops(adapter_weights, adapter_config, model_state_keys, profile)
    total_ops = sum(len(ops) for ops in merge_ops.values())
    logger.info("Planned %d merge operations across %d target keys", total_ops, len(merge_ops))

    # 6. Validate shapes upfront (catches mismatches before loading any shards)
    validate_merge_op_shapes(merge_ops, model_shapes)

    # 7. Resolve INT4 quantization config for re-quantization
    int4_group_size = _get_int4_group_size(config_dict) if packed_map else None

    # 8. Process shards
    out.mkdir(parents=True, exist_ok=False)

    try:
        shard_files = get_shard_files(model_dir)
        logger.info("Processing %d input shard(s)", len(shard_files))

        writer = ShardWriter(out)
        ops_applied = 0

        for i, shard_file in enumerate(shard_files):
            logger.info("Processing shard %d/%d: %s", i + 1, len(shard_files), shard_file)
            tensors = load_file(str(model_dir / shard_file))

            # Apply merge ops targeting keys in this shard.
            # For packed weights, we handle dequant → merge → requant.
            for key in list(tensors.keys()):
                # Direct key match (standard path)
                ops_for_key = merge_ops.pop(key, [])
                if ops_for_key:
                    for op in ops_for_key:
                        apply_merge_op(tensors, op)
                        ops_applied += 1
                    continue

                # Check if this packed key has ops targeting a virtual .weight
                if key.endswith(_PACKED_SUFFIX) and packed_map:
                    virtual_key = key.removesuffix(_PACKED_SUFFIX) + ".weight"
                    ops_for_virtual = merge_ops.pop(virtual_key, [])
                    if ops_for_virtual:
                        ops_applied += _apply_packed_merge_ops(
                            tensors, key, virtual_key, ops_for_virtual, int4_group_size
                        )

            # Write all tensors from this shard to output
            for key, tensor in tensors.items():
                writer.add_tensor(key, tensor)
            del tensors
            writer.flush()

        # 9. Verify all ops were consumed
        if merge_ops:
            unconsumed = list(merge_ops.keys())
            raise WeightsMergeError(
                f"Merge ops not applied — {len(unconsumed)} target keys not found in any shard: "
                f"{unconsumed[:5]}{'...' if len(unconsumed) > 5 else ''}"
            )

        logger.info("Applied %d/%d merge operations", ops_applied, total_ops)

        # 10. Finalize output shards
        weight_map = writer.finalize()

        # 11. Write index file (only for multi-shard output; HF convention
        #     is no index for single-shard models)
        shard_names = set(weight_map.values())
        if len(shard_names) > 1:
            index = {
                "metadata": {"total_size": writer.total_size},
                "weight_map": dict(sorted(weight_map.items())),
            }
            index_path = out / "model.safetensors.index.json"
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

        # 12. Save config, tokenizer, and model code files.
        #     Copy config.json directly (safe — it's a single known file).
        #     Copy *.py files for trust_remote_code model support.
        #     We intentionally don't glob-copy all non-weight files to avoid
        #     accidentally including stale index files or other artifacts that
        #     could break downstream loaders like vLLM/SGLang.
        src_config = model_dir / "config.json"
        if src_config.exists():
            copy_artifact_file(src_config, out / "config.json")
        copy_model_code_files(model_dir, out)
        save_tokenizer_and_processor(
            base_model, out, is_multimodal_from_dict(config_dict), trust_remote_code
        )

        logger.info("Done — merged model saved to %s", out)
    except Exception:
        cleanup_on_failure(out)
        raise


# ---------------------------------------------------------------------------
# INT4 packed weight handling
# ---------------------------------------------------------------------------


def _apply_packed_merge_ops(
    tensors: dict,
    packed_key: str,
    virtual_key: str,
    ops: list,
    group_size: int | None,
) -> int:
    """Dequantize a packed weight, apply merge ops, re-quantize.

    Modifies ``tensors`` in-place: the packed/scale/shape keys are updated
    with re-quantized values after the LoRA delta is applied.

    Args:
        tensors: Shard tensor dict (from safetensors load).
        packed_key: The actual key ending in ``.weight_packed``.
        virtual_key: The virtual ``.weight`` key that merge ops target.
        ops: List of merge ops targeting this weight.
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

    gs = group_size or 32
    original_shape = tuple(shape_tensor.tolist())
    assert len(original_shape) == 2, f"Expected 2D shape, got {original_shape}"

    # Dequantize INT4 → bf16
    weight_bf16 = dequantize_int4_group(packed_tensor, scale_tensor, original_shape, gs)

    # Apply merge ops on the dequantized weight
    temp = {virtual_key: weight_bf16}
    for op in ops:
        apply_merge_op(temp, op)
    weight_bf16 = temp[virtual_key]

    # Re-quantize bf16 → INT4
    new_packed, new_scale = quantize_int4_group(weight_bf16, gs)

    # Update tensors in-place
    tensors[packed_key] = new_packed
    tensors[scale_key] = new_scale
    # weight_shape stays unchanged

    return len(ops)


def _get_int4_group_size(config_dict: dict) -> int:
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
