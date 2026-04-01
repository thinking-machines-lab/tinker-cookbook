"""Shard-by-shard export strategy.

Processes one safetensors shard at a time, keeping peak memory proportional to
the largest shard rather than the full model. Produces output identical to the
full-model path.

Model-specific shard processing (e.g. INT4 dequant/requant for Kimi K2.5)
lives in dedicated ``_shard_<model>.py`` modules and is invoked via hooks
that this file dispatches to based on the detected merge profile.
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
    MergeOp,
    MergeProfile,
    apply_merge_op,
    detect_merge_profile,
    plan_merge_ops,
    validate_merge_op_shapes,
)

logger = logging.getLogger(__name__)


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

    # 4. Model-specific pre-planning (e.g. virtual key creation for packed weights)
    shard_hooks = _build_shard_hooks(profile, config_dict)
    if shard_hooks is not None:
        model_state_keys, model_shapes = shard_hooks.augment_for_planning(
            model_state_keys, model_shapes
        )

    # 5. Plan all merge ops (validates keys before any heavy I/O)
    merge_ops = plan_merge_ops(adapter_weights, adapter_config, model_state_keys, profile)
    total_ops = sum(len(ops) for ops in merge_ops.values())
    logger.info("Planned %d merge operations across %d target keys", total_ops, len(merge_ops))

    # 6. Validate shapes upfront (catches mismatches before loading any shards)
    validate_merge_op_shapes(merge_ops, model_shapes)

    # 7. Process shards
    out.mkdir(parents=True, exist_ok=False)

    try:
        shard_files = get_shard_files(model_dir)
        logger.info("Processing %d input shard(s)", len(shard_files))

        writer = ShardWriter(out)
        ops_applied = 0

        for i, shard_file in enumerate(shard_files):
            logger.info("Processing shard %d/%d: %s", i + 1, len(shard_files), shard_file)
            tensors = load_file(str(model_dir / shard_file))

            # Apply any merge ops targeting keys in this shard
            for key in list(tensors.keys()):
                ops_for_key = merge_ops.pop(key, [])
                if ops_for_key:
                    for op in ops_for_key:
                        apply_merge_op(tensors, op)
                        ops_applied += 1
                    continue

                # Model-specific handling (e.g. INT4 packed weights)
                if shard_hooks is not None:
                    ops_applied += shard_hooks.try_apply(key, tensors, merge_ops)

            # Write all tensors from this shard to output
            for key, tensor in tensors.items():
                writer.add_tensor(key, tensor)
            del tensors
            writer.flush()

        # 8. Verify all ops were consumed
        if merge_ops:
            unconsumed = list(merge_ops.keys())
            raise WeightsMergeError(
                f"Merge ops not applied — {len(unconsumed)} target keys not found in any shard: "
                f"{unconsumed[:5]}{'...' if len(unconsumed) > 5 else ''}"
            )

        logger.info("Applied %d/%d merge operations", ops_applied, total_ops)

        # 9. Finalize output shards
        weight_map = writer.finalize()

        # 10. Write index file (only for multi-shard output; HF convention
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

        # 11. Save config, tokenizer, and model code files.
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
# Model-specific shard hooks
# ---------------------------------------------------------------------------


class _ShardHooks:
    """Interface for model-specific shard processing hooks."""

    def augment_for_planning(
        self,
        model_state_keys: set[str],
        model_shapes: dict[str, tuple[int, ...]],
    ) -> tuple[set[str], dict[str, tuple[int, ...]]]:
        """Augment model state before planning (e.g. add virtual keys)."""
        return model_state_keys, model_shapes

    def try_apply(
        self,
        key: str,
        tensors: dict,
        merge_ops: dict[str, list[MergeOp]],
    ) -> int:
        """Try to apply model-specific merge handling for a key. Returns ops applied."""
        return 0


class _PackedInt4ShardHooks(_ShardHooks):
    """INT4 packed expert weights: dequant → merge → requant.

    Used for models with compressed-tensors ``pack-quantized`` format
    (e.g. Kimi K2, K2.5) where routed expert weights are stored as
    ``weight_packed``/``weight_scale``/``weight_shape``.
    """

    def __init__(self, config_dict: dict) -> None:
        from tinker_cookbook.weights._export._shard_packed_int4 import get_int4_group_size

        self._group_size = get_int4_group_size(config_dict)
        self._packed_map: dict[str, str] = {}

    def augment_for_planning(self, model_state_keys, model_shapes):
        from tinker_cookbook.weights._export._shard_packed_int4 import (
            augment_model_state_for_planning,
        )

        augmented_keys, augmented_shapes, self._packed_map = augment_model_state_for_planning(
            model_state_keys, model_shapes
        )
        return augmented_keys, augmented_shapes

    def try_apply(self, key, tensors, merge_ops):
        if not self._packed_map:
            return 0
        from tinker_cookbook.weights._export._shard_packed_int4 import try_apply_packed_ops

        return try_apply_packed_ops(key, tensors, merge_ops, self._packed_map, self._group_size)


def _build_shard_hooks(
    profile: MergeProfile,
    config_dict: dict,
) -> _ShardHooks | None:
    """Build shard hooks based on the model's quantization config.

    Uses the explicit ``format: pack-quantized`` field from the model's
    ``quantization_config`` to detect compressed-tensors INT4 format
    (Kimi K2, K2.5).  Other quantized formats (Nemotron NVFP4, DeepSeek
    native FP8) are NOT matched — they use different weight conventions.
    """
    from tinker_cookbook.weights._merge_utils import is_pack_quantized

    if is_pack_quantized(config_dict):
        return _PackedInt4ShardHooks(config_dict)
    return None
