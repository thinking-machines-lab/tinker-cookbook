"""Shared shard-by-shard merge engine.

Provides :func:`run_shard_merge`, the core shard iteration loop used by all
shard-based export strategies. Format-specific behavior is injected via
:class:`~._quant_format.ShardHooks` and
:class:`~._quant_format.QuantizationFormat` protocols.

This module eliminates the duplicated shard loops that previously existed
across ``_shard.py`` (standard merge) and ``_quantized.py`` (FP8 quantized
merge).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

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
from tinker_cookbook.weights._export._quant_format import (
    QuantizationFormat,
    ShardHooks,
)
from tinker_cookbook.weights._merge import (
    apply_merge_op,
    detect_merge_profile,
    plan_merge_ops,
    validate_merge_op_shapes,
)

logger = logging.getLogger(__name__)


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
# Total size computation for index file
# ---------------------------------------------------------------------------

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
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                sl = f.get_slice(key)
                numel = 1
                for dim in sl.get_shape():
                    numel *= dim
                total += numel * _DTYPE_SIZES.get(sl.get_dtype(), 4)
    return total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_shard_merge(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    trust_remote_code: bool,
    model_dir: Path,
    config_dict: dict,
    shard_hooks: ShardHooks | None = None,
    quant_format: QuantizationFormat | None = None,
    resume: bool = False,
    preserve_shard_names: bool = False,
) -> None:
    """Core shard-by-shard merge loop.

    Processes one safetensors shard at a time, applying LoRA merge ops and
    optional quantization transforms. Supports two output strategies:

    - **ShardWriter** (default): Repacks tensors by size, generates numbered
      output shard names. Used by the standard merge path.
    - **Preserve shard names** (``preserve_shard_names=True``): Writes
      output shards with the same filenames as input, enabling resume by
      shard name. Used by quantized export.

    Args:
        base_model: Model name or path (for tokenizer loading).
        adapter_path: Path to adapter directory.
        output_path: Where to write the merged model.
        trust_remote_code: Whether to trust remote code.
        model_dir: Resolved local model directory.
        config_dict: Parsed config.json dict.
        shard_hooks: Optional hooks for pre-quantized weights
            (dequant → merge → requant).
        quant_format: Optional post-merge quantization format.
        resume: Enable resume state tracking. Only used with
            ``preserve_shard_names=True``.
        preserve_shard_names: If True, output shards keep input filenames
            and are written atomically (enables resume). If False, uses
            :class:`~tinker_cookbook.weights._artifacts.ShardWriter`.
    """
    if resume and not preserve_shard_names:
        raise ValueError("resume=True requires preserve_shard_names=True")

    out = Path(output_path)

    # --- Resume / output directory setup ---
    resume_state: dict = {}
    if resume and out.exists():
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
    elif out.exists():
        raise FileExistsError(f"Output path already exists: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)

    try:
        _run_shard_merge_inner(
            base_model=base_model,
            adapter_path=adapter_path,
            out=out,
            trust_remote_code=trust_remote_code,
            model_dir=model_dir,
            config_dict=config_dict,
            shard_hooks=shard_hooks,
            quant_format=quant_format,
            resume=resume,
            preserve_shard_names=preserve_shard_names,
            resume_state=resume_state,
        )
    except Exception:
        if not resume:
            cleanup_on_failure(out)
        raise


def _run_shard_merge_inner(
    *,
    base_model: str,
    adapter_path: str,
    out: Path,
    trust_remote_code: bool,
    model_dir: Path,
    config_dict: dict,
    shard_hooks: ShardHooks | None,
    quant_format: QuantizationFormat | None,
    resume: bool,
    preserve_shard_names: bool,
    resume_state: dict,
) -> None:
    """Inner merge loop, separated for clean exception handling."""

    # 1. Load adapter (small — only LoRA matrices)
    adapter_weights, adapter_config = load_adapter_weights(Path(adapter_path))

    # 2. Read model state shapes from safetensors headers (no weight loading)
    model_shapes = get_model_state_shapes(model_dir)
    model_state_keys = set(model_shapes.keys())

    # 3. Shard hooks augmentation (e.g. virtual keys for INT4 packed weights)
    if shard_hooks is not None:
        model_state_keys, model_shapes = shard_hooks.augment_for_planning(
            model_state_keys, model_shapes
        )

    # 4. Quant format key filtering (e.g. exclude native scale tensors)
    if quant_format is not None:
        filtered_keys = quant_format.filter_model_keys(model_state_keys)
        filtered_shapes = {k: v for k, v in model_shapes.items() if k in filtered_keys}
    else:
        filtered_keys = model_state_keys
        filtered_shapes = model_shapes

    # 5. Detect model-specific merge profile and plan ops
    profile = detect_merge_profile(config_dict, model_state_keys)
    logger.info(
        "Detected merge profile: family=%s, expert_layout=%s, language_model_prefix=%s",
        profile.model_family,
        profile.expert_layout,
        profile.has_language_model_prefix,
    )

    merge_ops = plan_merge_ops(adapter_weights, adapter_config, filtered_keys, profile)
    total_ops = sum(len(ops) for ops in merge_ops.values())
    logger.info("Planned %d merge operations across %d target keys", total_ops, len(merge_ops))

    # 6. Validate shapes upfront (catches mismatches before loading any shards)
    validate_merge_op_shapes(merge_ops, filtered_shapes)

    # 7. Process shards
    shard_files = get_shard_files(model_dir)
    completed_shards = set(resume_state.get("completed_shards", []))
    all_completed: list[str] = list(completed_shards)
    weight_map: dict[str, str] = {}

    # Rebuild weight map from already-completed shards (resume support).
    # Uses safe_open to read only headers — avoids loading tensor data.
    if completed_shards:
        for shard_name in completed_shards:
            with safe_open(str(out / shard_name), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    weight_map[key] = shard_name
                    merge_ops.pop(key, None)

    logger.info(
        "Processing %d input shard(s)%s",
        len(shard_files),
        f" ({len(completed_shards)} already completed)" if completed_shards else "",
    )

    writer = None if preserve_shard_names else ShardWriter(out)
    ops_applied = 0

    for i, shard_file in enumerate(shard_files):
        out_shard_name = shard_file

        if out_shard_name in completed_shards:
            logger.info("Skipping completed shard %d/%d: %s", i + 1, len(shard_files), shard_file)
            continue

        logger.info("Processing shard %d/%d: %s", i + 1, len(shard_files), shard_file)
        tensors = load_file(str(model_dir / shard_file))

        if quant_format is not None:
            output_tensors: dict[str, torch.Tensor] = {}

            for key in list(tensors.keys()):
                tensor = tensors[key]

                if quant_format.should_skip_output_key(key):
                    continue

                tensor = quant_format.pre_merge_transform(key, tensor, tensors)

                ops_for_key = merge_ops.pop(key, [])
                if ops_for_key:
                    temp = {key: tensor}
                    for op in ops_for_key:
                        apply_merge_op(temp, op)
                        ops_applied += 1
                    tensor = temp[key]

                transformed = quant_format.post_merge_transform(key, tensor)
                output_tensors.update(transformed)
                for out_key in transformed:
                    weight_map[out_key] = out_shard_name

            del tensors
            _save_shard_atomic(out, out_shard_name, output_tensors)
            del output_tensors

        else:
            for key in list(tensors.keys()):
                ops_for_key = merge_ops.pop(key, [])
                if ops_for_key:
                    for op in ops_for_key:
                        apply_merge_op(tensors, op)
                        ops_applied += 1
                    continue

                if shard_hooks is not None:
                    ops_applied += shard_hooks.try_apply(key, tensors, merge_ops)

            assert writer is not None
            for key, tensor in tensors.items():
                writer.add_tensor(key, tensor)
            del tensors
            writer.flush()

        # Resume state tracking
        if resume:
            all_completed.append(out_shard_name)
            _save_merge_state(
                out,
                status="in_progress",
                completed_shards=all_completed,
                total_shards=len(shard_files),
            )

    # 8. Verify all merge ops were consumed
    if merge_ops:
        unconsumed = list(merge_ops.keys())
        raise WeightsMergeError(
            f"Merge ops not applied — {len(unconsumed)} target keys not found in any shard: "
            f"{unconsumed[:5]}{'...' if len(unconsumed) > 5 else ''}"
        )

    logger.info("Applied %d/%d merge operations", ops_applied, total_ops)

    # 9. Finalize output shards and build weight map
    if writer is not None:
        weight_map = writer.finalize()

    # 10. Write index file
    shard_names = set(weight_map.values())
    if preserve_shard_names or len(shard_names) > 1:
        if preserve_shard_names:
            total_size = _compute_total_size(out, shard_names)
        else:
            assert writer is not None
            total_size = writer.total_size
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        index_path = out / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    # 11. Copy config.json from source model, then apply format-specific patches.
    #     Always start from the raw file to preserve the original config format
    #     (AutoConfig serialization adds default fields that weren't in the original).
    src_config = model_dir / "config.json"
    if src_config.exists():
        copy_artifact_file(src_config, out / "config.json")

    if quant_format is not None:
        config_path = out / "config.json"
        with open(config_path) as f:
            on_disk_config = json.load(f)
        patched_config = quant_format.finalize_config(on_disk_config, weight_map)
        with open(config_path, "w") as f:
            json.dump(patched_config, f, indent=2)

    # 12. Copy model code and tokenizer
    copy_model_code_files(model_dir, out)
    save_tokenizer_and_processor(base_model, out, is_multimodal_from_dict(config_dict))

    # 13. Mark complete (resume support)
    if resume:
        _save_merge_state(
            out,
            status="completed",
            completed_shards=all_completed,
            total_shards=len(shard_files),
        )

    logger.info("Done — merged model saved to %s", out)
