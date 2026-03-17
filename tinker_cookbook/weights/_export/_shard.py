"""Shard-by-shard export strategy.

Processes one safetensors shard at a time, keeping peak memory proportional to
the largest shard rather than the full model. Produces output identical to the
full-model path.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from safetensors.torch import load_file

from tinker_cookbook.weights._artifacts import (
    ShardWriter,
    copy_model_code_files,
    get_model_state_keys,
    get_shard_files,
    load_adapter_weights,
    resolve_model_dir,
)
from tinker_cookbook.weights._export import (
    cleanup_on_failure,
    is_multimodal_from_dict,
    load_config_dict,
    save_tokenizer_and_processor,
)
from tinker_cookbook.weights._merge import (
    apply_merge_op,
    detect_merge_profile,
    plan_merge_ops,
)

logger = logging.getLogger(__name__)


def build_sharded(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    trust_remote_code: bool,
) -> None:
    """Merge by processing one safetensors shard at a time."""
    # 1. Resolve model directory (local or download from HF Hub)
    model_dir = resolve_model_dir(base_model)

    # 2. Load adapter (small — only LoRA matrices)
    adapter_weights, adapter_config = load_adapter_weights(Path(adapter_path))

    # 3. Read model state keys from safetensors headers (no weight loading)
    model_state_keys = get_model_state_keys(model_dir)

    # 4. Detect model-specific merge profile from config.json + key names
    config_dict = load_config_dict(model_dir)
    profile = detect_merge_profile(config_dict, model_state_keys)
    logger.info(
        "Detected merge profile: expert_layout=%s, language_model_prefix=%s",
        profile.expert_layout,
        profile.has_language_model_prefix,
    )

    # 5. Plan all merge ops (validates keys before any heavy I/O)
    merge_ops = plan_merge_ops(adapter_weights, adapter_config, model_state_keys, profile)
    total_ops = sum(len(ops) for ops in merge_ops.values())
    logger.info("Planned %d merge operations across %d target keys", total_ops, len(merge_ops))

    # 6. Process shards
    out = Path(output_path)
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
                for op in ops_for_key:
                    apply_merge_op(tensors, op)
                    ops_applied += 1

            # Write all tensors from this shard to output
            for key, tensor in tensors.items():
                writer.add_tensor(key, tensor)
            del tensors
            writer.flush()

        # 7. Verify all ops were consumed
        if merge_ops:
            unconsumed = list(merge_ops.keys())
            raise RuntimeError(
                f"Merge ops not applied — {len(unconsumed)} target keys not found in any shard: "
                f"{unconsumed[:5]}{'...' if len(unconsumed) > 5 else ''}"
            )

        logger.info("Applied %d/%d merge operations", ops_applied, total_ops)

        # 8. Finalize output shards
        weight_map = writer.finalize()

        # 9. Write index file (only for multi-shard output; HF convention
        #    is no index for single-shard models)
        shard_names = set(weight_map.values())
        if len(shard_names) > 1:
            index = {
                "metadata": {"total_size": writer.total_size},
                "weight_map": dict(sorted(weight_map.items())),
            }
            index_path = out / "model.safetensors.index.json"
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

        # 10. Save config, tokenizer, and model code files.
        #     Copy config.json directly (safe — it's a single known file).
        #     Copy *.py files for trust_remote_code model support.
        #     We intentionally don't glob-copy all non-weight files to avoid
        #     accidentally including stale index files or other artifacts that
        #     could break downstream loaders like vLLM/SGLang.
        src_config = model_dir / "config.json"
        if src_config.exists():
            shutil.copy2(src_config, out / "config.json")
        copy_model_code_files(model_dir, out)
        save_tokenizer_and_processor(
            base_model, out, is_multimodal_from_dict(config_dict), trust_remote_code
        )

        logger.info("Done — merged model saved to %s", out)
    except Exception:
        cleanup_on_failure(out)
        raise
