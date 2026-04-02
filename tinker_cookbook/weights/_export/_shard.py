"""Shard-by-shard export strategy.

Processes one safetensors shard at a time, keeping peak memory proportional to
the largest shard rather than the full model. Produces output identical to the
full-model path.

Model-specific shard processing (e.g. INT4 dequant/requant for Kimi K2.5)
lives in dedicated ``_shard_<model>.py`` modules and is invoked via hooks
that this file dispatches to based on the detected merge profile.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tinker_cookbook.weights._export._quant_format import ShardHooks
from tinker_cookbook.weights._export._shard_engine import run_shard_merge

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
    shard_hooks = _build_shard_hooks(config_dict)

    run_shard_merge(
        base_model=base_model,
        adapter_path=adapter_path,
        output_path=output_path,
        trust_remote_code=trust_remote_code,
        model_dir=model_dir,
        config_dict=config_dict,
        shard_hooks=shard_hooks,
    )


# ---------------------------------------------------------------------------
# Model-specific shard hooks
# ---------------------------------------------------------------------------


def _build_shard_hooks(
    config_dict: dict,
) -> ShardHooks | None:
    """Build shard hooks based on the model's quantization config.

    Uses the explicit ``format: pack-quantized`` field from the model's
    ``quantization_config`` to detect compressed-tensors INT4 format
    (Kimi K2, K2.5).  Other quantized formats (Nemotron NVFP4, DeepSeek
    native FP8) are NOT matched — they use different weight conventions.
    """
    from tinker_cookbook.weights._merge_utils import is_pack_quantized

    if is_pack_quantized(config_dict):
        from tinker_cookbook.weights._export._shard_packed_int4 import PackedInt4ShardHooks

        return PackedInt4ShardHooks(config_dict)
    return None
