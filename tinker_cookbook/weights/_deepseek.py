"""DeepSeek-specific mixed-precision export helpers.

This exporter has two jobs:

1. Quantization selection: only routed expert MLP weights are quantized to FP8,
   while dense, shared-expert, embedding, attention, and norm weights stay in
   higher precision.
2. Checkpoint packaging: process one reference shard at a time on CPU, preserve
   the upstream shard names and tensor placement, and emit a resumable output
   index that tracks the shards already written.

Preserving the reference shard layout is not a mathematical requirement for
checkpoint correctness. It is the chosen implementation strategy here because it
makes the work unit "one reference shard", keeps DeepSeek cross-shard scale
handling straightforward, and makes interrupted exports easy to resume.
"""

from __future__ import annotations

import errno
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel

from tinker_cookbook.weights._merge import apply_merged_weight

__all__ = ["is_deepseek_config", "is_deepseek_model", "save_deepseek_model"]

logger = logging.getLogger(__name__)

_DEEPSEEK_MODEL_TYPES = frozenset({"deepseek_v3"})
_DEEPSEEK_CONFIG_CLASS_NAMES = frozenset({"deepseekv3config"})
_DEEPSEEK_CUSTOM_FILES = (
    "configuration_deepseek.py",
    "modeling_deepseek.py",
)
_DEEPSEEK_ROUTED_EXPERT_PROJ_NAMES = (
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)
_DEEPSEEK_BLOCK_SIZE = (128, 128)
_DEEPSEEK_FP8_DTYPE = torch.float8_e4m3fn
_LINEAR_PROJ_SUFFIXES = (
    ".q_proj.weight",
    ".q_a_proj.weight",
    ".q_b_proj.weight",
    ".kv_a_proj_with_mqa.weight",
    ".kv_b_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)
_CHECKPOINT_ALLOW_PATTERNS = [
    "*.json",
    "*.py",
    "*.safetensors",
]
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
_VLLM_COMPAT_QUANT_SCHEME_FIELDS = {
    "format",
    "input_activations",
    "output_activations",
    "targets",
    "weights",
}
_VLLM_COMPAT_QUANT_CONFIG_FIELDS = {
    "config_groups",
    "format",
    "global_compression_ratio",
    "ignore",
    "kv_cache_scheme",
    "quantization_status",
}


class _HasQuantizationConfig(Protocol):
    quantization_config: Any


def is_deepseek_config(config: object) -> bool:
    """Return True when a config should use the DeepSeek export path."""
    if isinstance(config, dict):
        model_type = str(config.get("model_type", "")).lower()
        class_name = ""
    else:
        model_type = str(getattr(config, "model_type", "")).lower()
        class_name = config.__class__.__name__.lower()
    return model_type in _DEEPSEEK_MODEL_TYPES or class_name in _DEEPSEEK_CONFIG_CLASS_NAMES


def is_deepseek_model(hf_model: PreTrainedModel) -> bool:
    """Return True when the loaded model should use the DeepSeek export path."""
    return is_deepseek_config(hf_model.config)


def save_deepseek_model(
    *,
    base_model: str,
    adapter_path: str,
    output_path: Path,
) -> None:
    """Save a DeepSeek checkpoint with experts-only FP8 quantization on CPU.

    This path never loads a full ``PreTrainedModel``. It reads the upstream
    checkpoint shard-by-shard on CPU, applies LoRA deltas to the tensors that
    live in each shard, and writes the final mixed-precision output using the
    upstream shard names as resumable work units.
    """
    checkpoint_path = _resolve_hf_checkpoint_path(base_model)
    logger.info("DeepSeek export: resolved checkpoint path to %s", checkpoint_path)
    config = _load_checkpoint_config(checkpoint_path)
    shard_paths = _get_checkpoint_shard_paths(checkpoint_path)
    logger.info("DeepSeek export: found %d reference shard(s)", len(shard_paths))

    all_model_keys = _collect_model_keys(shard_paths)
    logger.info("DeepSeek export: discovered %d model tensor keys", len(all_model_keys))

    logger.info("DeepSeek export: loading adapter weights on CPU")
    adapter_weights, adapter_config = _load_adapter_weights(adapter_path)
    merge_ops = _build_merge_ops(
        adapter_weights=adapter_weights,
        adapter_config=adapter_config,
        current_keys=all_model_keys,
    )
    logger.info(
        "DeepSeek export: built %d merge op(s)",
        sum(len(ops) for ops in merge_ops.values()),
    )

    state, completed_shards = _load_resume_state(
        output_path=output_path,
        model_path=base_model,
        adapter_path=adapter_path,
    )
    load_merged_shard, load_merged_tensor = _make_merged_tensor_loader(
        checkpoint_path=checkpoint_path,
        config=config,
        merge_ops=merge_ops,
    )

    quantized_weight_keys = {key for key in all_model_keys if _is_routed_expert_weight(key)}
    weight_map = _load_checkpoint_weight_map(str(checkpoint_path))
    quantized_weight_keys_with_reference_scale = {
        key.removesuffix(".weight_scale_inv") + ".weight"
        for key in weight_map
        if key.endswith(".weight_scale_inv")
    }
    shard_progress_interval = _progress_interval(len(shard_paths))
    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        shard_name = shard_path.name
        output_shard_path = output_path / shard_name
        if shard_name in completed_shards and output_shard_path.exists():
            logger.info(
                "DeepSeek export: skipping completed shard %d/%d (%s)",
                shard_idx,
                len(shard_paths),
                shard_name,
            )
            continue

        logger.info(
            "DeepSeek export: processing shard %d/%d (%s)",
            shard_idx,
            len(shard_paths),
            shard_name,
        )
        reference_shard = _load_raw_shard(shard_path)
        merged_shard = load_merged_shard(shard_name)
        output_shard = _build_output_shard_matching_reference(
            reference_shard=reference_shard,
            merged_shard=merged_shard,
            quantized_weight_keys=quantized_weight_keys,
            quantized_weight_keys_with_reference_scale=quantized_weight_keys_with_reference_scale,
            load_merged_tensor=load_merged_tensor,
        )
        _save_shard_atomic(output_shard_path, output_shard)
        shard_weight_map, shard_total_size = _get_shard_index_data_from_tensors(
            shard_name, output_shard
        )
        state["weight_map"].update(shard_weight_map)
        state["total_size"] += shard_total_size
        completed_shards.add(shard_name)
        state["completed_shards"] = sorted(completed_shards)
        _write_json_atomic(output_path / "merge_state.json", state)
        _write_output_index(output_path, state)
        if (
            shard_idx == 1
            or shard_idx == len(shard_paths)
            or shard_idx % shard_progress_interval == 0
        ):
            logger.info(
                "DeepSeek export: wrote shard %d/%d (%s, %d tensors)",
                shard_idx,
                len(shard_paths),
                shard_name,
                len(output_shard),
            )

    logger.info("DeepSeek export: saving config files")
    _save_base_config_files(checkpoint_path=checkpoint_path, output_path=output_path)
    logger.info("DeepSeek export: patching config.json for compressed-tensors")
    _patch_config(output_path=output_path, output_weight_map=state["weight_map"])
    logger.info("DeepSeek export: copying custom DeepSeek files")
    _copy_custom_files(output_path=output_path, checkpoint_path=checkpoint_path)
    state["status"] = "completed"
    _write_json_atomic(output_path / "merge_state.json", state)
    logger.info("DeepSeek export: finished")


def _is_routed_expert_weight(key: str) -> bool:
    """Return True only for routed expert projection weights.

    This intentionally excludes shared experts and every non-expert tensor so
    the export stays in the "quantize only routed experts" regime.
    """
    return (
        ".mlp.experts." in key
        and ".shared_experts." not in key
        and key.endswith(_DEEPSEEK_ROUTED_EXPERT_PROJ_NAMES)
    )


def _should_keep_float32(key: str) -> bool:
    return key.endswith("e_score_correction_bias")


def _weight_scale_key(weight_key: str) -> str:
    return weight_key.removesuffix(".weight") + ".weight_scale"


def _resolve_hf_checkpoint_path(model_path: str) -> Path:
    """Resolve ``model_path`` to a local directory containing checkpoint shards."""
    candidate = Path(model_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    snapshot_path = snapshot_download(model_path, allow_patterns=_CHECKPOINT_ALLOW_PATTERNS)
    checkpoint_path = Path(snapshot_path)
    if not any(checkpoint_path.glob("*.safetensors")):
        raise FileNotFoundError(f"Checkpoint has no .safetensors files: {checkpoint_path}")
    return checkpoint_path


def _get_checkpoint_shard_paths(checkpoint_path: Path) -> list[Path]:
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
        shard_names = sorted(set(index_data["weight_map"].values()))
        return [checkpoint_path / shard_name for shard_name in shard_names]

    shard_paths = sorted(checkpoint_path.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors files found in {checkpoint_path}")
    return shard_paths


def _load_checkpoint_config(checkpoint_path: Path) -> dict:
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint is missing config.json: {checkpoint_path}")
    return json.loads(config_path.read_text())


@cache
def _load_checkpoint_weight_map(checkpoint_path: str) -> dict[str, str]:
    index_path = Path(checkpoint_path) / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    return json.loads(index_path.read_text())["weight_map"]


def _make_checkpoint_tensor_loader(checkpoint_path: Path):
    weight_map = _load_checkpoint_weight_map(str(checkpoint_path))
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    def load_tensor(name: str) -> torch.Tensor:
        if name not in weight_map:
            raise KeyError(f"Tensor {name} not found in checkpoint index at {checkpoint_path}")
        shard_name = weight_map[name]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = _load_raw_shard(checkpoint_path / shard_name)
        shard_dict = shard_cache[shard_name]
        if name not in shard_dict:
            raise KeyError(f"Tensor {name} not found in shard {shard_name} at {checkpoint_path}")
        return shard_dict[name]

    return load_tensor


def _has_native_fp8_quantization(config: object) -> bool:
    if isinstance(config, dict):
        quant_config = config.get("quantization_config")
    else:
        quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        return False
    if isinstance(quant_config, dict):
        method = quant_config.get("quant_method", "")
    else:
        method = getattr(quant_config, "quant_method", "") or ""
    return method == "fp8"


def _get_quant_config_dict(config: object) -> dict:
    if isinstance(config, dict):
        quant_config = config["quantization_config"]
    else:
        quant_config = cast(_HasQuantizationConfig, config).quantization_config
    if isinstance(quant_config, dict):
        return quant_config
    return quant_config.to_dict()


def _dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    block_h, block_w = block_size
    rows, cols = weight.shape
    scale = scale_inv.repeat_interleave(block_h, dim=0).repeat_interleave(block_w, dim=1)
    scale = scale[:rows, :cols]
    return (weight.float() * scale).to(output_dtype)


def _quantize_weight_blockwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D expert weight tensor, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    block_rows, block_cols = _DEEPSEEK_BLOCK_SIZE
    scale_rows = math.ceil(rows / block_rows)
    scale_cols = math.ceil(cols / block_cols)
    max_fp8 = _get_fp8_max()
    padded_rows = scale_rows * block_rows
    padded_cols = scale_cols * block_cols

    padded = torch.zeros((padded_rows, padded_cols), dtype=torch.float32, device=weight.device)
    padded[:rows, :cols] = weight.to(torch.float32)
    block_view = padded.view(scale_rows, block_rows, scale_cols, block_cols)
    max_abs = block_view.abs().amax(dim=(1, 3))
    scales = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / max_fp8)

    quantized = (
        (block_view / scales[:, None, :, None])
        .clamp(min=-max_fp8, max=max_fp8)
        .to(_DEEPSEEK_FP8_DTYPE)
    )

    return (
        quantized.view(padded_rows, padded_cols)[:rows, :cols].contiguous(),
        scales.to(torch.float32).contiguous(),
    )


def _get_fp8_max() -> float:
    try:
        return float(torch.finfo(_DEEPSEEK_FP8_DTYPE).max)
    except TypeError:
        return 448.0


def _progress_interval(total_items: int) -> int:
    if total_items <= 0:
        return 1
    return max(1, total_items // 20)


def _normalize_output_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor
    if _should_keep_float32(name):
        return tensor.to(torch.float32)
    return tensor.to(torch.bfloat16)


def _load_raw_shard(shard_path: Path) -> dict[str, torch.Tensor]:
    shard_dict: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        for name in tuple(handle.keys()):
            if _should_skip_checkpoint_key(name):
                continue
            shard_dict[name] = handle.get_tensor(name)
    return shard_dict


def _load_export_shard(shard_path: Path, config: object) -> dict[str, torch.Tensor]:
    shard_dict = _load_raw_shard(shard_path)
    if not _has_native_fp8_quantization(config):
        return {
            name: _normalize_output_tensor(name, tensor)
            for name, tensor in shard_dict.items()
            if not name.endswith(".weight_scale_inv")
        }

    quant_config = _get_quant_config_dict(config)
    block_size = tuple(quant_config.get("weight_block_size", list(_DEEPSEEK_BLOCK_SIZE)))
    tensor_loader = _make_checkpoint_tensor_loader(shard_path.parent)
    fp8_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
    export_shard: dict[str, torch.Tensor] = {}
    for name, tensor in shard_dict.items():
        if name.endswith(".weight_scale_inv"):
            continue
        if name.endswith(".weight") and tensor.dtype in fp8_dtypes:
            scale_key = name.removesuffix(".weight") + ".weight_scale_inv"
            scale_inv = shard_dict.get(scale_key)
            if scale_inv is None:
                scale_inv = tensor_loader(scale_key)
            export_shard[name] = _dequantize_fp8_blockwise(
                tensor,
                scale_inv,
                block_size,
                output_dtype=torch.bfloat16,
            )
            continue
        export_shard[name] = _normalize_output_tensor(name, tensor)
    return export_shard


def _collect_model_keys(shard_paths: list[Path]) -> set[str]:
    model_keys: set[str] = set()
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in tuple(handle.keys()):
                if _should_skip_checkpoint_key(name) or name.endswith(".weight_scale_inv"):
                    continue
                model_keys.add(name)
    return model_keys


@dataclass
class MergeOp:
    target_key: str
    lora_A: torch.Tensor
    lora_B: torch.Tensor


def _build_name_remaps(model_keys: set[str]) -> dict[str, str]:
    name_remaps = {
        "base_model.model.": "",
        "model.unembed_tokens": "lm_head",
    }
    if any(key.startswith("model.language_model.") for key in model_keys):
        name_remaps["model."] = "model.language_model."
    return name_remaps


def _expand_expert_lora_tensors(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lora_A.shape[0] == 1:
        if lora_B.shape[0] <= 1:
            raise ValueError(
                "Cannot broadcast expert LoRA: both A and B have 1 expert "
                f"(lora_A: {lora_A.shape}, lora_B: {lora_B.shape})"
            )
        lora_A = lora_A.expand(lora_B.shape[0], -1, -1)
    elif lora_B.shape[0] == 1:
        if lora_A.shape[0] <= 1:
            raise ValueError(
                "Cannot broadcast expert LoRA: both A and B have 1 expert "
                f"(lora_A: {lora_A.shape}, lora_B: {lora_B.shape})"
            )
        lora_B = lora_B.expand(lora_A.shape[0], -1, -1)
    return lora_A, lora_B


def _build_merge_ops(
    *,
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict,
    current_keys: set[str],
) -> dict[str, list[MergeOp]]:
    scaling = adapter_config["lora_alpha"] / adapter_config["r"]
    name_remaps = _build_name_remaps(current_keys)
    adapter_weight_names = [
        name.replace(".lora_A", "") for name in adapter_weights if ".lora_A" in name
    ]
    merge_ops: dict[str, list[MergeOp]] = {}
    missing_target_keys: set[str] = set()
    progress_interval = _progress_interval(len(adapter_weight_names))

    for idx, name in enumerate(adapter_weight_names, start=1):
        if idx == 1 or idx == len(adapter_weight_names) or idx % progress_interval == 0:
            logger.info(
                "DeepSeek export: building merge op %d/%d (%s)",
                idx,
                len(adapter_weight_names),
                name,
            )
        target_key = name
        for old, new in name_remaps.items():
            target_key = target_key.replace(old, new)

        lora_A = adapter_weights[name.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[name.replace(".weight", ".lora_B.weight")].float() * scaling
        if ".experts" not in name:
            if target_key not in current_keys:
                missing_target_keys.add(target_key)
                continue
            merge_ops.setdefault(target_key, []).append(
                MergeOp(target_key=target_key, lora_A=lora_A, lora_B=lora_B)
            )
            continue

        lora_A, lora_B = _expand_expert_lora_tensors(lora_A, lora_B)
        target_key = target_key.replace(".w1.weight", ".gate_proj.weight")
        target_key = target_key.replace(".w3.weight", ".up_proj.weight")
        target_key = target_key.replace(".w2.weight", ".down_proj.weight")
        for expert_idx in range(lora_A.shape[0]):
            expert_key = target_key.replace(".experts", f".experts.{expert_idx}")
            if expert_key not in current_keys:
                missing_target_keys.add(expert_key)
                continue
            merge_ops.setdefault(expert_key, []).append(
                MergeOp(target_key=expert_key, lora_A=lora_A[expert_idx], lora_B=lora_B[expert_idx])
            )

    if missing_target_keys:
        logger.warning(
            "DeepSeek export: skipped %d adapter target(s) missing from checkpoint, first few: %s",
            len(missing_target_keys),
            sorted(missing_target_keys)[:10],
        )
    return merge_ops


def _apply_merge_ops_to_state_dict(
    shard_state_dict: dict[str, torch.Tensor],
    merge_ops: dict[str, list[MergeOp]],
) -> None:
    with torch.no_grad():
        for target_key, target_tensor in shard_state_dict.items():
            if target_key not in merge_ops:
                continue
            for op in merge_ops[target_key]:
                merged_lora = op.lora_B @ op.lora_A
                apply_merged_weight(target_tensor, merged_lora)


def _make_merged_tensor_loader(
    *,
    checkpoint_path: Path,
    config: object,
    merge_ops: dict[str, list[MergeOp]],
):
    weight_map = _load_checkpoint_weight_map(str(checkpoint_path))

    @lru_cache(maxsize=4)
    def load_merged_shard(shard_name: str) -> dict[str, torch.Tensor]:
        merged_shard = _load_export_shard(checkpoint_path / shard_name, config)
        if merge_ops:
            _apply_merge_ops_to_state_dict(merged_shard, merge_ops)
        return merged_shard

    def load_merged_tensor(name: str) -> torch.Tensor:
        if name not in weight_map:
            raise KeyError(f"Tensor {name} not found in checkpoint index at {checkpoint_path}")
        shard_name = weight_map[name]
        merged_shard = load_merged_shard(shard_name)
        if name not in merged_shard:
            raise KeyError(f"Merged tensor {name} missing from virtual shard {shard_name}")
        return merged_shard[name]

    return load_merged_shard, load_merged_tensor


def _build_output_shard_matching_reference(
    *,
    reference_shard: dict[str, torch.Tensor],
    merged_shard: dict[str, torch.Tensor],
    quantized_weight_keys: set[str],
    quantized_weight_keys_with_reference_scale: set[str],
    load_merged_tensor,
) -> dict[str, torch.Tensor]:
    quantized_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    placed_scale_keys: set[str] = set()

    def get_merged_tensor(name: str) -> torch.Tensor:
        if name in merged_shard:
            return merged_shard[name]
        return load_merged_tensor(name)

    def get_quantized_weight_and_scale(weight_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        if weight_key not in quantized_cache:
            quantized_cache[weight_key] = _quantize_weight_blockwise(get_merged_tensor(weight_key))
        return quantized_cache[weight_key]

    output_shard: dict[str, torch.Tensor] = {}
    for name in reference_shard:
        if name in quantized_weight_keys:
            output_shard[name] = get_quantized_weight_and_scale(name)[0]
            continue

        if name.endswith(".weight_scale_inv"):
            weight_key = name.removesuffix(".weight_scale_inv") + ".weight"
            if weight_key in quantized_weight_keys:
                scale_key = _weight_scale_key(weight_key)
                output_shard[scale_key] = get_quantized_weight_and_scale(weight_key)[1]
                placed_scale_keys.add(scale_key)
            continue

        output_shard[name] = get_merged_tensor(name)

    # Local or dequantized test fixtures may not have native `.weight_scale_inv`
    # slots in the reference shard layout. In that case, place the new public
    # `.weight_scale` tensor next to the routed expert weight so the output is
    # still self-consistent and loadable.
    for weight_key in quantized_weight_keys:
        if (
            weight_key in reference_shard
            and weight_key not in quantized_weight_keys_with_reference_scale
        ):
            scale_key = _weight_scale_key(weight_key)
            if scale_key not in placed_scale_keys:
                output_shard[scale_key] = get_quantized_weight_and_scale(weight_key)[1]
    return output_shard


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def _save_shard_atomic(path: Path, shard_dict: dict[str, torch.Tensor]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    save_file(
        {
            name: tensor.contiguous() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in shard_dict.items()
        },
        str(tmp_path),
        metadata={"format": "pt"},
    )
    os.replace(tmp_path, path)


def _get_shard_index_data_from_tensors(
    shard_name: str,
    shard_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, str], int]:
    weight_map = dict.fromkeys(shard_dict, shard_name)
    total_size = sum(_tensor_nbytes(tensor) for tensor in shard_dict.values())
    return weight_map, total_size


def _get_shard_index_data_from_file(shard_path: Path) -> tuple[dict[str, str], int]:
    weight_map: dict[str, str] = {}
    total_size = 0
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        for name in tuple(handle.keys()):
            tensor = handle.get_tensor(name)
            weight_map[name] = shard_path.name
            total_size += _tensor_nbytes(tensor)
    return weight_map, total_size


def _load_resume_state(
    *,
    output_path: Path,
    model_path: str,
    adapter_path: str,
) -> tuple[dict, set[str]]:
    state_path = output_path / "merge_state.json"
    expected = {
        "version": 1,
        "model_path": model_path,
        "adapter_path": str(Path(adapter_path).expanduser().resolve()),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Existing merge_state.json at {output_path} does not match this run for {key}"
                )
    else:
        existing_files = [
            path.name for path in output_path.iterdir() if path.name != "merge_state.json"
        ]
        if existing_files:
            raise ValueError(
                f"Output path {output_path} already exists and is not resumable. Remove it and retry."
            )
        state = {
            **expected,
            "status": "in_progress",
            "completed_shards": [],
            "total_size": 0,
            "weight_map": {},
        }
        _write_json_atomic(state_path, state)

    tracked_completed = set(state.get("completed_shards", []))
    existing_output_shards = {path.name for path in output_path.glob("*.safetensors")}
    unexpected_output_shards = sorted(existing_output_shards - tracked_completed)
    if unexpected_output_shards:
        raise ValueError(
            f"Output path {output_path} contains shard files not tracked in merge_state.json: "
            f"{unexpected_output_shards[:5]}"
        )

    completed = {
        shard_name for shard_name in tracked_completed if (output_path / shard_name).exists()
    }
    if completed:
        total_size = 0
        weight_map: dict[str, str] = {}
        for shard_name in sorted(completed):
            shard_weight_map, shard_total_size = _get_shard_index_data_from_file(
                output_path / shard_name
            )
            weight_map.update(shard_weight_map)
            total_size += shard_total_size
        state["weight_map"] = weight_map
        state["total_size"] = total_size
    state["completed_shards"] = sorted(completed)
    _write_json_atomic(state_path, state)
    return state, completed


def _write_output_index(output_path: Path, state: dict) -> None:
    _write_json_atomic(
        output_path / "model.safetensors.index.json",
        {
            "metadata": {"total_size": state["total_size"]},
            "weight_map": state["weight_map"],
        },
    )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _patch_config(output_path: Path, output_weight_map: dict[str, str]) -> None:
    config_path = output_path / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("quantization_config", None)
    config.pop("compression_config", None)
    config["compression_config"] = _build_compressed_tensors_config(output_weight_map)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    logger.info("DeepSeek export: updated %s with compressed-tensors config", config_path)


def _build_compressed_tensors_config(output_weight_map: dict[str, str]) -> dict:
    try:
        from compressed_tensors import QuantizationConfig
        from compressed_tensors.quantization import (
            QuantizationArgs,
            QuantizationScheme,
            QuantizationStatus,
            QuantizationStrategy,
            QuantizationType,
        )
    except ImportError as exc:
        raise ImportError(
            "DeepSeek experts-only FP8 export requires the optional merge dependencies. "
            "Install them with: pip install 'tinker_cookbook[deepseek]'"
        ) from exc

    quantized_prefixes = {
        key.removesuffix(".weight_scale")
        for key in output_weight_map
        if key.endswith(".weight_scale")
    }

    ignore: list[str] = []
    for key in sorted(output_weight_map):
        if not any(key.endswith(suffix) for suffix in _LINEAR_PROJ_SUFFIXES):
            continue
        prefix = key.removesuffix(".weight")
        if prefix not in quantized_prefixes:
            ignore.append(prefix)

    if "lm_head.weight" in output_weight_map and "lm_head" not in quantized_prefixes:
        ignore.append("lm_head")

    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.BLOCK,
                    block_structure=list(_DEEPSEEK_BLOCK_SIZE),
                    symmetric=True,
                    dynamic=False,
                ),
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=True,
                ),
            )
        },
        format="float-quantized",
        quantization_status=QuantizationStatus.COMPRESSED,
        ignore=ignore,
    )
    return _serialize_vllm_compatible_quant_config(config.model_dump())


def _serialize_vllm_compatible_quant_config(config_dict: dict) -> dict:
    """Serialize only the compressed-tensors fields the current vLLM path needs.

    Using an allowlist here is more stable than stripping a few known extras:
    new compressed-tensors fields are omitted automatically instead of breaking
    older vLLM builds until we update a blacklist.
    """
    serialized: dict = {}
    for key, value in config_dict.items():
        if key == "config_groups" and isinstance(value, dict):
            serialized[key] = {
                group_name: _serialize_vllm_compatible_quant_scheme(group)
                for group_name, group in value.items()
                if isinstance(group, dict)
            }
            continue
        if key in _VLLM_COMPAT_QUANT_CONFIG_FIELDS:
            serialized[key] = value
    serialized["quant_method"] = "compressed-tensors"
    return serialized


def _serialize_vllm_compatible_quant_scheme(group: dict) -> dict:
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


def _copy_custom_files(
    *,
    output_path: Path,
    checkpoint_path: Path,
) -> None:
    for file_name in _DEEPSEEK_CUSTOM_FILES:
        destination = output_path / file_name
        if destination.exists():
            logger.info("DeepSeek export: custom file already present: %s", destination)
            continue

        source = checkpoint_path / file_name
        if not source.exists():
            raise FileNotFoundError(f"Could not locate required DeepSeek file: {file_name}")
        _copy_file_robustly(source, destination)
        logger.info("DeepSeek export: copied %s from %s", file_name, checkpoint_path)


def _copy_file_robustly(source: Path, destination: Path) -> None:
    """Copy file contents even when metadata-preserving copies fail on /gcs."""
    try:
        shutil.copy2(source, destination)
        return
    except PermissionError:
        pass
    except OSError as exc:
        if exc.errno != errno.EPERM:
            raise

    shutil.copyfile(source, destination)
    logger.warning(
        "DeepSeek export: metadata-preserving copy failed for %s; copied file contents only",
        destination,
    )


def _save_base_config_files(*, checkpoint_path: Path, output_path: Path) -> None:
    _copy_file_robustly(checkpoint_path / "config.json", output_path / "config.json")
    generation_config_path = checkpoint_path / "generation_config.json"
    if generation_config_path.exists():
        _copy_file_robustly(generation_config_path, output_path / "generation_config.json")


def _load_adapter_weights(adapter_path: str) -> tuple[dict[str, torch.Tensor], dict]:
    adapter_dir = Path(adapter_path).expanduser().resolve()
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {safetensors_path}")

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")

    weights = load_file(str(safetensors_path), device="cpu")
    config = json.loads(config_path.read_text())
    return weights, config


def _should_skip_checkpoint_key(name: str) -> bool:
    # DeepSeek V3.1 HF checkpoints include a MultiTokenPrediction layer that is
    # not part of the base causal LM weights we want to merge and serve.
    return "model.layers.61" in name or "rotary_emb.inv_freq" in name
