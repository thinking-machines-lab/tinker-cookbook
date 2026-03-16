"""DeepSeek-specific mixed-precision export helpers."""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

_DEFAULT_DEEPSEEK_REPO = "deepseek-ai/DeepSeek-V3.1"
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
_METADATA_ALLOW_PATTERNS = [
    "*.json",
    "*.py",
]


def is_deepseek_model(hf_model: PreTrainedModel) -> bool:
    """Return True when the loaded model should use the DeepSeek export path."""
    model_type = str(getattr(hf_model.config, "model_type", "")).lower()
    class_name = hf_model.__class__.__name__.lower()
    return model_type.startswith("deepseek") or class_name.startswith("deepseek")


def save_deepseek_model(
    *,
    hf_model: PreTrainedModel,
    base_model: str,
    output_path: Path,
) -> None:
    """Save a DeepSeek checkpoint with BF16 dense weights and FP8 routed experts."""
    state_dict = _build_mixed_precision_state_dict(hf_model)
    metadata_dir = _resolve_metadata_dir(base_model)

    hf_model.config.save_pretrained(output_path)
    generation_config = getattr(hf_model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output_path)

    _save_state_dict(state_dict=state_dict, output_path=output_path, metadata_dir=metadata_dir)
    _patch_config(output_path)
    _copy_custom_files(
        output_path=output_path,
        base_model=base_model,
        metadata_dir=metadata_dir,
    )


def _build_mixed_precision_state_dict(hf_model: PreTrainedModel) -> dict[str, torch.Tensor]:
    state_dict = hf_model.state_dict()
    mixed_state_dict: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        cpu_tensor = tensor.detach().cpu().contiguous()

        if _is_routed_expert_weight(key):
            quantized, scale = _quantize_weight_blockwise(cpu_tensor)
            mixed_state_dict[key] = quantized
            mixed_state_dict[_weight_scale_key(key)] = scale
            continue

        if _should_keep_float32(key):
            mixed_state_dict[key] = cpu_tensor.to(torch.float32)
            continue

        if cpu_tensor.is_floating_point():
            mixed_state_dict[key] = cpu_tensor.to(torch.bfloat16)
            continue

        mixed_state_dict[key] = cpu_tensor

    return mixed_state_dict


def _is_routed_expert_weight(key: str) -> bool:
    return (
        ".mlp.experts." in key
        and ".shared_experts." not in key
        and key.endswith(_DEEPSEEK_ROUTED_EXPERT_PROJ_NAMES)
    )


def _should_keep_float32(key: str) -> bool:
    return key.endswith("e_score_correction_bias")


def _weight_scale_key(weight_key: str) -> str:
    return weight_key.removesuffix(".weight") + ".weight_scale"


def _quantize_weight_blockwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D expert weight tensor, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    block_rows, block_cols = _DEEPSEEK_BLOCK_SIZE
    scale_rows = math.ceil(rows / block_rows)
    scale_cols = math.ceil(cols / block_cols)
    scales = torch.empty((scale_rows, scale_cols), dtype=torch.float32)
    quantized = torch.empty_like(weight, dtype=_DEEPSEEK_FP8_DTYPE)
    max_fp8 = _get_fp8_max()

    for row_idx in range(scale_rows):
        row_start = row_idx * block_rows
        row_end = min(row_start + block_rows, rows)
        for col_idx in range(scale_cols):
            col_start = col_idx * block_cols
            col_end = min(col_start + block_cols, cols)
            block = weight[row_start:row_end, col_start:col_end].to(torch.float32)
            max_abs = block.abs().max()
            if max_abs == 0:
                scale = torch.tensor(1.0, dtype=torch.float32)
            else:
                scale = max_abs / max_fp8
            scales[row_idx, col_idx] = scale
            quantized[row_start:row_end, col_start:col_end] = (
                block / scale
            ).clamp(min=-max_fp8, max=max_fp8).to(_DEEPSEEK_FP8_DTYPE)

    return quantized.contiguous(), scales.contiguous()


def _get_fp8_max() -> float:
    try:
        return float(torch.finfo(_DEEPSEEK_FP8_DTYPE).max)
    except TypeError:
        return 448.0


def _save_state_dict(
    *,
    state_dict: dict[str, torch.Tensor],
    output_path: Path,
    metadata_dir: Path | None,
) -> None:
    if metadata_dir is None:
        save_file(_contiguous_state_dict(state_dict), str(output_path / "model.safetensors"))
        return

    index_path = metadata_dir / "model.safetensors.index.json"
    if not index_path.exists():
        save_file(_contiguous_state_dict(state_dict), str(output_path / "model.safetensors"))
        return

    weight_map = json.loads(index_path.read_text())["weight_map"]
    if not weight_map:
        save_file(_contiguous_state_dict(state_dict), str(output_path / "model.safetensors"))
        return

    default_shard = next(iter(weight_map.values()))
    shard_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    output_weight_map: dict[str, str] = {}

    for key in sorted(state_dict):
        shard_name = weight_map.get(key)
        if shard_name is None and key.endswith(".weight_scale"):
            shard_name = weight_map.get(key.removesuffix(".weight_scale") + ".weight")
        if shard_name is None:
            shard_name = default_shard
        shard_tensors[shard_name][key] = state_dict[key].contiguous()
        output_weight_map[key] = shard_name

    for shard_name, shard_state_dict in shard_tensors.items():
        save_file(shard_state_dict, str(output_path / shard_name))

    total_size = sum(_tensor_nbytes(tensor) for tensor in state_dict.values())
    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    (output_path / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2, sort_keys=True) + "\n"
    )


def _contiguous_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.contiguous() for key, tensor in state_dict.items()}


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _patch_config(output_path: Path) -> None:
    config_path = output_path / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("quantization_config", None)
    config["compression_config"] = {
        "format": "float_quantized",
        "quant_method": "compressed-tensors",
        "config_groups": {
            "routed_experts_fp8": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "block",
                    "dynamic": False,
                    "symmetric": True,
                    "block_structure": list(_DEEPSEEK_BLOCK_SIZE),
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "tensor",
                    "dynamic": True,
                    "symmetric": True,
                },
            }
        },
        "ignore": [
            "re:.*embed_tokens.*",
            "re:.*lm_head.*",
            "re:.*norm.*",
            "re:.*self_attn.*",
            "re:.*\\.mlp\\.gate($|\\.)",
            "re:.*\\.shared_experts\\..*",
        ],
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def _copy_custom_files(
    *,
    output_path: Path,
    base_model: str,
    metadata_dir: Path | None,
) -> None:
    source_dirs = []
    base_model_path = Path(base_model).expanduser()
    if base_model_path.is_dir():
        source_dirs.append(base_model_path)
    if metadata_dir is not None and metadata_dir not in source_dirs:
        source_dirs.append(metadata_dir)

    for file_name in _DEEPSEEK_CUSTOM_FILES:
        destination = output_path / file_name
        if destination.exists():
            continue

        for source_dir in source_dirs:
            source = source_dir / file_name
            if source.exists():
                shutil.copy2(source, destination)
                break
        else:
            raise FileNotFoundError(f"Could not locate required DeepSeek file: {file_name}")


def _resolve_metadata_dir(base_model: str) -> Path | None:
    base_model_path = Path(base_model).expanduser()
    if base_model_path.is_dir():
        local_dir = base_model_path.resolve()
        if _has_required_metadata(local_dir):
            return local_dir

    for repo_id in _candidate_repo_ids(base_model):
        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=_METADATA_ALLOW_PATTERNS,
            )
        except Exception:
            logger.debug("Unable to download DeepSeek metadata for %s", repo_id, exc_info=True)
            continue

        metadata_dir = Path(snapshot_path)
        if _has_required_metadata(metadata_dir):
            return metadata_dir

    return base_model_path.resolve() if base_model_path.is_dir() else None


def _candidate_repo_ids(base_model: str) -> list[str]:
    repo_ids: list[str] = []
    if not Path(base_model).expanduser().exists():
        repo_ids.append(base_model)
    if _DEFAULT_DEEPSEEK_REPO not in repo_ids:
        repo_ids.append(_DEFAULT_DEEPSEEK_REPO)
    return repo_ids


def _has_required_metadata(path: Path) -> bool:
    if not path.exists():
        return False

    has_index = (path / "model.safetensors.index.json").exists()
    has_all_custom_files = all((path / file_name).exists() for file_name in _DEEPSEEK_CUSTOM_FILES)
    return has_index or has_all_custom_files
