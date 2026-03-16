"""DeepSeek-specific mixed-precision export helpers.

This exporter has two separate, equally important problems to solve:

1. Quantization selection: only routed expert MLP weights are quantized to FP8,
   while dense, shared-expert, embedding, attention, and norm weights stay in
   higher precision.
2. Cross-shard dependencies: a weight tensor and the scale tensor needed to
   interpret it are not guaranteed to live in the same safetensors shard.

Export code therefore has to get both the quantization scope and the on-disk
tensor placement right. It cannot assume "load one shard and all related
tensors are nearby". Instead it must consult ``model.safetensors.index.json``
and preserve the reference shard layout when re-emitting tensors.

This module treats the reference index as the source of truth for on-disk tensor
placement and uses it as the shard template during save.
"""

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
    """Save a DeepSeek checkpoint with experts-only FP8 quantization.

    Routed expert MLP weights are written as FP8 plus explicit scale tensors.
    Dense and shared-expert weights remain in higher precision so the exported
    checkpoint matches the intended "experts-only" quantization scope.
    """
    logger.info("DeepSeek export: building mixed-precision state dict")
    state_dict = _build_mixed_precision_state_dict(hf_model)

    logger.info("DeepSeek export: resolving metadata source")
    metadata_dir = _resolve_metadata_dir(base_model)
    if metadata_dir is not None:
        logger.info("DeepSeek export: using metadata from %s", metadata_dir)
    else:
        logger.info("DeepSeek export: no metadata snapshot found, writing unsharded checkpoint")

    logger.info("DeepSeek export: saving config files")
    hf_model.config.save_pretrained(output_path)
    generation_config = getattr(hf_model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output_path)

    logger.info("DeepSeek export: writing checkpoint tensors")
    output_weight_map = _save_state_dict(
        state_dict=state_dict,
        output_path=output_path,
        metadata_dir=metadata_dir,
    )
    logger.info("DeepSeek export: patching config.json for compressed-tensors")
    _patch_config(output_path=output_path, output_weight_map=output_weight_map)
    logger.info("DeepSeek export: copying custom DeepSeek files")
    _copy_custom_files(
        output_path=output_path,
        base_model=base_model,
        metadata_dir=metadata_dir,
    )
    logger.info("DeepSeek export: finished")


def _build_mixed_precision_state_dict(hf_model: PreTrainedModel) -> dict[str, torch.Tensor]:
    """Build the final mixed-precision checkpoint state dict.

    Only ``model.layers.*.mlp.experts.*.{gate,up,down}_proj.weight`` tensors are
    requantized to FP8. Other floating-point tensors remain BF16, except tensors
    such as ``e_score_correction_bias`` that must stay FP32.
    """
    state_dict = hf_model.state_dict()
    mixed_state_dict: dict[str, torch.Tensor] = {}
    routed_expert_keys = [key for key in state_dict if _is_routed_expert_weight(key)]
    total_routed_expert_weights = len(routed_expert_keys)
    routed_progress_interval = _progress_interval(total_routed_expert_weights)
    routed_expert_count = 0
    bf16_count = 0
    float32_count = 0
    non_floating_count = 0

    logger.info(
        "DeepSeek export: preparing %d tensors with %d routed expert weights for FP8 quantization",
        len(state_dict),
        total_routed_expert_weights,
    )

    for key, tensor in state_dict.items():
        cpu_tensor = tensor.detach().cpu().contiguous()

        if _is_routed_expert_weight(key):
            quantized, scale = _quantize_weight_blockwise(cpu_tensor)
            mixed_state_dict[key] = quantized
            mixed_state_dict[_weight_scale_key(key)] = scale
            routed_expert_count += 1
            if (
                routed_expert_count == 1
                or routed_expert_count == total_routed_expert_weights
                or routed_expert_count % routed_progress_interval == 0
            ):
                logger.info(
                    "DeepSeek export: quantized routed expert weight %d/%d (%s)",
                    routed_expert_count,
                    total_routed_expert_weights,
                    key,
                )
            continue

        if _should_keep_float32(key):
            mixed_state_dict[key] = cpu_tensor.to(torch.float32)
            float32_count += 1
            continue

        if cpu_tensor.is_floating_point():
            mixed_state_dict[key] = cpu_tensor.to(torch.bfloat16)
            bf16_count += 1
            continue

        mixed_state_dict[key] = cpu_tensor
        non_floating_count += 1

    logger.info(
        "DeepSeek export: prepared state dict with %d BF16 tensors, %d FP8 expert weights, %d FP8 scales, %d FP32 tensors, and %d non-floating tensors",
        bf16_count,
        routed_expert_count,
        routed_expert_count,
        float32_count,
        non_floating_count,
    )
    return mixed_state_dict


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


def _progress_interval(total_items: int) -> int:
    if total_items <= 0:
        return 1
    return max(1, total_items // 20)


def _save_state_dict(
    *,
    state_dict: dict[str, torch.Tensor],
    output_path: Path,
    metadata_dir: Path | None,
) -> dict[str, str]:
    """Write tensors using the reference DeepSeek shard layout when available.

    DeepSeek checkpoints cannot be reconstructed safely by assuming
    shard-local tensor neighborhoods. A weight tensor and the scale tensor
    needed to interpret it may live in different shards, so we consult
    ``model.safetensors.index.json`` instead of inferring placement from
    whichever file happened to be loaded.

    For the routed-experts-only ``compressed-tensors`` export we do not have
    reference ``.weight_scale`` entries in the upstream index, so we place each
    emitted scale tensor in the shard of its corresponding weight. The important
    compatibility guarantee is that we still use the reference ``weight_map`` as
    the save template for the existing tensors and never repack shards
    arbitrarily.
    """
    if metadata_dir is None:
        logger.info("DeepSeek export: writing unsharded model.safetensors")
        shard_name = "model.safetensors"
        save_file(_contiguous_state_dict(state_dict), str(output_path / shard_name))
        return {key: shard_name for key in state_dict}

    index_path = metadata_dir / "model.safetensors.index.json"
    if not index_path.exists():
        logger.info("DeepSeek export: metadata index missing, writing unsharded model.safetensors")
        shard_name = "model.safetensors"
        save_file(_contiguous_state_dict(state_dict), str(output_path / shard_name))
        return {key: shard_name for key in state_dict}

    # The reference index is the authoritative source for shard placement.
    # DeepSeek weights and their related scale tensors are not guaranteed to be
    # colocated, so the exporter must reason from the global index rather than
    # shard-local state.
    weight_map = json.loads(index_path.read_text())["weight_map"]
    if not weight_map:
        logger.info("DeepSeek export: empty metadata index, writing unsharded model.safetensors")
        shard_name = "model.safetensors"
        save_file(_contiguous_state_dict(state_dict), str(output_path / shard_name))
        return {key: shard_name for key in state_dict}

    default_shard = next(iter(weight_map.values()))
    state_dict = _contiguous_state_dict(state_dict)
    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}
    output_weight_map: dict[str, str] = {}
    remaining_state_dict = dict(state_dict)
    shard_order: list[str] = []

    for shard_name, reference_keys in _build_reference_shard_layout(weight_map):
        shard_order.append(shard_name)
        output_shard: dict[str, torch.Tensor] = {}
        for reference_key in reference_keys:
            if reference_key in remaining_state_dict:
                output_shard[reference_key] = remaining_state_dict.pop(reference_key)
                output_weight_map[reference_key] = shard_name
                continue

            replacement_scale_key = _replacement_scale_key(reference_key)
            if replacement_scale_key is not None and replacement_scale_key in remaining_state_dict:
                output_shard[replacement_scale_key] = remaining_state_dict.pop(replacement_scale_key)
                output_weight_map[replacement_scale_key] = shard_name

        if output_shard:
            shard_tensors[shard_name] = output_shard

    if remaining_state_dict:
        logger.warning(
            "DeepSeek export: assigning %d tensor(s) not present in the reference layout using best-effort shard placement",
            len(remaining_state_dict),
        )
        fallback_shards: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for key, tensor in sorted(remaining_state_dict.items()):
            shard_name = weight_map.get(key)
            if shard_name is None and key.endswith(".weight_scale"):
                shard_name = weight_map.get(key.removesuffix(".weight_scale") + ".weight")
            if shard_name is None:
                shard_name = default_shard
            fallback_shards[shard_name][key] = tensor
            output_weight_map[key] = shard_name

        for shard_name, shard_dict in fallback_shards.items():
            if shard_name not in shard_order:
                shard_order.append(shard_name)
            shard_tensors.setdefault(shard_name, {}).update(shard_dict)

    shard_items = [(shard_name, shard_tensors[shard_name]) for shard_name in shard_order if shard_name in shard_tensors]
    shard_progress_interval = _progress_interval(len(shard_items))
    logger.info(
        "DeepSeek export: writing %d tensors across %d shards using %s",
        len(state_dict),
        len(shard_items),
        index_path,
    )
    for shard_idx, (shard_name, shard_state_dict) in enumerate(shard_items, start=1):
        save_file(shard_state_dict, str(output_path / shard_name))
        if (
            shard_idx == 1
            or shard_idx == len(shard_items)
            or shard_idx % shard_progress_interval == 0
        ):
            logger.info(
                "DeepSeek export: wrote shard %d/%d (%s, %d tensors)",
                shard_idx,
                len(shard_items),
                shard_name,
                len(shard_state_dict),
            )

    total_size = sum(_tensor_nbytes(tensor) for tensor in state_dict.values())
    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    (output_path / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2, sort_keys=True) + "\n"
    )
    logger.info("DeepSeek export: wrote model.safetensors.index.json")
    return output_weight_map


def _build_reference_shard_layout(weight_map: dict[str, str]) -> list[tuple[str, list[str]]]:
    shard_layout: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        shard_layout.setdefault(shard_name, []).append(tensor_name)
    return list(shard_layout.items())


def _replacement_scale_key(reference_key: str) -> str | None:
    if not reference_key.endswith(".weight_scale_inv"):
        return None
    return reference_key.removesuffix(".weight_scale_inv") + ".weight_scale"


def _contiguous_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.contiguous() for key, tensor in state_dict.items()}


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
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    except ImportError as exc:
        raise ImportError(
            "DeepSeek experts-only FP8 export requires the optional merge dependencies. "
            "Install them with: pip install 'tinker_cookbook[merge]'"
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
                    type="float",
                    strategy="block",
                    block_structure=list(_DEEPSEEK_BLOCK_SIZE),
                    symmetric=True,
                    dynamic=False,
                ),
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type="float",
                    strategy="tensor",
                    symmetric=True,
                    dynamic=True,
                ),
            )
        },
        format="float-quantized",
        quantization_status="compressed",
        ignore=ignore,
    )
    config_dict = config.model_dump()
    _strip_unsupported_quant_args_fields(config_dict)
    config_dict["quant_method"] = "compressed-tensors"
    return config_dict


def _strip_unsupported_quant_args_fields(config_dict: dict) -> None:
    """Drop fields newer vLLM builds may reject as extras."""
    extra_fields = {"scale_dtype", "zp_dtype"}
    for group in config_dict.get("config_groups", {}).values():
        if not isinstance(group, dict):
            continue
        for section in ("weights", "input_activations", "output_activations"):
            args = group.get(section)
            if not isinstance(args, dict):
                continue
            for key in extra_fields:
                args.pop(key, None)


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
            logger.info("DeepSeek export: custom file already present: %s", destination)
            continue

        for source_dir in source_dirs:
            source = source_dir / file_name
            if source.exists():
                shutil.copy2(source, destination)
                logger.info("DeepSeek export: copied %s from %s", file_name, source_dir)
                break
        else:
            raise FileNotFoundError(f"Could not locate required DeepSeek file: {file_name}")


def _resolve_metadata_dir(base_model: str) -> Path | None:
    """Resolve a directory that provides the reference DeepSeek shard/index metadata.

    We fetch metadata even for local merge inputs because the real DeepSeek
    shard topology often lives in the upstream snapshot. Preserving that
    topology avoids introducing layout drift when exporting mixed-precision
    checkpoints.
    """
    base_model_path = Path(base_model).expanduser()
    if base_model_path.is_dir():
        local_dir = base_model_path.resolve()
        if _has_required_metadata(local_dir):
            logger.info("DeepSeek export: using local metadata from %s", local_dir)
            return local_dir

    for repo_id in _candidate_repo_ids(base_model):
        try:
            logger.info("DeepSeek export: downloading metadata snapshot for %s", repo_id)
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=_METADATA_ALLOW_PATTERNS,
            )
        except Exception:
            logger.debug("Unable to download DeepSeek metadata for %s", repo_id, exc_info=True)
            continue

        metadata_dir = Path(snapshot_path)
        if _has_required_metadata(metadata_dir):
            logger.info("DeepSeek export: downloaded metadata snapshot to %s", metadata_dir)
            return metadata_dir

    logger.warning(
        "DeepSeek export: unable to locate metadata snapshot for %s; proceeding without shard template",
        base_model,
    )
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
