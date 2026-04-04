"""
Utilities for guessing good hyperparameters for fine-tuning.
"""

import json
import math
import struct

import huggingface_hub
import numpy as np

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.utils.misc_utils import not_none


def _list_param_shapes_from_safetensors_remote(
    repo_id: str,
    revision: str = "main",
    token: str | None = None,
) -> dict[str, tuple[int, ...]]:
    """
    Returns {param_name: shape_tuple} by reading ONLY the safetensors header(s)
    over HTTP (ranged requests). No full file download.
    """
    fs = huggingface_hub.HfFileSystem(token=token)
    info = huggingface_hub.model_info(repo_id, revision=revision, token=token)

    # find all .safetensors files (handles sharded checkpoints)
    st_files = [
        s.rfilename for s in not_none(info.siblings) if s.rfilename.endswith(".safetensors")
    ]
    if not st_files:
        raise FileNotFoundError("No .safetensors files found in this repo.")

    shapes: dict[str, tuple[int, ...]] = {}

    for fname in st_files:
        # Open remote file via fsspec; this performs HTTP range reads under the hood
        path = f"{repo_id}@{revision}/{fname}"  # HfFileSystem path format
        with fs.open(path, "rb") as f:
            # safetensors spec:
            # [0:8] = little-endian u64 header_len
            # [8:8+header_len] = UTF-8 JSON header
            header_len_bytes = f.read(8)
            assert isinstance(header_len_bytes, bytes)
            if len(header_len_bytes) < 8:
                raise OSError(f"File too small or not safetensors: {fname}")
            (header_len,) = struct.unpack("<Q", header_len_bytes)

            header_bytes = f.read(header_len)
            assert isinstance(header_bytes, bytes)
            if len(header_bytes) < header_len:
                raise OSError(f"Incomplete header read for {fname}")

            header = json.loads(header_bytes.decode("utf-8"))
            # header maps tensor_name -> { "dtype": "...", "shape": [...], "data_offsets": [start, end] }
            for name, meta in header.items():
                if name == "__metadata__":  # optional global metadata block
                    continue
                shapes[name] = tuple(meta["shape"])

    return shapes


def get_lora_lr_over_full_finetune_lr(model_name: str, lora_alpha: int = 32) -> float:
    """
    Return the factor that you should scale the full fine-tuning learning rate by to get the equivalent LoRA learning rate.
    Previously we had a more complicated formula, but the factor of 10 was more accurate empirically.
    See Lora Without Regret (https://thinkingmachines.ai/blog/lora/) for more details.

    Args:
        model_name: HuggingFace model identifier (currently unused but kept for API consistency).
        lora_alpha: LoRA alpha scaling parameter (currently unused; multiplier is fixed at 10).
    """
    return 10.0



def get_lora_param_count(
    model_name: str,
    lora_rank: int = 32,
    detailed: bool = False,
    include_experts: bool = True,
    shared_expert_outer_loras: bool = True,
) -> int | dict[str, int]:
    """Get the number of parameters in the LoRA adapter.

    Args:
        model_name: HuggingFace model identifier.
        lora_rank: Rank of the LoRA decomposition.
        detailed: If True, return a dict with expert/non-expert/total breakdowns.
        include_experts: Whether to include MoE expert layers in the count.
        shared_expert_outer_loras: If True, count shared outer dimensions only once
            across experts (reflects actual parameter sharing).

    Returns:
        Total parameter count as an int, or a detailed breakdown dict if ``detailed`` is True.
    """

    dim_sum = 0
    dim_sum_experts = 0
    ignore = ["gate", "embed_tokens", "q_b_proj", "kv_b_proj"]
    if not include_experts:
        ignore.append("experts")

    for name, shape in _list_param_shapes_from_safetensors_remote(model_name).items():
        if (
            len(shape) == 2
            and name.endswith(".weight")
            and not any(v in name.split(".") for v in ignore)
        ):
            parts = name.split(".")
            if "experts" not in parts or not shared_expert_outer_loras:
                dim_sum += shape[0] + shape[1]
            else:
                # For expert shared outer_loras, we only count the outer dims once, since they are shared across experts
                expert_idx = int(parts[parts.index("experts") + 1])
                weight_name = parts[parts.index("experts") + 2]
                assert weight_name in ["gate_proj", "down_proj", "up_proj"], (
                    f"Unexpected expert weight name: {weight_name}"
                )
                intermediate_dim = shape[1] if weight_name == "down_proj" else shape[0]
                outer_dim = shape[0] if weight_name == "down_proj" else shape[1]

                dim_sum_experts += intermediate_dim
                if expert_idx == 0:
                    dim_sum_experts += outer_dim

    non_expert_params = lora_rank * dim_sum
    expert_params = lora_rank * dim_sum_experts

    return (
        (expert_params + non_expert_params)
        if not detailed
        else {
            "expert_params": expert_params,
            "non_expert_params": non_expert_params,
            "total_params": expert_params + non_expert_params,
        }
    )


def get_lr(model_name: str, is_lora: bool = True) -> float:
    """Get a recommended learning rate for the given model.

    Returns empirically-determined optimal LoRA learning rates from a
    comprehensive SFT sweep across all Tinker models (tulu3 dataset,
    batch_size=128, 780 steps). For full fine-tuning, divides by 10x.

    See ``tinker_cookbook/recipes/chat_sl/results/sft_sweep.md`` for details.

    Args:
        model_name: HuggingFace model identifier.
        is_lora: If True, return the LoRA LR. If False, scale down by 10x
            for full fine-tuning.

    Returns:
        The recommended learning rate.
    """
    # Empirical best LoRA learning rates from SFT sweep.
    _LORA_LR: dict[str, float] = {
        # Qwen3
        "Qwen/Qwen3-235B-A22B-Instruct-2507": 3e-04,
        "Qwen/Qwen3-30B-A3B": 1e-03,
        "Qwen/Qwen3-30B-A3B-Base": 1e-03,
        "Qwen/Qwen3-30B-A3B-Instruct-2507": 1e-03,
        "Qwen/Qwen3-32B": 3e-04,
        "Qwen/Qwen3-4B-Instruct-2507": 1e-03,
        "Qwen/Qwen3-8B": 1e-03,
        "Qwen/Qwen3-8B-Base": 1e-03,
        # Qwen3-VL
        "Qwen/Qwen3-VL-235B-A22B-Instruct": 3e-04,
        "Qwen/Qwen3-VL-30B-A3B-Instruct": 1e-03,
        # Qwen3.5
        "Qwen/Qwen3.5-397B-A17B": 3e-04,
        "Qwen/Qwen3.5-35B-A3B": 3e-04,
        "Qwen/Qwen3.5-27B": 3e-04,
        "Qwen/Qwen3.5-4B": 3e-04,
        # DeepSeek
        "deepseek-ai/DeepSeek-V3.1": 4e-04,
        "deepseek-ai/DeepSeek-V3.1-Base": 4e-04,
        # Llama
        "meta-llama/Llama-3.1-70B": 3e-04,
        "meta-llama/Llama-3.1-8B": 3e-04,
        "meta-llama/Llama-3.1-8B-Instruct": 3e-04,
        "meta-llama/Llama-3.2-1B": 1e-03,
        "meta-llama/Llama-3.2-3B": 1e-03,
        "meta-llama/Llama-3.3-70B-Instruct": 3e-04,
        # Kimi
        "moonshotai/Kimi-K2-Thinking": 3e-04,
        "moonshotai/Kimi-K2.5": 3e-04,
        # NVIDIA Nemotron
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 4e-04,
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 1e-03,
        # OpenAI
        "openai/gpt-oss-120b": 3e-04,
        "openai/gpt-oss-20b": 1e-04,
    }

    lora_lr = _LORA_LR.get(model_name)
    if lora_lr is None:
        raise ConfigurationError(
            f"No calibrated learning rate for {model_name}. "
            f"Known models: {sorted(_LORA_LR.keys())}. "
            "Please specify a learning rate manually or run a sweep."
        )

    if is_lora:
        return lora_lr
    return lora_lr / get_lora_lr_over_full_finetune_lr(model_name)


def get_full_finetune_param_count(model_name: str) -> float:
    """Get the total parameter count for a model by reading safetensors headers.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Total number of parameters as a float.
    """
    count = 0
    for _name, shape in _list_param_shapes_from_safetensors_remote(model_name).items():
        count += np.prod(shape)
    return float(count)


def get_full_finetune_lr_multiplier(model_name: str) -> float:
    """Get a model-specific LR multiplier for full fine-tuning, proportional to 1/sqrt(param_count).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        The LR multiplier for full fine-tuning.
    """
    return 1.0 / math.sqrt(get_full_finetune_param_count(model_name))


def get_lora_lr_multiplier(model_name: str) -> float:
    """Get a model-specific multiplier for the LR, when training with LoRA.

    Given two models A and B, and learning rate LR_A that's known to be optimal for A,
    we can guess an optimal learning rate for B as
    LR_B = LR_A * get_lora_lr_multiplier(B) / get_lora_lr_multiplier(A)

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        The LoRA LR multiplier combining full-finetune scaling and LoRA factor.
    """
    return get_full_finetune_lr_multiplier(model_name) * get_lora_lr_over_full_finetune_lr(
        model_name
    )
