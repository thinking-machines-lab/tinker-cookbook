"""
Utilities for guessing good hyperparameters for fine-tuning.
"""

import json
import math
import struct
from typing import Dict, Tuple

import huggingface_hub
import numpy as np
from transformers import AutoConfig

from tinker_cookbook.utils.misc_utils import not_none


def _list_param_shapes_from_safetensors_remote(
    repo_id: str,
    revision: str = "main",
    token: str | None = None,
) -> Dict[str, Tuple[int, ...]]:
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

    shapes: Dict[str, Tuple[int, ...]] = {}

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
                raise IOError(f"File too small or not safetensors: {fname}")
            (header_len,) = struct.unpack("<Q", header_len_bytes)

            header_bytes = f.read(header_len)
            assert isinstance(header_bytes, bytes)
            if len(header_bytes) < header_len:
                raise IOError(f"Incomplete header read for {fname}")

            header = json.loads(header_bytes.decode("utf-8"))
            # header maps tensor_name -> { "dtype": "...", "shape": [...], "data_offsets": [start, end] }
            for name, meta in header.items():
                if name == "__metadata__":  # optional global metadata block
                    continue
                shapes[name] = tuple(meta["shape"])

    return shapes


def get_lora_lr_over_full_finetune_lr(model_name: str, lora_alpha: int = 32) -> float:
    """
    Find the factor that you should scale the full fine-tuning learning rate by to get the equivalent LoRA learning rate.
    """

    return _get_hidden_size(model_name) / (2 * lora_alpha)


def _get_hidden_size(model_name: str) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    return config.hidden_size


def get_lora_param_count(
    model_name: str,
    lora_rank: int = 32,
    scale_moe_rank_by_topk: bool = True,
    detailed: bool = False,
    include_experts: bool = True,
) -> int | dict[str, int]:
    """
    Get the number of parameters in the LoRA adapter.
    """
    model_config = AutoConfig.from_pretrained(model_name)

    dim_sum = 0
    dim_sum_experts = 0
    ignore = ["gate", "embed_tokens", "q_b_proj", "kv_b_proj"]
    if not include_experts:
        ignore.append("experts")

    for name, shape in _list_param_shapes_from_safetensors_remote(model_name).items():
        if (
            len(shape) == 2
            and name.endswith(".weight")
            and not any([v in name.split(".") for v in ignore])
        ):
            if "experts" not in name.split("."):
                dim_sum += shape[0] + shape[1]
            else:
                dim_sum_experts += shape[0] + shape[1]

    non_expert_params = lora_rank * dim_sum

    if scale_moe_rank_by_topk and dim_sum_experts > 0:
        assert hasattr(model_config, "num_experts_per_tok"), (
            "num_experts_per_tok is not in the model config, can't calculate with scale_moe_rank_by_topk"
        )
        moe_rank_scale = model_config.num_experts_per_tok
    else:
        moe_rank_scale = 1
    expert_params = dim_sum_experts * (lora_rank // moe_rank_scale)

    return (
        (expert_params + non_expert_params)
        if not detailed
        else {
            "expert_params": expert_params,
            "non_expert_params": non_expert_params,
            "total_params": expert_params + non_expert_params,
        }
    )


def get_full_finetune_param_count(model_name: str) -> float:
    count = 0
    for name, shape in _list_param_shapes_from_safetensors_remote(model_name).items():
        count += np.prod(shape)
    return float(count)


def get_full_finetune_lr_multiplier(model_name: str):
    return 1.0 / math.sqrt(get_full_finetune_param_count(model_name))


def get_lora_lr_multiplier(model_name: str):
    """
    Get a model-specific mutliplier for the LR, when training with LoRA.
    Given two models A and B, and learning rate LR_A that's known to be optimal for A,
    we can guess an optimal learning rate for B as
    LR_B = LR_A * get_lora_lr_multiplier(B) / get_lora_lr_multiplier(A)
    """
    return get_full_finetune_lr_multiplier(model_name) * get_lora_lr_over_full_finetune_lr(
        model_name
    )
