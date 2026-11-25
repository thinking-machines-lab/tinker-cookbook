# SPDX-License-Identifier: Apache-2.0
"""Model-specific FP8 export filtering through local checkpoint files."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinker_cookbook.weights import build_hf_model
from tinker_cookbook.weights._export._quantized import dequantize_blockwise


@pytest.mark.parametrize(
    ("model_type", "adapter_layer"),
    [("qwen3_moe", 0), ("qwen3_moe", 61), ("deepseek_v3", 0)],
)
def test_layer_61_is_only_skipped_for_deepseek(
    tmp_path: Path, model_type: str, adapter_layer: int
) -> None:
    model_dir = tmp_path / "model"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    adapter_dir.mkdir()

    dense_key = "model.layers.61.self_attn.q_proj.weight"
    expert_key = "model.layers.61.mlp.experts.0.gate_proj.weight"
    weights = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 4, dtype=torch.bfloat16),
        dense_key: torch.zeros(8, 4, dtype=torch.bfloat16),
        expert_key: torch.full((8, 4), 0.125, dtype=torch.bfloat16),
    }
    save_file(weights, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(
        json.dumps(
            {"model_type": model_type, "num_hidden_layers": 94 if model_type == "qwen3_moe" else 61}
        )
    )
    prefix = f"base_model.model.model.layers.{adapter_layer}.self_attn.q_proj"
    save_file(
        {
            f"{prefix}.lora_A.weight": torch.full((1, 4), 0.25),
            f"{prefix}.lora_B.weight": torch.ones(8, 1),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": 1}))

    build_hf_model(
        base_model=str(model_dir),
        adapter_path=str(adapter_dir),
        output_path=str(output_dir),
        quantize="experts-fp8",
        serving_format="vllm",
        device="cpu",
    )

    exported = load_file(str(output_dir / "model.safetensors"))
    if model_type == "deepseek_v3":
        assert dense_key not in exported
        assert expert_key not in exported
    else:
        assert dense_key in exported, "Export dropped a real layer's attention weights"
        recovered = dequantize_blockwise(
            exported[expert_key],
            exported[expert_key.removesuffix(".weight") + ".weight_scale"],
        )
        torch.testing.assert_close(recovered, weights[expert_key])
    target_key = f"model.layers.{adapter_layer}.self_attn.q_proj.weight"
    torch.testing.assert_close(exported[target_key], torch.full((8, 4), 0.25, dtype=torch.bfloat16))
