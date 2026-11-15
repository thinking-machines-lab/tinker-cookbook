# SPDX-License-Identifier: Apache-2.0
"""Regression tests for quantized model export using local synthetic checkpoints."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinker_cookbook.weights import build_hf_model
from tinker_cookbook.weights._export._quantized import dequantize_blockwise


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_export_preserves_small_expert_weights(tmp_path: Path, dtype: torch.dtype) -> None:
    """Small weights and zero blocks must survive FP8 export from low-precision checkpoints."""
    model_dir = tmp_path / "model"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    adapter_dir.mkdir()

    expert_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    dense_key = "model.layers.0.self_attn.q_proj.weight"
    weights = torch.zeros(128, 256, dtype=dtype)
    weights[:, :128] = torch.tensor([0.0, 0.001, -0.002, 0.003], dtype=dtype).repeat(32)
    save_file(
        {expert_key: weights, dense_key: torch.zeros(4, 4, dtype=torch.bfloat16)},
        str(model_dir / "model.safetensors"),
    )
    (model_dir / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}))
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    save_file(
        {
            f"{prefix}.lora_A.weight": torch.full((1, 4), 0.01),
            f"{prefix}.lora_B.weight": torch.ones(4, 1),
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
    recovered = dequantize_blockwise(
        exported[expert_key],
        exported[expert_key.removesuffix(".weight") + ".weight_scale"],
        dtype=torch.float32,
    )
    assert torch.isfinite(recovered).all(), "FP8 export introduced non-finite expert weights"
    torch.testing.assert_close(recovered, weights.float(), rtol=0.04, atol=1e-6)
    torch.testing.assert_close(
        exported[dense_key].float(), torch.full((4, 4), 0.01), atol=1e-4, rtol=0
    )
