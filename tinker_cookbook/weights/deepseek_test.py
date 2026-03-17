"""Offline unit tests for DeepSeek export helpers."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import tinker_cookbook.weights._deepseek as deepseek_export


def _write_resume_state(
    output_path: Path,
    *,
    model_path: str,
    adapter_path: Path,
    completed_shards: list[str],
) -> None:
    state = {
        "version": 1,
        "model_path": model_path,
        "adapter_path": str(adapter_path.expanduser().resolve()),
        "status": "in_progress",
        "completed_shards": completed_shards,
        "total_size": 0,
        "weight_map": {},
    }
    (output_path / "merge_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    )


def test_is_deepseek_config_matches_known_deepseek_configs_only():
    deepseek_class = type("DeepseekV3Config", (), {})
    deepseekish_class = type("DeepseekishConfig", (), {})

    assert deepseek_export.is_deepseek_config({"model_type": "deepseek_v3"})
    assert deepseek_export.is_deepseek_config(deepseek_class())
    assert not deepseek_export.is_deepseek_config({"model_type": "deepseek_custom"})
    assert not deepseek_export.is_deepseek_config(deepseekish_class())


def test_is_routed_expert_weight_classifies_only_routed_expert_projections():
    assert deepseek_export._is_routed_expert_weight("model.layers.0.mlp.experts.1.gate_proj.weight")
    assert deepseek_export._is_routed_expert_weight("model.layers.0.mlp.experts.1.up_proj.weight")
    assert deepseek_export._is_routed_expert_weight("model.layers.0.mlp.experts.1.down_proj.weight")
    assert not deepseek_export._is_routed_expert_weight(
        "model.layers.0.mlp.shared_experts.gate_proj.weight"
    )
    assert not deepseek_export._is_routed_expert_weight("model.layers.0.self_attn.q_proj.weight")
    assert not deepseek_export._is_routed_expert_weight(
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale"
    )


def test_quantize_and_dequantize_fp8_blockwise_round_trip(monkeypatch):
    monkeypatch.setattr(deepseek_export, "_DEEPSEEK_BLOCK_SIZE", (1, 1))
    weight = torch.tensor(
        [
            [1.5, -2.0, 0.25],
            [0.0, 3.0, -4.0],
        ],
        dtype=torch.bfloat16,
    )

    quantized, scales = deepseek_export._quantize_weight_blockwise(weight)
    dequantized = deepseek_export._dequantize_fp8_blockwise(
        quantized,
        scales,
        deepseek_export._DEEPSEEK_BLOCK_SIZE,
        output_dtype=torch.bfloat16,
    )

    assert quantized.dtype == deepseek_export._DEEPSEEK_FP8_DTYPE
    assert scales.dtype == torch.float32
    torch.testing.assert_close(dequantized, weight, atol=0, rtol=0)


def test_build_merge_ops_remaps_names_and_broadcasts_shared_expert_lora():
    adapter_weights = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 1),
        "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.ones(1, 1, 4),
        "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(2, 8, 1),
    }
    current_keys = {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.1.gate_proj.weight",
    }

    merge_ops = deepseek_export._build_merge_ops(
        adapter_weights=adapter_weights,
        adapter_config={"lora_alpha": 1, "r": 1},
        current_keys=current_keys,
    )

    dense_key = "model.language_model.layers.0.self_attn.q_proj.weight"
    expert0_key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    expert1_key = "model.language_model.layers.0.mlp.experts.1.gate_proj.weight"
    assert set(merge_ops) == {dense_key, expert0_key, expert1_key}
    assert len(merge_ops[dense_key]) == 1
    assert len(merge_ops[expert0_key]) == 1
    assert len(merge_ops[expert1_key]) == 1
    assert merge_ops[dense_key][0].target_key == dense_key
    assert merge_ops[expert0_key][0].lora_A.ndim == 2
    assert merge_ops[expert0_key][0].lora_B.ndim == 2
    torch.testing.assert_close(merge_ops[expert0_key][0].lora_A, torch.ones(1, 4))
    torch.testing.assert_close(merge_ops[expert1_key][0].lora_A, torch.ones(1, 4))


def test_load_resume_state_rebuilds_index_from_tracked_shards(tmp_path):
    output_path = tmp_path / "merged"
    output_path.mkdir()
    adapter_path = tmp_path / "adapter"
    model_path = "deepseek-ai/DeepSeek-V3.1"
    shard_name = "model-00001-of-00002.safetensors"
    shard_tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    save_file(
        {"model.layers.0.self_attn.q_proj.weight": shard_tensor},
        str(output_path / shard_name),
    )
    _write_resume_state(
        output_path,
        model_path=model_path,
        adapter_path=adapter_path,
        completed_shards=[shard_name],
    )

    state, completed = deepseek_export._load_resume_state(
        output_path=output_path,
        model_path=model_path,
        adapter_path=str(adapter_path),
    )

    assert completed == {shard_name}
    assert state["completed_shards"] == [shard_name]
    assert state["weight_map"] == {"model.layers.0.self_attn.q_proj.weight": shard_name}
    assert state["total_size"] == deepseek_export._tensor_nbytes(shard_tensor)


def test_load_resume_state_rejects_untracked_output_shards(tmp_path):
    output_path = tmp_path / "merged"
    output_path.mkdir()
    adapter_path = tmp_path / "adapter"
    model_path = "deepseek-ai/DeepSeek-V3.1"
    save_file({"unexpected.weight": torch.ones(2, 2)}, str(output_path / "leftover.safetensors"))
    _write_resume_state(
        output_path,
        model_path=model_path,
        adapter_path=adapter_path,
        completed_shards=[],
    )

    with pytest.raises(ValueError, match=r"not tracked in merge_state\.json"):
        deepseek_export._load_resume_state(
            output_path=output_path,
            model_path=model_path,
            adapter_path=str(adapter_path),
        )


def test_build_output_shard_matching_reference_quantizes_experts_and_preserves_dense_tensors():
    expert_weight_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    expert_scale_inv_key = expert_weight_key.removesuffix(".weight") + ".weight_scale_inv"
    expert_scale_key = expert_weight_key.removesuffix(".weight") + ".weight_scale"
    dense_key = "model.layers.0.self_attn.q_proj.weight"
    bias_key = "model.layers.0.self_attn.e_score_correction_bias"

    reference_shard = {
        dense_key: torch.zeros(2, 2, dtype=torch.bfloat16),
        expert_weight_key: torch.zeros(2, 2, dtype=torch.bfloat16),
        expert_scale_inv_key: torch.ones(1, 1, dtype=torch.float32),
        bias_key: torch.zeros(2, dtype=torch.float32),
    }
    merged_shard = {
        expert_weight_key: torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.bfloat16),
        bias_key: torch.tensor([1.0, 2.0], dtype=torch.float32),
    }

    output_shard = deepseek_export._build_output_shard_matching_reference(
        reference_shard=reference_shard,
        merged_shard=merged_shard,
        quantized_weight_keys={expert_weight_key},
        quantized_weight_keys_with_reference_scale={expert_weight_key},
        load_merged_tensor=lambda name: {dense_key: torch.ones(2, 2, dtype=torch.bfloat16)}[name],
    )

    assert output_shard[dense_key].dtype == torch.bfloat16
    assert torch.equal(output_shard[dense_key], torch.ones(2, 2, dtype=torch.bfloat16))
    assert output_shard[expert_weight_key].dtype == deepseek_export._DEEPSEEK_FP8_DTYPE
    assert output_shard[expert_scale_key].dtype == torch.float32
    assert expert_scale_inv_key not in output_shard
    assert output_shard[bias_key].dtype == torch.float32
