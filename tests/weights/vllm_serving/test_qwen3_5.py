"""vLLM serving tests for Qwen3.5 family: full attention + split QKV.

Qwen3.5 has hybrid attention: some layers use standard full_attention (with
q_proj/k_proj/v_proj), others use linear_attention / GDN (with fused
in_proj_qkv). The adapter produces separate in_proj_q/k/v PEFT keys.

Note: Uses load_hf_config_dict to parse config.json directly because vLLM
pins transformers 4.x which doesn't recognize qwen3_5 model_type. vLLM
loads the model using its own built-in Qwen3.5 config class.
"""

from __future__ import annotations

import pytest
import torch
from vllm import LLM
from vllm.lora.request import LoRARequest

from tests.weights.vllm_serving.conftest import (
    LORA_RANK,
    PROMPT,
    convert_and_load,
    generate,
    load_hf_config_dict,
    save_tinker_adapter,
)

MODEL = "Qwen/Qwen3.5-4B"


def _get_text_config() -> dict:
    config_dict = load_hf_config_dict(MODEL)
    return config_dict["text_config"]


class TestQwen35FullAttention:
    """Baseline: adapter targeting full_attention layers (standard q_proj/v_proj)."""

    @pytest.fixture(scope="class")
    def llm(self):
        return LLM(
            model=MODEL,
            enable_lora=True,
            max_lora_rank=LORA_RANK,
            max_loras=1,
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.4,
        )

    def test_full_attention_adapter(self, llm, tmp_path):
        tc = _get_text_config()
        hidden = tc["hidden_size"]
        num_heads = tc["num_attention_heads"]
        num_kv_heads = tc["num_key_value_heads"]
        head_dim = tc.get("head_dim", hidden // num_heads)

        q_dim = num_heads * head_dim
        v_dim = num_kv_heads * head_dim

        layer_types = tc.get("layer_types", [])
        full_attn_idx = next(
            (i for i, lt in enumerate(layer_types) if lt == "full_attention"), None
        )
        if full_attn_idx is None:
            pytest.skip("No full_attention layer found in Qwen3.5-4B config")

        prefix = f"base_model.model.model.layers.{full_attn_idx}.self_attn"
        weights = {
            f"{prefix}.q_proj.lora_A.weight": torch.randn(LORA_RANK, hidden) * 0.01,
            f"{prefix}.q_proj.lora_B.weight": torch.randn(q_dim, LORA_RANK) * 0.01,
            f"{prefix}.v_proj.lora_A.weight": torch.randn(LORA_RANK, hidden) * 0.01,
            f"{prefix}.v_proj.lora_B.weight": torch.randn(v_dim, LORA_RANK) * 0.01,
        }

        adapter_path = tmp_path / "tinker_adapter"
        peft_path = tmp_path / "peft_adapter"
        save_tinker_adapter(adapter_path, weights)
        peft_weights, peft_config = convert_and_load(MODEL, adapter_path, peft_path)

        base_text = generate(llm, PROMPT)
        assert len(base_text) > 0

        lora_req = LoRARequest("qwen35_full_attn", 1, str(peft_path))
        lora_text = generate(llm, PROMPT, lora_request=lora_req)
        assert len(lora_text) > 0

        print(f"\n  target_modules: {peft_config['target_modules']}")
        print(f"  Base: {base_text!r}")
        print(f"  LoRA: {lora_text!r}")


class TestQwen35SplitQkv:
    """Critical test: split in_proj_q/k/v on linear_attention (GDN) layers.

    vLLM's packed_modules_mapping for Qwen3.5:
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"]

    Despite the fused mapping, vLLM correctly handles separate in_proj_q/k/v
    LoRA modules (verified 2026-03-24 with vLLM 0.18).
    """

    @pytest.fixture(scope="class")
    def llm(self):
        return LLM(
            model=MODEL,
            enable_lora=True,
            max_lora_rank=LORA_RANK,
            max_loras=1,
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.4,
        )

    def test_split_qkv_adapter(self, llm, tmp_path):
        tc = _get_text_config()
        hidden = tc["hidden_size"]

        layer_types = tc.get("layer_types", [])
        linear_attn_idx = next(
            (i for i, lt in enumerate(layer_types) if lt == "linear_attention"), None
        )
        if linear_attn_idx is None:
            pytest.skip("No linear_attention layer found in Qwen3.5-4B config")

        q_dim = tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
        k_dim = q_dim
        v_dim = tc["linear_num_value_heads"] * tc["linear_value_head_dim"]

        prefix = f"base_model.model.model.layers.{linear_attn_idx}.linear_attn"
        weights = {
            f"{prefix}.in_proj_q.lora_A.weight": torch.randn(LORA_RANK, hidden) * 0.01,
            f"{prefix}.in_proj_q.lora_B.weight": torch.randn(q_dim, LORA_RANK) * 0.01,
            f"{prefix}.in_proj_k.lora_A.weight": torch.randn(LORA_RANK, hidden) * 0.01,
            f"{prefix}.in_proj_k.lora_B.weight": torch.randn(k_dim, LORA_RANK) * 0.01,
            f"{prefix}.in_proj_v.lora_A.weight": torch.randn(LORA_RANK, hidden) * 0.01,
            f"{prefix}.in_proj_v.lora_B.weight": torch.randn(v_dim, LORA_RANK) * 0.01,
        }

        adapter_path = tmp_path / "tinker_adapter"
        peft_path = tmp_path / "peft_adapter"
        save_tinker_adapter(adapter_path, weights)
        peft_weights, peft_config = convert_and_load(MODEL, adapter_path, peft_path)

        assert any("in_proj_q" in k for k in peft_weights)
        assert any("in_proj_k" in k for k in peft_weights)
        assert any("in_proj_v" in k for k in peft_weights)

        base_text = generate(llm, PROMPT)
        assert len(base_text) > 0

        lora_req = LoRARequest("qwen35_split_qkv", 1, str(peft_path))
        lora_text = generate(llm, PROMPT, lora_request=lora_req)
        assert len(lora_text) > 0

        print(f"\n  target_modules: {peft_config['target_modules']}")
        print(f"  Base: {base_text!r}")
        print(f"  LoRA: {lora_text!r}")
