"""vLLM serving tests for Qwen3 family: dense + MoE.

Qwen3 dense: standard q_proj/v_proj attention LoRA.
Qwen3 MoE: 3D expert tensors expanded to per-expert 2D PEFT keys.

Note on MoE: vLLM's pack_moe requires ALL 3 expert projections (gate/down/up)
to be present in the adapter. Providing only a subset will fail with an
assertion error.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig
from vllm import LLM
from vllm.lora.request import LoRARequest

from tests.weights.vllm_serving.conftest import (
    LORA_RANK,
    PROMPT,
    convert_and_load,
    generate,
    save_tinker_adapter,
)


class TestQwen3Dense:
    """Qwen3-8B dense: standard q_proj/v_proj attention LoRA."""

    MODEL = "Qwen/Qwen3-8B"

    @pytest.fixture(scope="class")
    def llm(self):
        return LLM(
            model=self.MODEL,
            enable_lora=True,
            max_lora_rank=LORA_RANK,
            max_loras=1,
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.4,
        )

    def test_dense_attn_adapter(self, llm, tmp_path):
        config = AutoConfig.from_pretrained(self.MODEL, trust_remote_code=True)
        hidden = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads

        q_dim = num_heads * head_dim
        v_dim = num_kv_heads * head_dim

        prefix = "base_model.model.model.layers.0.self_attn"
        weights = {
            f"{prefix}.q_proj.lora_A.weight": torch.randn(LORA_RANK, hidden) * 0.01,
            f"{prefix}.q_proj.lora_B.weight": torch.randn(q_dim, LORA_RANK) * 0.01,
            f"{prefix}.v_proj.lora_A.weight": torch.randn(LORA_RANK, hidden) * 0.01,
            f"{prefix}.v_proj.lora_B.weight": torch.randn(v_dim, LORA_RANK) * 0.01,
        }

        adapter_path = tmp_path / "tinker_adapter"
        peft_path = tmp_path / "peft_adapter"
        save_tinker_adapter(adapter_path, weights)
        peft_weights, peft_config = convert_and_load(self.MODEL, adapter_path, peft_path)

        assert "q_proj" in peft_config["target_modules"]
        assert "v_proj" in peft_config["target_modules"]

        base_text = generate(llm, PROMPT)
        assert len(base_text) > 0, "Base model should generate text"

        lora_req = LoRARequest("qwen3_dense", 1, str(peft_path))
        lora_text = generate(llm, PROMPT, lora_request=lora_req)
        assert len(lora_text) > 0, "LoRA adapter should generate text"

        print(f"\n  Base: {base_text!r}")
        print(f"  LoRA: {lora_text!r}")


class TestQwen3Moe:
    """Qwen3-30B-A3B MoE: attention + expert LoRA (all 3 projections)."""

    MODEL = "Qwen/Qwen3-30B-A3B"

    @pytest.fixture(scope="class")
    def llm(self):
        return LLM(
            model=self.MODEL,
            enable_lora=True,
            max_lora_rank=LORA_RANK,
            max_loras=1,
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=2,
        )

    def test_expert_adapter(self, llm, tmp_path):
        config = AutoConfig.from_pretrained(self.MODEL, trust_remote_code=True)
        hidden = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        intermediate = config.intermediate_size
        num_experts = config.num_experts

        q_dim = num_heads * head_dim

        weights: dict[str, torch.Tensor] = {}

        # Attention adapter
        attn_prefix = "base_model.model.model.layers.0.self_attn"
        weights[f"{attn_prefix}.q_proj.lora_A.weight"] = torch.randn(LORA_RANK, hidden) * 0.01
        weights[f"{attn_prefix}.q_proj.lora_B.weight"] = torch.randn(q_dim, LORA_RANK) * 0.01

        # Expert adapter: all 3 projections required by vLLM's pack_moe
        expert_prefix = "base_model.model.model.layers.0.mlp.experts"
        for wname, out_dim, in_dim in [
            ("w1", intermediate, hidden),  # gate_proj
            ("w2", hidden, intermediate),  # down_proj
            ("w3", intermediate, hidden),  # up_proj
        ]:
            weights[f"{expert_prefix}.{wname}.lora_A.weight"] = (
                torch.randn(num_experts, LORA_RANK, in_dim) * 0.01
            )
            weights[f"{expert_prefix}.{wname}.lora_B.weight"] = (
                torch.randn(num_experts, out_dim, LORA_RANK) * 0.01
            )

        adapter_path = tmp_path / "tinker_adapter"
        peft_path = tmp_path / "peft_adapter"
        save_tinker_adapter(adapter_path, weights)
        peft_weights, peft_config = convert_and_load(self.MODEL, adapter_path, peft_path)

        # Verify per-expert expansion
        for i in range(num_experts):
            for proj in ("gate_proj", "down_proj", "up_proj"):
                key = f"base_model.model.model.layers.0.mlp.experts.{i}.{proj}.lora_A.weight"
                assert key in peft_weights, f"Missing per-expert key: {key}"
                assert peft_weights[key].ndim == 2

        base_text = generate(llm, PROMPT)
        assert len(base_text) > 0

        lora_req = LoRARequest("qwen3_moe", 1, str(peft_path))
        lora_text = generate(llm, PROMPT, lora_request=lora_req)
        assert len(lora_text) > 0

        print(f"\n  PEFT key count: {len(peft_weights)}")
        print(f"  Base: {base_text!r}")
        print(f"  LoRA: {lora_text!r}")
