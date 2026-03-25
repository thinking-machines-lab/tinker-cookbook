"""vLLM serving tests for Nemotron-3.

Nemotron-3 uses 'backbone.*' weight prefix in HF checkpoints, but vLLM
remaps to 'model.*' internally via WeightsMapper. The adapter conversion
applies the same remap so PEFT keys match vLLM's parameter names.

Nemotron-3 is a hybrid Mamba+Attention MoE architecture.
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

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


class TestNemotron3:
    """Nemotron-3-Nano-30B-A3B: hybrid Mamba+Attention MoE with backbone.* prefix."""

    @pytest.fixture(scope="class")
    def llm(self):
        return LLM(
            model=MODEL,
            enable_lora=True,
            max_lora_rank=LORA_RANK,
            max_loras=1,
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=2,
            trust_remote_code=True,
        )

    def test_attention_adapter(self, llm, tmp_path):
        """Create q_proj adapter for an attention layer, verify backbone→model remap."""
        config = load_hf_config_dict(MODEL)
        hidden = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        head_dim = config.get("head_dim", hidden // num_heads)
        q_dim = num_heads * head_dim

        # Find an attention layer. Nemotron-3 uses hybrid_override_pattern:
        # M=Mamba, E=MoE, *=Attention
        pattern = config.get("hybrid_override_pattern", "")
        attn_idx = None
        for i, ch in enumerate(pattern):
            if ch == "*":
                attn_idx = i
                break
        if attn_idx is None:
            pytest.skip("No attention layer found in Nemotron config")

        # Tinker adapter uses backbone.* prefix (matching HF checkpoint)
        prefix = f"base_model.model.backbone.layers.{attn_idx}.mixer"
        weights = {
            f"{prefix}.q_proj.lora_A.weight": torch.randn(LORA_RANK, hidden) * 0.01,
            f"{prefix}.q_proj.lora_B.weight": torch.randn(q_dim, LORA_RANK) * 0.01,
        }

        adapter_path = tmp_path / "tinker_adapter"
        peft_path = tmp_path / "peft_adapter"
        save_tinker_adapter(adapter_path, weights)
        peft_weights, peft_config = convert_and_load(MODEL, adapter_path, peft_path)

        # PEFT keys should use model.* (vLLM's internal names), not backbone.*
        assert not any("backbone" in k for k in peft_weights), (
            "PEFT keys should not contain 'backbone'"
        )
        assert any("model.layers" in k for k in peft_weights), (
            "PEFT keys should use 'model.layers' prefix"
        )
        assert "q_proj" in peft_config["target_modules"]

        print(f"\n  PEFT keys: {sorted(peft_weights.keys())}")
        print(f"  target_modules: {peft_config['target_modules']}")

        base_text = generate(llm, PROMPT)
        assert len(base_text) > 0
        print(f"  Base: {base_text!r}")

        lora_req = LoRARequest("nemotron3", 1, str(peft_path))
        lora_text = generate(llm, PROMPT, lora_request=lora_req)
        assert len(lora_text) > 0
        print(f"  LoRA: {lora_text!r}")
