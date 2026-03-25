"""vLLM serving tests for GPT-OSS: .attn -> .self_attn remap.

GPT-OSS-20B ships with mxfp4 quantization. As of vLLM 0.18, LoRA is not
supported with mxfp4 quantization. This test verifies the adapter conversion
produces correct keys but skips actual vLLM serving.

When vLLM adds mxfp4+LoRA support or a non-quantized checkpoint is available,
add a serving test similar to the other model families.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig

from tests.weights.vllm_serving.conftest import (
    LORA_RANK,
    convert_and_load,
    save_tinker_adapter,
)

MODEL = "openai/gpt-oss-20b"


class TestGptOss:
    """GPT-OSS: .attn -> .self_attn remap verified via conversion.

    vLLM serving skipped — mxfp4+LoRA not yet supported (vLLM 0.18).
    """

    def test_attn_remap_adapter_conversion(self, tmp_path):
        """Verify .attn -> .self_attn remap produces correct PEFT keys."""
        config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        hidden = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        q_dim = num_heads * head_dim

        # Use .attn naming (Tinker internal)
        prefix = "base_model.model.model.layers.0.attn"
        weights = {
            f"{prefix}.q_proj.lora_A.weight": torch.randn(LORA_RANK, hidden) * 0.01,
            f"{prefix}.q_proj.lora_B.weight": torch.randn(q_dim, LORA_RANK) * 0.01,
        }

        adapter_path = tmp_path / "tinker_adapter"
        peft_path = tmp_path / "peft_adapter"
        save_tinker_adapter(adapter_path, weights)
        peft_weights, peft_config = convert_and_load(MODEL, adapter_path, peft_path)

        assert not any(".attn." in k for k in peft_weights), ".attn should be remapped"
        assert any(".self_attn." in k for k in peft_weights), "Should have .self_attn keys"
        assert "q_proj" in peft_config["target_modules"]

        print(f"\n  PEFT keys: {sorted(peft_weights.keys())}")
        print("  (vLLM serving skipped — mxfp4+LoRA not yet supported)")
