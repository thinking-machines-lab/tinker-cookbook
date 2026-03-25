"""E2e adapter tests for DeepSeek V3.1: verify unsupported error."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    PretrainedConfig,
)

from tests.weights.conftest import save_model_to_disk
from tinker_cookbook.exceptions import WeightsAdapterError
from tinker_cookbook.weights import build_lora_adapter

# ---------------------------------------------------------------------------
# DeepSeek V3.1 — verify unsupported error
# ---------------------------------------------------------------------------


def _deepseek_needs_custom_code() -> bool:
    import transformers

    return int(transformers.__version__.split(".")[0]) < 5


def _make_tiny_deepseek_v31_config() -> PretrainedConfig:
    needs_custom = _deepseek_needs_custom_code()
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3.1", trust_remote_code=needs_custom)
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 64
    config.moe_intermediate_size = 16
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.n_routed_experts = 2
    config.n_shared_experts = 1
    config.num_experts_per_tok = 1
    config.first_k_dense_replace = 0
    config.vocab_size = 256
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    return config


class TestDeepSeekV31Adapter:
    """DeepSeek V3.1: adapter conversion should raise WeightsAdapterError."""

    def test_raises_unsupported_error(self) -> None:
        config = _make_tiny_deepseek_v31_config()
        needs_custom = _deepseek_needs_custom_code()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="deepseek-ai/DeepSeek-V3.1",
                trust_remote_code=needs_custom,
            )

            # Create a minimal adapter
            rank = 1
            in_dim = 64
            out_dim = 64
            weights = {
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.ones(
                    rank, in_dim
                ),
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.ones(
                    out_dim, rank
                ),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            with pytest.raises(WeightsAdapterError, match="DeepSeek"):
                build_lora_adapter(
                    base_model=str(model_path),
                    adapter_path=str(adapter_path),
                    output_path=str(output_path),
                )
