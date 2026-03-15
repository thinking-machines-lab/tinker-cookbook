"""End-to-end and unit tests for build_hf_model.

E2e tests instantiate a real (but tiny) model from HuggingFace config,
save model + adapter to disk, call build_hf_model, reload, and verify
that LoRA deltas landed in the correct weight slots.
"""

import json
import os
import tempfile

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tinker_cookbook.weights import build_hf_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GATE_FILL = 0.01
UP_FILL = 0.05


def _make_tiny_gpt_oss_config() -> AutoConfig:
    """Create a minimal GPT-OSS config (~26M params, <1s to instantiate)."""
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.num_local_experts = 2
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.layer_types = ["full_attention"]
    del config.quantization_config
    return config


def _save_model_to_disk(config: AutoConfig, path: str) -> None:
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.float32)
    model.save_pretrained(path)
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    tok.save_pretrained(path)


def _save_adapter_to_disk(
    path: str,
    num_experts: int,
    in_dim: int,
    out_dim: int,
    rank: int,
    gate_fill: float,
    up_fill: float,
    include_gate: bool = True,
    include_up: bool = True,
) -> None:
    prefix = "base_model.model.model.layers.0.mlp.experts"
    weights: dict[str, torch.Tensor] = {}
    if include_gate:
        weights[f"{prefix}.w1.lora_A.weight"] = torch.ones(num_experts, rank, in_dim) * gate_fill
        weights[f"{prefix}.w1.lora_B.weight"] = torch.ones(num_experts, out_dim, rank)
    if include_up:
        weights[f"{prefix}.w3.lora_A.weight"] = torch.ones(num_experts, rank, in_dim) * up_fill
        weights[f"{prefix}.w3.lora_B.weight"] = torch.ones(num_experts, out_dim, rank)

    os.makedirs(path)
    save_file(weights, os.path.join(path, "adapter_model.safetensors"))
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump({"lora_alpha": 1, "r": rank}, f)


# ---------------------------------------------------------------------------
# E2E tests — GPT-OSS (interleaved fused gate_up_proj)
# ---------------------------------------------------------------------------


class TestBuildHfModelGptOss:
    """E2e tests using a real GPT-OSS model architecture.

    GPT-OSS uses interleaved gate_up_proj: [g0, u0, g1, u1, ...] along the
    last dimension. gate (w1) maps to even indices, up (w3) to odd indices.
    """

    def test_gate_and_up_deltas_land_in_correct_interleaved_slots(self):
        config = _make_tiny_gpt_oss_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")
            adapter_path = os.path.join(tmpdir, "adapter")
            output_path = os.path.join(tmpdir, "merged")

            _save_model_to_disk(config, model_path)

            original_model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = original_model.state_dict()[
                "model.layers.0.mlp.experts.gate_up_proj"
            ].clone()
            num_experts, in_dim, fused_dim = orig_fused.shape
            out_dim = fused_dim // 2

            _save_adapter_to_disk(
                adapter_path,
                num_experts,
                in_dim,
                out_dim,
                rank=1,
                gate_fill=GATE_FILL,
                up_fill=UP_FILL,
            )

            build_hf_model(
                base_model=model_path,
                adapter_path=adapter_path,
                output_path=output_path,
            )

            reloaded = AutoModelForCausalLM.from_pretrained(
                output_path, trust_remote_code=True, dtype=torch.float32
            )
            final_fused = reloaded.state_dict()["model.layers.0.mlp.experts.gate_up_proj"]
            delta = final_fused - orig_fused
            gate_delta = delta[:, :, 0::2]
            up_delta = delta[:, :, 1::2]

            assert torch.allclose(gate_delta, torch.full_like(gate_delta, GATE_FILL), atol=1e-3), (
                f"Gate delta wrong: mean={gate_delta.mean().item():.6f}, expected {GATE_FILL}"
            )
            assert torch.allclose(up_delta, torch.full_like(up_delta, UP_FILL), atol=1e-3), (
                f"Up delta wrong: mean={up_delta.mean().item():.6f}, expected {UP_FILL}"
            )

    def test_up_only_adapter_does_not_modify_gate_slots(self):
        config = _make_tiny_gpt_oss_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")
            adapter_path = os.path.join(tmpdir, "adapter")
            output_path = os.path.join(tmpdir, "merged")

            _save_model_to_disk(config, model_path)

            original_model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = original_model.state_dict()[
                "model.layers.0.mlp.experts.gate_up_proj"
            ].clone()
            num_experts, in_dim, fused_dim = orig_fused.shape
            out_dim = fused_dim // 2

            _save_adapter_to_disk(
                adapter_path,
                num_experts,
                in_dim,
                out_dim,
                rank=1,
                gate_fill=0.0,
                up_fill=UP_FILL,
                include_gate=False,
                include_up=True,
            )

            build_hf_model(
                base_model=model_path,
                adapter_path=adapter_path,
                output_path=output_path,
            )

            reloaded = AutoModelForCausalLM.from_pretrained(
                output_path, trust_remote_code=True, dtype=torch.float32
            )
            final_fused = reloaded.state_dict()["model.layers.0.mlp.experts.gate_up_proj"]
            orig_gate = orig_fused[:, :, 0::2]
            merged_gate = final_fused[:, :, 0::2]

            assert torch.allclose(merged_gate, orig_gate, atol=1e-3), (
                "up_proj adapter modified gate slots!"
            )

    def test_gate_only_adapter_does_not_modify_up_slots(self):
        config = _make_tiny_gpt_oss_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")
            adapter_path = os.path.join(tmpdir, "adapter")
            output_path = os.path.join(tmpdir, "merged")

            _save_model_to_disk(config, model_path)

            original_model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = original_model.state_dict()[
                "model.layers.0.mlp.experts.gate_up_proj"
            ].clone()
            num_experts, in_dim, fused_dim = orig_fused.shape
            out_dim = fused_dim // 2

            _save_adapter_to_disk(
                adapter_path,
                num_experts,
                in_dim,
                out_dim,
                rank=1,
                gate_fill=GATE_FILL,
                up_fill=0.0,
                include_gate=True,
                include_up=False,
            )

            build_hf_model(
                base_model=model_path,
                adapter_path=adapter_path,
                output_path=output_path,
            )

            reloaded = AutoModelForCausalLM.from_pretrained(
                output_path, trust_remote_code=True, dtype=torch.float32
            )
            final_fused = reloaded.state_dict()["model.layers.0.mlp.experts.gate_up_proj"]
            orig_up = orig_fused[:, :, 1::2]
            merged_up = final_fused[:, :, 1::2]

            assert torch.allclose(merged_up, orig_up, atol=1e-3), (
                "gate_proj adapter modified up slots!"
            )
