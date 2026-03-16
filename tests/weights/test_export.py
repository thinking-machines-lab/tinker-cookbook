"""End-to-end tests for build_hf_model across all supported model families.

Each test instantiates a tiny real HuggingFace model from config (no weight
download), saves it to disk with synthetic LoRA adapter weights, runs the
full build_hf_model pipeline, reloads, and verifies correctness.

Model families tested:
- GPT-OSS: fused interleaved gate_up_proj
- Qwen3-VL MoE: fused concatenated gate_up_proj + vision model prefix
- Qwen3 MoE: separate per-expert weights
- Qwen3 dense: standard linear layers (no experts)
"""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PretrainedConfig,
)

from tinker_cookbook.weights import build_hf_model


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FILL_A = 0.01  # LoRA fill for gate / first projection
FILL_B = 0.05  # LoRA fill for up / second projection


def _save_model_to_disk(
    config: PretrainedConfig,
    path: Path,
    *,
    tokenizer_name: str,
    is_vision: bool = False,
) -> None:
    auto_cls = AutoModelForImageTextToText if is_vision else AutoModelForCausalLM
    model = auto_cls.from_config(config, trust_remote_code=True, dtype=torch.float32)
    model.save_pretrained(path)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tok.save_pretrained(path)


def _save_expert_adapter(
    path: Path,
    *,
    num_experts: int,
    in_dim: int,
    out_dim: int,
    gate_fill: float = FILL_A,
    up_fill: float = FILL_B,
    layer_prefix: str = "base_model.model.model.layers.0.mlp.experts",
) -> None:
    """Save a LoRA adapter for expert gate (w1) and up (w3) projections."""
    weights: dict[str, torch.Tensor] = {}
    rank = 1
    weights[f"{layer_prefix}.w1.lora_A.weight"] = torch.ones(num_experts, rank, in_dim) * gate_fill
    weights[f"{layer_prefix}.w1.lora_B.weight"] = torch.ones(num_experts, out_dim, rank)
    weights[f"{layer_prefix}.w3.lora_A.weight"] = torch.ones(num_experts, rank, in_dim) * up_fill
    weights[f"{layer_prefix}.w3.lora_B.weight"] = torch.ones(num_experts, out_dim, rank)

    path.mkdir(parents=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


def _save_dense_adapter(
    path: Path,
    *,
    in_dim: int,
    out_dim: int,
    fill: float = FILL_A,
    layer_prefix: str = "base_model.model.model.layers.0.mlp",
) -> None:
    """Save a LoRA adapter for a dense (non-expert) linear layer."""
    rank = 1
    weights = {
        f"{layer_prefix}.gate_proj.lora_A.weight": torch.ones(rank, in_dim) * fill,
        f"{layer_prefix}.gate_proj.lora_B.weight": torch.ones(out_dim, rank),
    }

    path.mkdir(parents=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


def _run_build_and_reload(
    model_path: Path,
    adapter_path: Path,
    output_path: Path,
    *,
    is_vision: bool = False,
) -> dict[str, torch.Tensor]:
    """Run build_hf_model and return the reloaded state dict."""
    build_hf_model(
        base_model=str(model_path),
        adapter_path=str(adapter_path),
        output_path=str(output_path),
    )
    auto_cls = AutoModelForImageTextToText if is_vision else AutoModelForCausalLM
    reloaded = auto_cls.from_pretrained(output_path, trust_remote_code=True, dtype=torch.float32)
    return reloaded.state_dict()


# ---------------------------------------------------------------------------
# 1. GPT-OSS — fused interleaved gate_up_proj
# ---------------------------------------------------------------------------


def _make_tiny_gpt_oss_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.num_local_experts = 2
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.layer_types = ["full_attention"]
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    return config


class TestGptOssFusedInterleaved:
    """GPT-OSS: gate_up_proj with interleaved layout [g0, u0, g1, u1, ...]."""

    FUSED_KEY = "model.layers.0.mlp.experts.gate_up_proj"

    def test_gate_and_up_deltas_in_correct_interleaved_slots(self):
        config = _make_tiny_gpt_oss_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            _save_model_to_disk(config, model_path, tokenizer_name="openai/gpt-oss-20b")
            orig = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = orig.state_dict()[self.FUSED_KEY].clone()
            num_experts, in_dim, fused_dim = orig_fused.shape

            _save_expert_adapter(
                adapter_path, num_experts=num_experts, in_dim=in_dim, out_dim=fused_dim // 2
            )
            merged_sd = _run_build_and_reload(model_path, adapter_path, output_path)

            delta = merged_sd[self.FUSED_KEY] - orig_fused
            gate_delta = delta[:, :, 0::2]
            up_delta = delta[:, :, 1::2]

            assert torch.allclose(gate_delta, torch.full_like(gate_delta, FILL_A), atol=1e-3)
            assert torch.allclose(up_delta, torch.full_like(up_delta, FILL_B), atol=1e-3)

    def test_up_only_does_not_modify_gate_slots(self):
        config = _make_tiny_gpt_oss_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            _save_model_to_disk(config, model_path, tokenizer_name="openai/gpt-oss-20b")
            orig = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_gate = orig.state_dict()[self.FUSED_KEY][:, :, 0::2].clone()
            num_experts, in_dim, fused_dim = orig.state_dict()[self.FUSED_KEY].shape

            # Save only w3 (up) adapter
            prefix = "base_model.model.model.layers.0.mlp.experts"
            rank = 1
            up_only = {
                f"{prefix}.w3.lora_A.weight": torch.ones(num_experts, rank, in_dim) * FILL_B,
                f"{prefix}.w3.lora_B.weight": torch.ones(num_experts, fused_dim // 2, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(up_only, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            merged_sd = _run_build_and_reload(model_path, adapter_path, output_path)
            merged_gate = merged_sd[self.FUSED_KEY][:, :, 0::2]

            assert torch.allclose(merged_gate, orig_gate, atol=1e-3), (
                "up adapter modified gate slots"
            )


# ---------------------------------------------------------------------------
# 2. Qwen3-VL MoE — fused concatenated gate_up_proj + vision prefix
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_vl_moe_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True)
    tc = config.text_config
    tc.num_hidden_layers = 1
    tc.num_experts = 2
    tc.num_experts_per_tok = 1
    tc.hidden_size = 64
    tc.intermediate_size = 64
    tc.num_attention_heads = 2
    tc.num_key_value_heads = 2
    config.vision_config.num_hidden_layers = 1
    config.vision_config.hidden_size = 64
    config.vision_config.intermediate_size = 64
    config.vision_config.num_attention_heads = 2
    return config


class TestQwen3VlMoeFusedConcatenated:
    """Qwen3-VL MoE: gate_up_proj with concatenated layout [gate | up].

    Also tests the vision model language_model prefix remapping.
    """

    FUSED_KEY = "model.language_model.layers.0.mlp.experts.gate_up_proj"

    def test_gate_and_up_deltas_in_correct_halves(self):
        config = _make_tiny_qwen3_vl_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            _save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                is_vision=True,
            )
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = orig.state_dict()[self.FUSED_KEY].clone()
            num_experts, in_dim, fused_dim = orig_fused.shape
            sz = fused_dim // 2

            # Vision model: adapter uses model.layers... but HF has model.language_model.layers...
            _save_expert_adapter(
                adapter_path,
                num_experts=num_experts,
                in_dim=in_dim,
                out_dim=sz,
            )

            merged_sd = _run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

            delta = merged_sd[self.FUSED_KEY] - orig_fused
            gate_half = delta[:, :, :sz]
            up_half = delta[:, :, sz:]

            assert torch.allclose(gate_half, torch.full_like(gate_half, FILL_A), atol=1e-3)
            assert torch.allclose(up_half, torch.full_like(up_half, FILL_B), atol=1e-3)

    def test_up_only_does_not_modify_gate_half(self):
        config = _make_tiny_qwen3_vl_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            _save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                is_vision=True,
            )
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = orig.state_dict()[self.FUSED_KEY].clone()
            num_experts, in_dim, fused_dim = orig_fused.shape
            sz = fused_dim // 2

            # Only w3 (up) adapter
            prefix = "base_model.model.model.layers.0.mlp.experts"
            rank = 1
            weights = {
                f"{prefix}.w3.lora_A.weight": torch.ones(num_experts, rank, in_dim) * FILL_B,
                f"{prefix}.w3.lora_B.weight": torch.ones(num_experts, sz, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            merged_sd = _run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

            orig_gate = orig_fused[:, :, :sz]
            merged_gate = merged_sd[self.FUSED_KEY][:, :, :sz]

            assert torch.allclose(merged_gate, orig_gate, atol=1e-3), (
                "up adapter modified gate half"
            )


# ---------------------------------------------------------------------------
# 3. Qwen3 MoE — separate per-expert weights
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_moe_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.num_experts = 2
    config.num_experts_per_tok = 1
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    return config


class TestQwen3MoeSeparateExperts:
    """Qwen3 MoE: individual gate_proj/up_proj per expert."""

    def test_per_expert_weights_updated(self):
        config = _make_tiny_qwen3_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            _save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-30B-A3B")
            orig = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_sd = {k: v.clone() for k, v in orig.state_dict().items()}
            num_experts = 2

            # Read actual dims from model (gate_proj shape is [intermediate, hidden])
            gate_shape = orig_sd["model.layers.0.mlp.experts.0.gate_proj.weight"].shape
            expert_out_dim, expert_in_dim = gate_shape
            _save_expert_adapter(
                adapter_path, num_experts=num_experts, in_dim=expert_in_dim, out_dim=expert_out_dim
            )
            merged_sd = _run_build_and_reload(model_path, adapter_path, output_path)

            for i in range(num_experts):
                gate_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
                up_key = f"model.layers.0.mlp.experts.{i}.up_proj.weight"

                gate_delta = (merged_sd[gate_key] - orig_sd[gate_key]).abs().sum()
                up_delta = (merged_sd[up_key] - orig_sd[up_key]).abs().sum()

                assert gate_delta > 0, f"Expert {i} gate_proj not updated"
                assert up_delta > 0, f"Expert {i} up_proj not updated"


# ---------------------------------------------------------------------------
# 4. Qwen3 dense — standard linear layers (no experts)
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_dense_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    if hasattr(config, "layer_types") and config.layer_types is not None:
        config.layer_types = config.layer_types[:1]
    return config


class TestQwen3Dense:
    """Qwen3 dense: standard MLP with gate_proj/up_proj (no experts)."""

    def test_dense_linear_merge(self):
        config = _make_tiny_qwen3_dense_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            _save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-8B")
            orig = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_gate = orig.state_dict()["model.layers.0.mlp.gate_proj.weight"].clone()

            _save_dense_adapter(adapter_path, in_dim=64, out_dim=64, fill=FILL_A)
            merged_sd = _run_build_and_reload(model_path, adapter_path, output_path)

            delta = (merged_sd["model.layers.0.mlp.gate_proj.weight"] - orig_gate).abs().sum()
            assert delta > 0, "Dense gate_proj not updated"
