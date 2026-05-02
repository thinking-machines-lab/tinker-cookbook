"""E2e export tests for Qwen3.5 family: dense + MoE with split QKV fusion."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    PretrainedConfig,
)

from tests.weights.conftest import (
    FILL_A,
    FILL_B,
    load_merged_tensors,
    run_build_and_reload,
    save_model_to_disk,
)
from tinker_cookbook.weights import build_hf_model

# ---------------------------------------------------------------------------
# Qwen3.5 dense — split in_proj_q/k/v → fused in_proj_qkv + tied embeddings
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_5_dense_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    tc = config.text_config
    tc.num_hidden_layers = 1
    tc.layer_types = ["linear_attention"]  # single linear_attn layer
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.linear_key_head_dim = 8
    tc.linear_value_head_dim = 8
    tc.hidden_size = 64
    tc.intermediate_size = 64
    tc.num_attention_heads = 2
    tc.num_key_value_heads = 2
    tc.head_dim = 32
    tc.vocab_size = 256
    tc.mtp_num_hidden_layers = 0
    config.vision_config.num_hidden_layers = 1
    config.vision_config.hidden_size = 64
    config.vision_config.intermediate_size = 64
    config.vision_config.num_attention_heads = 2
    return config


def _save_qkv_adapter(
    path: Path,
    *,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    in_dim: int,
    q_fill: float = FILL_A,
    k_fill: float = FILL_A,
    v_fill: float = FILL_B,
    layer_prefix: str = "base_model.model.model.layers.0.linear_attn",
) -> None:
    """Save a LoRA adapter for split in_proj_q/k/v projections."""
    rank = 1
    weights: dict[str, torch.Tensor] = {
        f"{layer_prefix}.in_proj_q.lora_A.weight": torch.ones(rank, in_dim) * q_fill,
        f"{layer_prefix}.in_proj_q.lora_B.weight": torch.ones(q_dim, rank),
        f"{layer_prefix}.in_proj_k.lora_A.weight": torch.ones(rank, in_dim) * k_fill,
        f"{layer_prefix}.in_proj_k.lora_B.weight": torch.ones(k_dim, rank),
        f"{layer_prefix}.in_proj_v.lora_A.weight": torch.ones(rank, in_dim) * v_fill,
        f"{layer_prefix}.in_proj_v.lora_B.weight": torch.ones(v_dim, rank),
    }

    path.mkdir(parents=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


class TestQwen35DenseSplitQkv:
    """Qwen3.5 dense: split in_proj_q/k/v merged into fused in_proj_qkv.

    Also tests vision model language_model prefix remapping and tied embeddings.
    """

    FUSED_KEY = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"

    def test_qkv_deltas_land_in_correct_slices(self):
        config = _make_tiny_qwen3_5_dense_config()
        tc = config.text_config
        q_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
        k_dim = q_dim
        v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
        in_dim = tc.hidden_size

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3.5-4B",
                is_vision=True,
            )
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = orig.state_dict()[self.FUSED_KEY].clone()

            _save_qkv_adapter(adapter_path, q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, in_dim=in_dim)
            merged_sd = run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

            delta = merged_sd[self.FUSED_KEY] - orig_fused
            q_delta = delta[:q_dim]
            k_delta = delta[q_dim : q_dim + k_dim]
            v_delta = delta[q_dim + k_dim :]

            assert torch.allclose(q_delta, torch.full_like(q_delta, FILL_A), atol=1e-3)
            assert torch.allclose(k_delta, torch.full_like(k_delta, FILL_A), atol=1e-3)
            assert torch.allclose(v_delta, torch.full_like(v_delta, FILL_B), atol=1e-3)

    def test_v_only_does_not_modify_q_or_k(self):
        config = _make_tiny_qwen3_5_dense_config()
        tc = config.text_config
        q_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
        k_dim = q_dim
        v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
        in_dim = tc.hidden_size

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3.5-4B",
                is_vision=True,
            )
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_qk = orig.state_dict()[self.FUSED_KEY][: q_dim + k_dim].clone()

            # Set Q/K fills to 0, only V has nonzero delta
            rank = 1
            prefix = "base_model.model.model.layers.0.linear_attn"
            weights: dict[str, torch.Tensor] = {
                f"{prefix}.in_proj_q.lora_A.weight": torch.zeros(rank, in_dim),
                f"{prefix}.in_proj_q.lora_B.weight": torch.ones(q_dim, rank),
                f"{prefix}.in_proj_k.lora_A.weight": torch.zeros(rank, in_dim),
                f"{prefix}.in_proj_k.lora_B.weight": torch.ones(k_dim, rank),
                f"{prefix}.in_proj_v.lora_A.weight": torch.ones(rank, in_dim) * FILL_B,
                f"{prefix}.in_proj_v.lora_B.weight": torch.ones(v_dim, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            merged_sd = run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

            merged_qk = merged_sd[self.FUSED_KEY][: q_dim + k_dim]
            assert torch.allclose(merged_qk, orig_qk, atol=1e-5), (
                "V-only adapter modified Q or K slices"
            )

    def test_unembed_tokens_merged_into_embed_tokens_when_tied(self):
        """With tie_word_embeddings=True (Qwen3.5-4B), unembed_tokens delta
        must land on model.language_model.embed_tokens."""
        config = _make_tiny_qwen3_5_dense_config()
        tc = config.text_config
        assert tc.tie_word_embeddings is True

        vocab = tc.vocab_size
        hidden = tc.hidden_size

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3.5-4B",
                is_vision=True,
            )
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_embed = orig.state_dict()["model.language_model.embed_tokens.weight"].clone()

            rank = 1
            unembed_fill = 0.02
            weights: dict[str, torch.Tensor] = {
                "base_model.model.model.unembed_tokens.lora_A.weight": (
                    torch.ones(rank, hidden) * unembed_fill
                ),
                "base_model.model.model.unembed_tokens.lora_B.weight": torch.ones(vocab, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            merged_sd = run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

            merged_embed = merged_sd["model.language_model.embed_tokens.weight"]
            delta = (merged_embed - orig_embed).abs().sum()
            assert delta > 0, "embed_tokens was not updated by unembed_tokens adapter"

            expected_delta = torch.full_like(orig_embed, unembed_fill)
            assert torch.allclose(merged_embed - orig_embed, expected_delta, atol=1e-3)


# ---------------------------------------------------------------------------
# Qwen3.5 MoE — split QKV + fused experts + vision prefix
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_5_moe_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)
    tc = config.text_config
    tc.num_hidden_layers = 1
    tc.layer_types = ["linear_attention"]
    tc.linear_num_key_heads = 2
    tc.linear_num_value_heads = 4
    tc.linear_key_head_dim = 8
    tc.linear_value_head_dim = 8
    tc.hidden_size = 64
    tc.intermediate_size = 64
    # Use asymmetric moe_intermediate_size (≠ hidden_size) to catch
    # transposition bugs — real Qwen3.5-35B has hidden=2048, moe_inter=512.
    tc.moe_intermediate_size = 48
    tc.num_attention_heads = 2
    tc.num_key_value_heads = 2
    tc.head_dim = 32
    tc.vocab_size = 256
    tc.mtp_num_hidden_layers = 0
    tc.num_experts = 2
    tc.num_experts_per_tok = 1
    tc.shared_expert_intermediate_size = 64
    config.vision_config.num_hidden_layers = 1
    config.vision_config.hidden_size = 64
    config.vision_config.intermediate_size = 64
    config.vision_config.num_attention_heads = 2
    return config


class TestQwen35MoeSplitQkvAndExperts:
    """Qwen3.5 MoE: split QKV fusion + fused expert weights + vision prefix.

    Tests that both QKV fusion and expert merging work together on a single
    model that has both features.
    """

    FUSED_QKV_KEY = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"

    def test_qkv_and_expert_deltas_applied(self):
        config = _make_tiny_qwen3_5_moe_config()
        tc = config.text_config
        q_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
        v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
        in_dim = tc.hidden_size

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3.5-35B-A3B",
                is_vision=True,
            )

            saved = load_file(str(model_path / "model.safetensors"))
            orig_qkv = saved[self.FUSED_QKV_KEY].clone()
            expert_in_dim = config.text_config.hidden_size
            expert_out_dim = config.text_config.moe_intermediate_size
            num_experts = config.text_config.num_experts

            # Build adapter with both QKV and expert LoRA weights
            rank = 1
            qkv_prefix = "base_model.model.model.layers.0.linear_attn"
            exp_prefix = "base_model.model.model.layers.0.mlp.experts"
            adapter_weights: dict[str, torch.Tensor] = {
                # QKV adapter
                f"{qkv_prefix}.in_proj_q.lora_A.weight": torch.ones(rank, in_dim) * FILL_A,
                f"{qkv_prefix}.in_proj_q.lora_B.weight": torch.ones(q_dim, rank),
                f"{qkv_prefix}.in_proj_k.lora_A.weight": torch.ones(rank, in_dim) * FILL_A,
                f"{qkv_prefix}.in_proj_k.lora_B.weight": torch.ones(q_dim, rank),
                f"{qkv_prefix}.in_proj_v.lora_A.weight": torch.ones(rank, in_dim) * FILL_B,
                f"{qkv_prefix}.in_proj_v.lora_B.weight": torch.ones(v_dim, rank),
                # Expert adapter (broadcast pattern matches real Tinker adapters)
                f"{exp_prefix}.w1.lora_A.weight": (torch.ones(1, rank, expert_in_dim) * FILL_A),
                f"{exp_prefix}.w1.lora_B.weight": torch.ones(num_experts, expert_out_dim, rank),
                f"{exp_prefix}.w3.lora_A.weight": (torch.ones(1, rank, expert_in_dim) * FILL_B),
                f"{exp_prefix}.w3.lora_B.weight": torch.ones(num_experts, expert_out_dim, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(adapter_weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(output_path),
            )

            # Compare saved safetensors directly
            merged = load_file(str(output_path / "model.safetensors"))

            # QKV deltas applied
            qkv_delta = (merged[self.FUSED_QKV_KEY] - orig_qkv).abs().sum()
            assert qkv_delta > 0, "QKV weights not updated"

            # Expert gate deltas applied. Experts are stored fused as
            # `experts.gate_up_proj` of shape (num_experts, 2*moe_inter, hidden);
            # the first moe_inter rows (along dim=1) are gate, the rest are up.
            gate_up_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
            for i in range(num_experts):
                gate_delta = (
                    merged[gate_up_key][i, :expert_out_dim]
                    - saved[gate_up_key][i, :expert_out_dim]
                ).abs().sum()
                assert gate_delta > 0, f"Expert {i} gate_proj not updated"

    def test_down_proj_with_broadcast_w2(self):
        """w2 (down_proj) uses reversed broadcast: A per-expert, B shared."""
        config = _make_tiny_qwen3_5_moe_config()
        num_experts = config.text_config.num_experts
        down_in_dim = config.text_config.moe_intermediate_size
        down_out_dim = config.text_config.hidden_size

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "merged",
            )

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3.5-35B-A3B",
                is_vision=True,
            )

            saved = load_file(str(model_path / "model.safetensors"))

            # w2 adapter: A per-expert, B shared (reversed broadcast)
            rank = 1
            exp_prefix = "base_model.model.model.layers.0.mlp.experts"
            adapter_weights: dict[str, torch.Tensor] = {
                f"{exp_prefix}.w2.lora_A.weight": (
                    torch.ones(num_experts, rank, down_in_dim) * 0.03
                ),
                f"{exp_prefix}.w2.lora_B.weight": torch.ones(1, down_out_dim, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(
                adapter_weights,
                str(adapter_path / "adapter_model.safetensors"),
            )
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(output_path),
            )

            merged = load_file(str(output_path / "model.safetensors"))
            # Experts are stored fused as `experts.down_proj` of shape
            # (num_experts, hidden, moe_inter); slice along dim=0 per expert.
            dk = "model.language_model.layers.0.mlp.experts.down_proj"
            for i in range(num_experts):
                delta = (merged[dk][i] - saved[dk][i]).abs().sum()
                assert delta > 0, f"Expert {i} down_proj not updated"


# ---------------------------------------------------------------------------
# Qwen3.5 MoE — FP8 quantized export (experts-fp8 + vllm)
# ---------------------------------------------------------------------------


class TestQwen35MoeFP8Export:
    """Qwen3.5 MoE: routed experts quantized to FP8, everything else stays BF16.

    Verifies that the compression_config ignore list correctly covers all of
    Qwen 3.5's non-standard nn.Linear modules (linear attention projections,
    vision encoder, shared expert gate) so vLLM doesn't try to load them as FP8.
    """

    def test_routed_experts_fp8_dense_bf16_ignore_list_complete(self):
        config = _make_tiny_qwen3_5_moe_config()
        num_experts = config.text_config.num_experts
        expert_in_dim = config.text_config.hidden_size
        expert_out_dim = config.text_config.moe_intermediate_size

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model"
            adapter_path = root / "adapter"
            output_path = root / "merged"

            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3.5-35B-A3B",
                is_vision=True,
            )

            # Re-save weights in BF16 (from_config creates float32 by default)
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            save_file(
                {k: v.to(torch.bfloat16) for k, v in orig_tensors.items()},
                str(model_path / "model.safetensors"),
            )

            # Build adapter with expert LoRA weights
            rank = 1
            exp_prefix = "base_model.model.model.layers.0.mlp.experts"
            adapter_weights: dict[str, torch.Tensor] = {
                f"{exp_prefix}.w1.lora_A.weight": torch.ones(1, rank, expert_in_dim) * FILL_A,
                f"{exp_prefix}.w1.lora_B.weight": torch.ones(num_experts, expert_out_dim, rank),
                f"{exp_prefix}.w3.lora_A.weight": torch.ones(1, rank, expert_in_dim) * FILL_B,
                f"{exp_prefix}.w3.lora_B.weight": torch.ones(num_experts, expert_out_dim, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(adapter_weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(output_path),
                quantize="experts-fp8",
                serving_format="vllm",
            )

            saved_sd = load_merged_tensors(output_path)
            saved_config = json.loads((output_path / "config.json").read_text())

            # -- Routed experts: FP8 with per-expert scale tensors. Experts
            # are stored fused (`experts.gate_up_proj`, `experts.down_proj`),
            # with weight_scale of shape (num_experts, 1, 1) for per-expert
            # scaling.
            for fused in ("gate_up_proj", "down_proj"):
                key = f"model.language_model.layers.0.mlp.experts.{fused}"
                assert saved_sd[key].dtype == torch.float8_e4m3fn, (
                    f"Routed expert should be FP8: {key}"
                )
                scale_key = key + ".weight_scale"
                assert scale_key in saved_sd, f"Missing scale: {scale_key}"
                assert saved_sd[scale_key].dtype == torch.float32
                assert saved_sd[scale_key].shape[0] == num_experts

            # -- Non-expert weights: BF16, no scale --
            qkv_key = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            assert saved_sd[qkv_key].dtype == torch.bfloat16
            assert qkv_key.removesuffix(".weight") + ".weight_scale" not in saved_sd

            # -- Compressed-tensors config present --
            cc = saved_config.get("compression_config")
            assert cc is not None
            assert "quantization_config" not in saved_config
            assert cc["quant_method"] == "compressed-tensors"
            assert cc["config_groups"]["group_0"]["targets"] == ["Linear"]

            # -- Ignore list covers ALL non-quantized nn.Linear modules --
            # These are the modules that would cause vLLM to fail if missing
            # from the ignore list (they are nn.Linear but not FP8-quantized).
            ignore = set(cc["ignore"])

            # Linear attention projections
            for suffix in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                module = f"model.language_model.layers.0.linear_attn.{suffix}"
                assert module in ignore, f"Linear attention module missing from ignore: {module}"

            # Shared expert projections
            for proj in ("gate_proj", "up_proj", "down_proj"):
                module = f"model.language_model.layers.0.mlp.shared_expert.{proj}"
                assert module in ignore, f"Shared expert module missing from ignore: {module}"

            # Shared expert gate
            assert "model.language_model.layers.0.mlp.shared_expert_gate" in ignore

            # Vision encoder modules (path varies by transformers version:
            # older releases used `model.visual.*`, newer ones nest it under
            # `model.language_model.visual.*`).
            vision_modules_in_ignore = [m for m in ignore if ".visual." in m]
            assert len(vision_modules_in_ignore) > 0, (
                "Vision encoder nn.Linear modules missing from ignore list"
            )

            # Routed experts must NOT be in ignore. (Routed-expert modules end
            # in `.experts` or `.experts.<something>` — distinct from
            # `.shared_expert.*`, which is excluded from quantization.)
            routed_in_ignore = [
                m for m in ignore if ".experts" in m and "shared_expert" not in m
            ]
            assert not routed_in_ignore, (
                f"Quantized routed experts should not be in ignore: {routed_in_ignore}"
            )
