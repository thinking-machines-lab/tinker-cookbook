"""End-to-end tests for build_lora_adapter across supported model families.

Each test instantiates a tiny real HuggingFace model from config (no weight
download), saves it to disk with synthetic Tinker LoRA adapter weights, runs
the full build_lora_adapter pipeline, and verifies the output PEFT adapter.

Model families tested:
- Qwen3 dense: standard linear layers
- Qwen3 MoE: separate per-expert weight expansion
- GPT-OSS: .attn → .self_attn remap + interleaved expert expansion
- Qwen3-VL MoE: fused concatenated gate_up_proj + vision prefix
- Qwen3.5 dense: split in_proj_q/k/v + vision prefix + tied embeddings
- DeepSeek V3.1: verify unsupported error
"""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PretrainedConfig,
)

from tinker_cookbook.weights import build_hf_model, build_lora_adapter

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FILL_A = 0.01
FILL_B = 0.05


def _save_model_to_disk(
    config: PretrainedConfig,
    path: Path,
    *,
    tokenizer_name: str,
    is_vision: bool = False,
    trust_remote_code: bool = True,
) -> None:
    auto_cls = AutoModelForImageTextToText if is_vision else AutoModelForCausalLM
    model = auto_cls.from_config(config, trust_remote_code=trust_remote_code, dtype=torch.float32)
    model.save_pretrained(path)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
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
    rank = 1
    weights: dict[str, torch.Tensor] = {
        f"{layer_prefix}.w1.lora_A.weight": torch.ones(num_experts, rank, in_dim) * gate_fill,
        f"{layer_prefix}.w1.lora_B.weight": torch.ones(num_experts, out_dim, rank),
        f"{layer_prefix}.w3.lora_A.weight": torch.ones(num_experts, rank, in_dim) * up_fill,
        f"{layer_prefix}.w3.lora_B.weight": torch.ones(num_experts, out_dim, rank),
    }
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


def _load_peft_output(output_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load PEFT adapter output (weights + config)."""
    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    with open(output_dir / "adapter_config.json") as f:
        config = json.load(f)
    return weights, config


def _run_build_adapter(
    model_path: Path,
    adapter_path: Path,
    output_path: Path,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Run build_lora_adapter and return (peft_weights, peft_config)."""
    build_lora_adapter(
        base_model=str(model_path),
        adapter_path=str(adapter_path),
        output_path=str(output_path),
    )
    return _load_peft_output(output_path)


# ---------------------------------------------------------------------------
# 1. Qwen3 dense — standard linear layers
# ---------------------------------------------------------------------------


def _make_tiny_qwen3_dense_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    return config


class TestQwen3DenseAdapter:
    """Qwen3 dense: standard linear layers, no special remapping."""

    def test_peft_keys_match_hf_model_params(self) -> None:
        config = _make_tiny_qwen3_dense_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            _save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-8B")

            # Read actual model dims
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            gate_shape = orig_tensors["model.layers.0.mlp.gate_proj.weight"].shape
            out_dim, in_dim = gate_shape

            _save_dense_adapter(adapter_path, in_dim=in_dim, out_dim=out_dim)
            peft_weights, peft_config = _run_build_adapter(model_path, adapter_path, output_path)

            # PEFT keys should reference actual model parameter paths
            assert "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight" in peft_weights
            assert "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight" in peft_weights
            assert peft_config["peft_type"] == "LORA"
            assert "gate_proj" in peft_config["target_modules"]

    def test_mathematical_equivalence_with_merge(self) -> None:
        """Verify adapter conversion is lossless: merge path == adapter + manual delta."""
        config = _make_tiny_qwen3_dense_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model"
            adapter_path = root / "adapter"
            merged_path = root / "merged"
            peft_path = root / "peft"

            _save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-8B")
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            gate_shape = orig_tensors["model.layers.0.mlp.gate_proj.weight"].shape
            out_dim, in_dim = gate_shape

            _save_dense_adapter(adapter_path, in_dim=in_dim, out_dim=out_dim)

            # Path 1: merge into base model
            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(merged_path),
            )
            merged_tensors = load_file(str(merged_path / "model.safetensors"))

            # Path 2: build PEFT adapter, manually apply delta
            peft_weights, peft_config = _run_build_adapter(model_path, adapter_path, peft_path)
            alpha = peft_config["lora_alpha"]
            rank = peft_config["r"]
            scaling = alpha / rank

            lora_A = peft_weights["base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"]
            lora_B = peft_weights["base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight"]
            delta = (lora_B.float() @ lora_A.float()) * scaling
            manual_merged = orig_tensors["model.layers.0.mlp.gate_proj.weight"].float() + delta

            assert torch.allclose(
                merged_tensors["model.layers.0.mlp.gate_proj.weight"].float(),
                manual_merged,
                atol=1e-3,
            ), "Merge path and adapter+manual delta path should produce identical results"


# ---------------------------------------------------------------------------
# 2. Qwen3 MoE — separate per-expert expansion
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


class TestQwen3MoeAdapter:
    """Qwen3 MoE: 3D expert tensors expanded to per-expert 2D PEFT keys."""

    def test_expert_expansion(self) -> None:
        config = _make_tiny_qwen3_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            _save_model_to_disk(config, model_path, tokenizer_name="Qwen/Qwen3-30B-A3B")
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            gate_shape = orig_tensors["model.layers.0.mlp.experts.0.gate_proj.weight"].shape
            expert_out_dim, expert_in_dim = gate_shape
            num_experts = 2

            _save_expert_adapter(
                adapter_path, num_experts=num_experts, in_dim=expert_in_dim, out_dim=expert_out_dim
            )
            peft_weights, peft_config = _run_build_adapter(model_path, adapter_path, output_path)

            # Per-expert keys should exist
            for i in range(num_experts):
                for proj in ("gate_proj", "up_proj"):
                    key_a = f"base_model.model.model.layers.0.mlp.experts.{i}.{proj}.lora_A.weight"
                    key_b = f"base_model.model.model.layers.0.mlp.experts.{i}.{proj}.lora_B.weight"
                    assert key_a in peft_weights, f"Missing {key_a}"
                    assert key_b in peft_weights, f"Missing {key_b}"
                    assert peft_weights[key_a].ndim == 2
                    assert peft_weights[key_b].ndim == 2

            assert "gate_proj" in peft_config["target_modules"]
            assert "up_proj" in peft_config["target_modules"]


# ---------------------------------------------------------------------------
# 3. GPT-OSS — .attn → .self_attn remap + expert expansion
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


class TestGptOssAdapter:
    """GPT-OSS: .attn → .self_attn remap in PEFT output."""

    def test_attn_to_self_attn_and_expert_expansion(self) -> None:
        config = _make_tiny_gpt_oss_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            _save_model_to_disk(config, model_path, tokenizer_name="openai/gpt-oss-20b")
            orig_tensors = load_file(str(model_path / "model.safetensors"))

            # Get attention dims
            q_shape = orig_tensors["model.layers.0.self_attn.q_proj.weight"].shape
            attn_out_dim, attn_in_dim = q_shape

            # Adapter with .attn naming (Tinker internal) + expert weights
            rank = 1
            fused_key = "model.layers.0.mlp.experts.gate_up_proj"
            num_experts, expert_in_dim, fused_dim = orig_tensors[fused_key].shape
            expert_out_dim = fused_dim // 2

            weights: dict[str, torch.Tensor] = {
                # Attention with .attn naming
                "base_model.model.model.layers.0.attn.q_proj.lora_A.weight": (
                    torch.ones(rank, attn_in_dim) * FILL_A
                ),
                "base_model.model.model.layers.0.attn.q_proj.lora_B.weight": torch.ones(
                    attn_out_dim, rank
                ),
                # Expert weights
                "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": (
                    torch.ones(num_experts, rank, expert_in_dim) * FILL_A
                ),
                "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(
                    num_experts, expert_out_dim, rank
                ),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            peft_weights, peft_config = _run_build_adapter(model_path, adapter_path, output_path)

            # .attn should be remapped to .self_attn
            assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in peft_weights
            assert not any(".attn." in k for k in peft_weights)

            # Expert keys should be expanded
            for i in range(num_experts):
                assert (
                    f"base_model.model.model.layers.0.mlp.experts.{i}.gate_proj.lora_A.weight"
                    in peft_weights
                )

            assert "q_proj" in peft_config["target_modules"]
            assert "gate_proj" in peft_config["target_modules"]


# ---------------------------------------------------------------------------
# 4. Qwen3-VL MoE — fused concatenated gate_up_proj + vision prefix
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


class TestQwen3VlMoeAdapter:
    """Qwen3-VL MoE: vision prefix + expert expansion."""

    def test_vision_prefix_and_expert_expansion(self) -> None:
        config = _make_tiny_qwen3_vl_moe_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            _save_model_to_disk(
                config,
                model_path,
                tokenizer_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                is_vision=True,
            )
            orig_tensors = load_file(str(model_path / "model.safetensors"))

            # Transformers 5.x saves fused expert keys (gate_up_proj).
            # Read dims from the fused key.
            fused_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
            if fused_key in orig_tensors:
                num_experts, fused_dim, expert_in_dim = orig_tensors[fused_key].shape
                expert_out_dim = fused_dim // 2
            else:
                # Older transformers: per-expert keys
                gate_key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
                gate_shape = orig_tensors[gate_key].shape
                expert_out_dim, expert_in_dim = gate_shape
                num_experts = 2

            # Adapter uses model.layers... (without language_model prefix)
            _save_expert_adapter(
                adapter_path, num_experts=num_experts, in_dim=expert_in_dim, out_dim=expert_out_dim
            )
            peft_weights, peft_config = _run_build_adapter(model_path, adapter_path, output_path)

            # PEFT keys should have model.language_model prefix
            for i in range(num_experts):
                key = f"base_model.model.model.language_model.layers.0.mlp.experts.{i}.gate_proj.lora_A.weight"
                assert key in peft_weights, f"Missing vision-prefixed expert key: {key}"

            assert peft_config["peft_type"] == "LORA"


# ---------------------------------------------------------------------------
# 5. Qwen3.5 dense — split QKV + vision prefix
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


class TestQwen35DenseAdapter:
    """Qwen3.5 dense: split in_proj_q/k/v preserved as separate PEFT keys.

    Uses the same tiny config as the merge e2e test (test_export.py) to
    ensure we get a linear_attention layer with in_proj_qkv.
    """

    FUSED_KEY = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"

    def test_split_qkv_keys_in_peft_output(self) -> None:
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
                root / "peft",
            )

            _save_model_to_disk(
                config, model_path, tokenizer_name="Qwen/Qwen3.5-4B", is_vision=True
            )
            _save_qkv_adapter(adapter_path, q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, in_dim=in_dim)

            peft_weights, peft_config = _run_build_adapter(model_path, adapter_path, output_path)

            # Split QKV should be preserved as separate PEFT keys with vision prefix.
            for proj in ("in_proj_q", "in_proj_k", "in_proj_v"):
                matching = [k for k in peft_weights if proj in k]
                assert matching, f"Missing {proj} in PEFT output"
                assert all("language_model" in k for k in matching), (
                    f"{proj} keys missing language_model prefix"
                )
                assert proj in peft_config["target_modules"]

            # All tensors should be 2D and unscaled.
            for key, tensor in peft_weights.items():
                assert tensor.ndim == 2, f"{key} is {tensor.ndim}D"

    def test_mathematical_equivalence_with_merge(self) -> None:
        """Verify the PEFT adapter produces the same delta as the merge path.

        The merge path computes: fused_qkv += [B_q@A_q; B_k@A_k; B_v@A_v] * (alpha/r)
        The adapter path should produce separate in_proj_q/k/v tensors that,
        when manually applied with the same formula, yield the identical delta.
        """
        config = _make_tiny_qwen3_5_dense_config()
        tc = config.text_config
        q_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
        k_dim = q_dim
        v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
        in_dim = tc.hidden_size

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model"
            adapter_path = root / "adapter"
            merged_path = root / "merged"
            peft_path = root / "peft"

            _save_model_to_disk(
                config, model_path, tokenizer_name="Qwen/Qwen3.5-4B", is_vision=True
            )
            orig = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32
            )
            orig_fused = orig.state_dict()[self.FUSED_KEY].clone()
            del orig

            _save_qkv_adapter(adapter_path, q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, in_dim=in_dim)

            # Path 1: merge into base model
            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(merged_path),
            )
            merged_model = AutoModelForImageTextToText.from_pretrained(
                merged_path, trust_remote_code=True, dtype=torch.float32
            )
            merged_fused = merged_model.state_dict()[self.FUSED_KEY].clone()
            del merged_model
            merge_delta = merged_fused - orig_fused

            # Path 2: build PEFT adapter, manually compute delta
            peft_weights, peft_config = _run_build_adapter(model_path, adapter_path, peft_path)
            alpha = peft_config["lora_alpha"]
            rank = peft_config["r"]
            scaling = alpha / rank

            prefix = "base_model.model.model.language_model.layers.0.linear_attn"
            q_A = peft_weights[f"{prefix}.in_proj_q.lora_A.weight"].float()
            q_B = peft_weights[f"{prefix}.in_proj_q.lora_B.weight"].float()
            k_A = peft_weights[f"{prefix}.in_proj_k.lora_A.weight"].float()
            k_B = peft_weights[f"{prefix}.in_proj_k.lora_B.weight"].float()
            v_A = peft_weights[f"{prefix}.in_proj_v.lora_A.weight"].float()
            v_B = peft_weights[f"{prefix}.in_proj_v.lora_B.weight"].float()

            # Manual delta: concatenate [B_q@A_q, B_k@A_k, B_v@A_v] * scaling
            peft_delta = torch.cat([q_B @ q_A, k_B @ k_A, v_B @ v_A], dim=0) * scaling

            assert torch.allclose(merge_delta, peft_delta, atol=1e-3), (
                "Merge path and PEFT adapter + manual delta should produce identical results "
                f"for Qwen3.5 split QKV. Max diff: {(merge_delta - peft_delta).abs().max()}"
            )

    def test_tied_embeddings_remapped(self) -> None:
        """With tie_word_embeddings=True (Qwen3.5-4B), unembed_tokens should
        be remapped to embed_tokens in the PEFT output."""
        config = _make_tiny_qwen3_5_dense_config()
        tc = config.text_config
        assert tc.tie_word_embeddings is True

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path, adapter_path, output_path = (
                root / "model",
                root / "adapter",
                root / "peft",
            )

            _save_model_to_disk(
                config, model_path, tokenizer_name="Qwen/Qwen3.5-4B", is_vision=True
            )
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            embed_shape = orig_tensors["model.language_model.embed_tokens.weight"].shape
            vocab_size, hidden = embed_shape

            rank = 1
            weights = {
                "base_model.model.model.unembed_tokens.lora_A.weight": (
                    torch.ones(rank, hidden) * FILL_A
                ),
                "base_model.model.model.unembed_tokens.lora_B.weight": torch.ones(vocab_size, rank),
            }
            adapter_path.mkdir(parents=True)
            save_file(weights, str(adapter_path / "adapter_model.safetensors"))
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"lora_alpha": 1, "r": rank})
            )

            peft_weights, _ = _run_build_adapter(model_path, adapter_path, output_path)

            # Should be remapped to embed_tokens (not lm_head) due to tied embeddings.
            embed_keys = [k for k in peft_weights if "embed_tokens" in k]
            lm_head_keys = [k for k in peft_weights if "lm_head" in k]
            assert embed_keys, "unembed_tokens should map to embed_tokens for tied embeddings"
            assert not lm_head_keys, "Should not have lm_head keys for tied embeddings"


# ---------------------------------------------------------------------------
# 6. DeepSeek V3.1 — verify unsupported error
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

            _save_model_to_disk(
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

            from tinker_cookbook.exceptions import WeightsAdapterError

            with __import__("pytest").raises(WeightsAdapterError, match="DeepSeek"):
                build_lora_adapter(
                    base_model=str(model_path),
                    adapter_path=str(adapter_path),
                    output_path=str(output_path),
                )
