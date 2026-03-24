"""End-to-end tests for build_hf_model across all supported model families.

Each test instantiates a tiny real HuggingFace model from config (no weight
download), saves it to disk with synthetic LoRA adapter weights, runs the
full build_hf_model pipeline, reloads, and verifies correctness.

Model families tested:
- GPT-OSS: fused interleaved gate_up_proj
- Qwen3-VL MoE: fused concatenated gate_up_proj + vision model prefix
- Qwen3 MoE: separate per-expert weights
- DeepSeek V3.1: separate per-expert weights + FP8 quantized export
- Qwen3 dense: standard linear layers (no experts)
- Qwen3.5 dense: split in_proj_q/k/v → fused in_proj_qkv + tied vision embeddings
"""

import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
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
# 4. DeepSeek V3.1 — separate per-expert weights + FP8 quantized export
# ---------------------------------------------------------------------------


def _make_tiny_deepseek_v31_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3.1", trust_remote_code=True)
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


def _copy_hf_files(repo_id: str, output_path: Path, file_names: tuple[str, ...]) -> None:
    """Download specific files from a HF repo and copy to output_path."""
    snapshot_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=list(file_names)))
    for file_name in file_names:
        shutil.copy2(snapshot_path / file_name, output_path / file_name)


def _save_mixed_deepseek_adapter(
    path: Path,
    *,
    num_experts: int,
    expert_in_dim: int,
    expert_out_dim: int,
    dense_in_dim: int,
    dense_out_dim: int,
    dense_fill: float = FILL_A,
    gate_fill: float = FILL_A,
    up_fill: float = FILL_B,
) -> None:
    """Save a DeepSeek adapter with both dense and routed-expert LoRA weights."""
    rank = 1
    weights: dict[str, torch.Tensor] = {
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": (
            torch.ones(rank, dense_in_dim, dtype=torch.bfloat16) * dense_fill
        ),
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.ones(
            dense_out_dim, rank, dtype=torch.bfloat16
        ),
        "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": (
            torch.ones(num_experts, rank, expert_in_dim, dtype=torch.bfloat16) * gate_fill
        ),
        "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(
            num_experts, expert_out_dim, rank, dtype=torch.bfloat16
        ),
        "base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight": (
            torch.ones(num_experts, rank, expert_in_dim, dtype=torch.bfloat16) * up_fill
        ),
        "base_model.model.model.layers.0.mlp.experts.w3.lora_B.weight": torch.ones(
            num_experts, expert_out_dim, rank, dtype=torch.bfloat16
        ),
    }

    path.mkdir(parents=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


def _reshard_saved_model(
    model_path: Path,
    *,
    shard_assignments: dict[str, str],
    default_shard: str = "model-00002-of-00002.safetensors",
) -> dict[str, str]:
    """Rewrite a local checkpoint into a small sharded layout with an HF index."""
    source_path = model_path / "model.safetensors"
    state_dict = load_file(str(source_path))
    shard_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    weight_map: dict[str, str] = {}

    for key, tensor in state_dict.items():
        shard_name = shard_assignments.get(key, default_shard)
        shard_state_dicts.setdefault(shard_name, {})[key] = tensor
        weight_map[key] = shard_name

    source_path.unlink()
    for shard_name, shard_sd in sorted(shard_state_dicts.items()):
        save_file(shard_sd, str(model_path / shard_name))

    total_size = sum(t.nelement() * t.element_size() for t in state_dict.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (model_path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return weight_map


def _load_saved_state_dict(output_path: Path) -> dict[str, torch.Tensor]:
    """Load tensors exactly as written to disk, preserving saved dtypes."""
    state_dict: dict[str, torch.Tensor] = {}
    for safetensors_path in sorted(output_path.glob("*.safetensors")):
        state_dict.update(load_file(str(safetensors_path)))
    return state_dict


class TestDeepSeekV31FP8Export:
    """DeepSeek V3.1: dense weights stay BF16 while routed experts are quantized to FP8.

    Uses real DeepSeek config from HF (downloads config + custom code, not weights).
    """

    def test_dense_weights_change_but_only_routed_experts_are_quantized_to_fp8(self):
        config = _make_tiny_deepseek_v31_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model"
            adapter_path = root / "adapter"
            output_path = root / "merged"

            # Create model in BF16 to match real DeepSeek checkpoint format
            _save_model_to_disk(config, model_path, tokenizer_name="deepseek-ai/DeepSeek-V3.1")
            _copy_hf_files(
                "deepseek-ai/DeepSeek-V3.1",
                model_path,
                ("configuration_deepseek.py", "modeling_deepseek.py"),
            )
            # Re-save weights in BF16 (from_config creates float32 by default)
            orig = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
            save_file(
                {k: v.to(torch.bfloat16) for k, v in orig.state_dict().items()},
                str(model_path / "model.safetensors"),
            )
            orig = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
            num_experts = 2

            gate_shape = orig.state_dict()["model.layers.0.mlp.experts.0.gate_proj.weight"].shape
            expert_out_dim, expert_in_dim = gate_shape
            dense_shape = orig.state_dict()["model.layers.0.self_attn.q_a_proj.weight"].shape
            dense_out_dim, dense_in_dim = dense_shape
            dense_key = "model.layers.0.self_attn.q_a_proj.weight"
            shared_expert_key = "model.layers.0.mlp.shared_experts.gate_proj.weight"
            gate_keys = [
                f"model.layers.0.mlp.experts.{i}.gate_proj.weight" for i in range(num_experts)
            ]
            up_keys = [f"model.layers.0.mlp.experts.{i}.up_proj.weight" for i in range(num_experts)]

            reference_weight_map = _reshard_saved_model(
                model_path,
                shard_assignments={
                    dense_key: "model-00001-of-00002.safetensors",
                    shared_expert_key: "model-00002-of-00002.safetensors",
                    gate_keys[0]: "model-00001-of-00002.safetensors",
                    up_keys[0]: "model-00002-of-00002.safetensors",
                    gate_keys[1]: "model-00002-of-00002.safetensors",
                    up_keys[1]: "model-00001-of-00002.safetensors",
                },
            )

            _save_mixed_deepseek_adapter(
                adapter_path,
                num_experts=num_experts,
                expert_in_dim=expert_in_dim,
                expert_out_dim=expert_out_dim,
                dense_in_dim=dense_in_dim,
                dense_out_dim=dense_out_dim,
            )

            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(output_path),
                quantize="experts-fp8",
                serving_format="vllm",
            )

            saved_sd = _load_saved_state_dict(output_path)
            saved_index = json.loads((output_path / "model.safetensors.index.json").read_text())
            saved_config = json.loads((output_path / "config.json").read_text())

            # -- Custom files copied --
            assert (output_path / "configuration_deepseek.py").exists()
            assert (output_path / "modeling_deepseek.py").exists()
            assert (output_path / "model.safetensors.index.json").exists()

            # -- Dense weight: merged, BF16, shard preserved --
            dense_delta = (
                (saved_sd[dense_key].float() - orig.state_dict()[dense_key].float()).abs().sum()
            )
            assert dense_delta > 0, "Dense q_a_proj weight was not updated"
            assert saved_sd[dense_key].dtype == torch.bfloat16
            assert saved_index["weight_map"][dense_key] == reference_weight_map[dense_key], (
                "Dense tensor should preserve reference shard placement"
            )

            # -- Routed experts: merged, FP8, scale present, shard preserved --
            for i in range(num_experts):
                gate_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
                up_key = f"model.layers.0.mlp.experts.{i}.up_proj.weight"
                gate_scale_key = gate_key.removesuffix(".weight") + ".weight_scale"
                up_scale_key = up_key.removesuffix(".weight") + ".weight_scale"

                assert saved_sd[gate_key].dtype == torch.float8_e4m3fn, (
                    f"Routed expert should be FP8: {gate_key}"
                )
                assert saved_sd[gate_scale_key].dtype == torch.float32, (
                    f"Scale should be float32: {gate_scale_key}"
                )
                assert saved_sd[up_scale_key].dtype == torch.float32
                assert saved_index["weight_map"][gate_key] == reference_weight_map[gate_key], (
                    "Routed expert should preserve reference shard placement"
                )
                assert (
                    saved_index["weight_map"][gate_scale_key] == reference_weight_map[gate_key]
                ), "Scale should be in same shard as weight"

            # -- Shared experts: BF16, not quantized, shard preserved --
            assert saved_sd[shared_expert_key].dtype == torch.bfloat16
            assert (
                saved_index["weight_map"][shared_expert_key]
                == reference_weight_map[shared_expert_key]
            )

            # -- No .weight_scale_inv in output (compressed-tensors convention) --
            assert not any(key.endswith(".weight_scale_inv") for key in saved_sd), (
                "Should emit .weight_scale, not .weight_scale_inv"
            )

            # -- Index consistency --
            assert set(saved_index["weight_map"]) == set(saved_sd)
            shard_membership: dict[str, set[str]] = {}
            for shard_path in sorted(output_path.glob("*.safetensors")):
                shard_membership[shard_path.name] = set(load_file(str(shard_path)).keys())
            assert set(saved_index["weight_map"].values()) == set(shard_membership)

            # -- Compressed-tensors config --
            cc = saved_config.get("compression_config")
            assert "quantization_config" not in saved_config
            assert cc is not None
            assert cc["quant_method"] == "compressed-tensors"
            assert cc["format"] == "float-quantized"
            assert cc["quantization_status"] == "compressed"
            assert cc["config_groups"]["group_0"]["targets"] == ["Linear"]
            assert cc["config_groups"]["group_0"]["weights"]["strategy"] == "block"
            assert cc["config_groups"]["group_0"]["weights"]["block_structure"] == [128, 128]
            assert cc["config_groups"]["group_0"]["input_activations"]["dynamic"] is True

            ignore = set(cc["ignore"])
            assert "model.layers.0.self_attn.q_a_proj" in ignore
            assert "model.layers.0.mlp.shared_experts.gate_proj" in ignore
            assert "model.layers.0.mlp.experts.0.gate_proj" not in ignore


# ---------------------------------------------------------------------------
# 5. Qwen3 dense — standard linear layers (no experts)
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


# ---------------------------------------------------------------------------
# 6. Qwen3.5 dense — split in_proj_q/k/v → fused in_proj_qkv + tied embeddings
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

            _save_model_to_disk(
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
            merged_sd = _run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

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

            _save_model_to_disk(
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

            merged_sd = _run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

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

            _save_model_to_disk(
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

            merged_sd = _run_build_and_reload(model_path, adapter_path, output_path, is_vision=True)

            merged_embed = merged_sd["model.language_model.embed_tokens.weight"]
            delta = (merged_embed - orig_embed).abs().sum()
            assert delta > 0, "embed_tokens was not updated by unembed_tokens adapter"

            expected_delta = torch.full_like(orig_embed, unembed_fill)
            assert torch.allclose(merged_embed - orig_embed, expected_delta, atol=1e-3)
