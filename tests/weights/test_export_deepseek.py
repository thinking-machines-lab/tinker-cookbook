"""E2e export tests for DeepSeek V3.1: FP8 quantized export."""

import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, PretrainedConfig

from tests.weights.conftest import FILL_A, FILL_B, save_model_to_disk
from tinker_cookbook.weights import build_hf_model


def _deepseek_needs_custom_code() -> bool:
    """Check if DeepSeek requires trust_remote_code (transformers < 5.0)."""
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
            torch.ones(1, rank, expert_in_dim, dtype=torch.bfloat16) * gate_fill
        ),
        "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(
            num_experts, expert_out_dim, rank, dtype=torch.bfloat16
        ),
        "base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight": (
            torch.ones(1, rank, expert_in_dim, dtype=torch.bfloat16) * up_fill
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

    Uses real DeepSeek config from HF. On transformers < 5.0, uses custom code
    (trust_remote_code). On 5.0+, uses native support.
    """

    def test_dense_weights_change_but_only_routed_experts_are_quantized_to_fp8(self):
        config = _make_tiny_deepseek_v31_config()
        needs_custom = _deepseek_needs_custom_code()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model"
            adapter_path = root / "adapter"
            output_path = root / "merged"

            # Create model in BF16 to match real DeepSeek checkpoint format
            save_model_to_disk(
                config,
                model_path,
                tokenizer_name="deepseek-ai/DeepSeek-V3.1",
                trust_remote_code=needs_custom,
            )
            if needs_custom:
                _copy_hf_files(
                    "deepseek-ai/DeepSeek-V3.1",
                    model_path,
                    ("configuration_deepseek.py", "modeling_deepseek.py"),
                )
            # Re-save weights in BF16 (from_config creates float32 by default).
            # Read from saved safetensors (separate format) and cast to BF16,
            # rather than using state_dict() which may return fused keys on 5.x.
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            save_file(
                {k: v.to(torch.bfloat16) for k, v in orig_tensors.items()},
                str(model_path / "model.safetensors"),
            )
            num_experts = 2

            # Re-read after BF16 conversion
            orig_tensors = load_file(str(model_path / "model.safetensors"))
            gate_shape = orig_tensors["model.layers.0.mlp.experts.0.gate_proj.weight"].shape
            expert_out_dim, expert_in_dim = gate_shape
            dense_shape = orig_tensors["model.layers.0.self_attn.q_a_proj.weight"].shape
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

            # -- Custom files copied (only when trust_remote_code is used) --
            if needs_custom:
                assert (output_path / "configuration_deepseek.py").exists()
                assert (output_path / "modeling_deepseek.py").exists()
            assert (output_path / "model.safetensors.index.json").exists()

            # -- Dense weight: merged, BF16, shard preserved --
            dense_delta = (
                (saved_sd[dense_key].float() - orig_tensors[dense_key].float()).abs().sum()
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
