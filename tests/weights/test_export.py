"""End-to-end tests for build_hf_model across all supported model families.

Each test instantiates a tiny real HuggingFace model from config (no weight
download), saves it to disk with synthetic LoRA adapter weights, runs the
full build_hf_model pipeline, reloads, and verifies correctness.

Model families tested:
- GPT-OSS: fused interleaved gate_up_proj
- Qwen3-VL MoE: fused concatenated gate_up_proj + vision model prefix
- Qwen3 MoE: separate per-expert weights
- DeepSeek V3.1: separate per-expert weights
- Qwen3 dense: standard linear layers (no experts)
"""

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import snapshot_download
import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PretrainedConfig,
)

import tinker_cookbook.weights._deepseek as deepseek_export
from tinker_cookbook.weights import build_hf_model

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FILL_A = 0.01  # LoRA fill for gate / first projection
FILL_B = 0.05  # LoRA fill for up / second projection


@contextmanager
def _local_hf_cache_env():
    """Keep test-time HF dynamic module/cache writes off shared shell defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir) / "hf-cache"
        xdg_root = Path(tmpdir) / "xdg-cache"
        env = {
            "HF_HOME": str(cache_root),
            "HF_HUB_CACHE": str(cache_root / "hub"),
            "HF_XET_CACHE": str(cache_root / "xet"),
            "HF_MODULES_CACHE": str(cache_root / "modules"),
            "XDG_CACHE_HOME": str(xdg_root),
        }
        for value in env.values():
            Path(value).mkdir(parents=True, exist_ok=True)
        old_env = {key: os.environ.get(key) for key in env}
        import huggingface_hub.constants as hf_constants
        import transformers.dynamic_module_utils as dynamic_module_utils
        import transformers.utils.hub as transformers_hub

        old_globals = {
            "HF_HOME": hf_constants.HF_HOME,
            "HF_HUB_CACHE": hf_constants.HF_HUB_CACHE,
            "HF_XET_CACHE": getattr(hf_constants, "HF_XET_CACHE", None),
            "dynamic_HF_MODULES_CACHE": dynamic_module_utils.HF_MODULES_CACHE,
            "hub_HF_MODULES_CACHE": transformers_hub.HF_MODULES_CACHE,
        }
        os.environ.update(env)
        hf_constants.HF_HOME = env["HF_HOME"]
        hf_constants.HF_HUB_CACHE = env["HF_HUB_CACHE"]
        if hasattr(hf_constants, "HF_XET_CACHE"):
            hf_constants.HF_XET_CACHE = env["HF_XET_CACHE"]
        dynamic_module_utils.HF_MODULES_CACHE = env["HF_MODULES_CACHE"]
        transformers_hub.HF_MODULES_CACHE = env["HF_MODULES_CACHE"]
        try:
            yield
        finally:
            hf_constants.HF_HOME = old_globals["HF_HOME"]
            hf_constants.HF_HUB_CACHE = old_globals["HF_HUB_CACHE"]
            if hasattr(hf_constants, "HF_XET_CACHE"):
                hf_constants.HF_XET_CACHE = old_globals["HF_XET_CACHE"]
            dynamic_module_utils.HF_MODULES_CACHE = old_globals["dynamic_HF_MODULES_CACHE"]
            transformers_hub.HF_MODULES_CACHE = old_globals["hub_HF_MODULES_CACHE"]
            for key, previous in old_env.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous


def _save_model_to_disk(
    config: PretrainedConfig,
    path: Path,
    *,
    tokenizer_name: str,
    is_vision: bool = False,
) -> None:
    auto_cls = AutoModelForImageTextToText if is_vision else AutoModelForCausalLM
    model = auto_cls.from_config(config, trust_remote_code=True, dtype=torch.float32)
    try:
        model.save_pretrained(path)
    except FileNotFoundError:
        # Remote-code classes can retain a source path inside a temporary module
        # cache that no longer exists by the time a later test saves the model.
        path.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), str(path / "model.safetensors"))
        (path / "config.json").write_text(config.to_json_string(use_diff=False))
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(path)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tok.save_pretrained(path)


def _copy_hf_files(repo_id: str, output_path: Path, file_names: tuple[str, ...]) -> None:
    snapshot_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=list(file_names)))
    for file_name in file_names:
        shutil.copy2(snapshot_path / file_name, output_path / file_name)


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


def _run_build_and_reload(
    model_path: Path,
    adapter_path: Path,
    output_path: Path,
    *,
    is_vision: bool = False,
    trust_remote_code: bool = True,
) -> dict[str, torch.Tensor]:
    """Run build_hf_model and return the reloaded state dict in arithmetic-safe dtypes."""
    build_hf_model(
        base_model=str(model_path),
        adapter_path=str(adapter_path),
        output_path=str(output_path),
    )
    auto_cls = AutoModelForImageTextToText if is_vision else AutoModelForCausalLM
    load_kwargs: dict[str, object] = {"dtype": torch.float32}
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    reloaded = auto_cls.from_pretrained(output_path, **load_kwargs)
    return {
        key: value.float() if value.is_floating_point() else value
        for key, value in reloaded.state_dict().items()
    }


def _load_saved_state_dict(output_path: Path) -> dict[str, torch.Tensor]:
    """Load tensors exactly as written to disk, preserving saved dtypes."""
    state_dict: dict[str, torch.Tensor] = {}
    for safetensors_path in sorted(output_path.glob("*.safetensors")):
        state_dict.update(load_file(str(safetensors_path)))
    return state_dict


def _load_saved_index(output_path: Path) -> dict:
    return json.loads((output_path / "model.safetensors.index.json").read_text())


def _load_merge_state(output_path: Path) -> dict:
    return json.loads((output_path / "merge_state.json").read_text())


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _load_saved_shard_membership(output_path: Path) -> dict[str, set[str]]:
    membership: dict[str, set[str]] = {}
    for shard_path in sorted(output_path.glob("*.safetensors")):
        membership[shard_path.name] = set(load_file(str(shard_path)).keys())
    return membership


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
    for shard_name, shard_state_dict in sorted(shard_state_dicts.items()):
        save_file(shard_state_dict, str(model_path / shard_name))

    index_payload = {
        "metadata": {"total_size": sum(_tensor_nbytes(tensor) for tensor in state_dict.values())},
        "weight_map": weight_map,
    }
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n"
    )
    return weight_map


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
# 4. DeepSeek V3.1 — separate per-expert weights
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


class TestDeepSeekV31SeparateExperts:
    """DeepSeek V3.1: dense weights stay BF16 while routed experts should be FP8."""

    def test_dense_weights_change_but_only_routed_experts_are_quantized_to_fp8(self):
        with _local_hf_cache_env():
            config = _make_tiny_deepseek_v31_config()

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                model_path = root / "model"
                adapter_path = root / "adapter"
                output_path = root / "merged"

                _save_model_to_disk(config, model_path, tokenizer_name="deepseek-ai/DeepSeek-V3.1")
                _copy_hf_files(
                    "deepseek-ai/DeepSeek-V3.1",
                    model_path,
                    ("configuration_deepseek.py", "modeling_deepseek.py"),
                )
                orig = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
                num_experts = 2

                gate_shape = orig.state_dict()["model.layers.0.mlp.experts.0.gate_proj.weight"].shape
                expert_out_dim, expert_in_dim = gate_shape
                dense_shape = orig.state_dict()["model.layers.0.self_attn.q_a_proj.weight"].shape
                dense_out_dim, dense_in_dim = dense_shape
                dense_key = "model.layers.0.self_attn.q_a_proj.weight"
                shared_expert_key = "model.layers.0.mlp.shared_experts.gate_proj.weight"
                gate_keys = [f"model.layers.0.mlp.experts.{i}.gate_proj.weight" for i in range(num_experts)]
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

                merged_sd = _run_build_and_reload(
                    model_path, adapter_path, output_path, trust_remote_code=False
                )
                saved_sd = _load_saved_state_dict(output_path)
                saved_index = _load_saved_index(output_path)
                shard_membership = _load_saved_shard_membership(output_path)
                saved_config = json.loads((output_path / "config.json").read_text())

                assert (output_path / "configuration_deepseek.py").exists(), (
                    "Merged DeepSeek artifact should include configuration_deepseek.py"
                )
                assert (output_path / "modeling_deepseek.py").exists(), (
                    "Merged DeepSeek artifact should include modeling_deepseek.py"
                )
                assert (output_path / "model.safetensors.index.json").exists(), (
                    "Merged DeepSeek artifact should preserve sharded safetensors metadata"
                )

                dense_delta = (merged_sd[dense_key] - orig.state_dict()[dense_key]).abs().sum()
                assert dense_delta > 0, "Dense non-expert q_a_proj weight was not updated"
                assert saved_sd[dense_key].dtype == torch.bfloat16, (
                    "Dense non-expert q_a_proj should remain higher precision (BF16)"
                )
                assert saved_index["weight_map"][dense_key] == reference_weight_map[dense_key], (
                    "Dense non-expert tensors should preserve the reference shard placement"
                )

                for i in range(num_experts):
                    gate_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
                    up_key = f"model.layers.0.mlp.experts.{i}.up_proj.weight"
                    gate_scale_key = gate_key.removesuffix(".weight") + ".weight_scale"
                    up_scale_key = up_key.removesuffix(".weight") + ".weight_scale"

                    gate_delta = (merged_sd[gate_key] - orig.state_dict()[gate_key]).abs().sum()
                    up_delta = (merged_sd[up_key] - orig.state_dict()[up_key]).abs().sum()

                    assert gate_delta > 0, f"Expert {i} gate_proj not updated"
                    assert up_delta > 0, f"Expert {i} up_proj not updated"
                    assert saved_sd[gate_key].dtype in {
                        torch.float8_e4m3fn,
                        torch.float8_e4m3fnuz,
                        torch.float8_e5m2,
                        torch.float8_e5m2fnuz,
                    }, f"Routed expert weight should be quantized to FP8: {gate_key}"
                    assert saved_sd[gate_scale_key].dtype == torch.float32, (
                        f"Routed expert scale tensor should be stored as float32: {gate_scale_key}"
                    )
                    assert saved_sd[up_scale_key].dtype == torch.float32, (
                        f"Routed expert scale tensor should be stored as float32: {up_scale_key}"
                    )
                    assert saved_index["weight_map"][gate_key] == reference_weight_map[gate_key], (
                        "Routed expert tensors should preserve the reference shard placement"
                    )
                    assert saved_index["weight_map"][up_key] == reference_weight_map[up_key], (
                        "Routed expert tensors should preserve the reference shard placement"
                    )
                    assert saved_index["weight_map"][gate_scale_key] == reference_weight_map[
                        gate_key
                    ], (
                        "Without native reference scale slots, routed expert scales should be "
                        "written alongside their weights"
                    )
                    assert saved_index["weight_map"][up_scale_key] == reference_weight_map[
                        up_key
                    ], (
                        "Without native reference scale slots, routed expert scales should be "
                        "written alongside their weights"
                    )

                assert saved_sd[shared_expert_key].dtype == torch.bfloat16, (
                    "Shared expert weights should remain higher precision (BF16)"
                )
                assert saved_index["weight_map"][shared_expert_key] == reference_weight_map[
                    shared_expert_key
                ], "Shared expert tensors should preserve the reference shard placement"
                assert any(
                    "weight_scale" in key and ".mlp.experts." in key for key in saved_sd
                ), "Expected routed-expert FP8 scale tensors in saved checkpoint"
                assert not any(key.endswith(".weight_scale_inv") for key in saved_sd), (
                    "Experts-only export should emit compressed-tensors .weight_scale tensors, "
                    "not DeepSeek-native .weight_scale_inv tensors"
                )

                saved_weight_map = saved_index["weight_map"]
                assert set(saved_weight_map) == set(saved_sd), (
                    "model.safetensors.index.json should cover every emitted tensor exactly once"
                )
                assert saved_index["metadata"]["total_size"] == sum(
                    _tensor_nbytes(tensor) for tensor in saved_sd.values()
                ), "Index total_size should match the actual emitted tensors"
                assert set(saved_weight_map.values()) == set(shard_membership), (
                    "Every shard referenced by the index should exist, and every emitted shard "
                    "should be referenced by the index"
                )
                for tensor_name, shard_name in saved_weight_map.items():
                    assert tensor_name in shard_membership[shard_name], (
                        f"Index incorrectly maps {tensor_name} to {shard_name}"
                    )

                compression_config = saved_config.get("compression_config")
                assert "quantization_config" not in saved_config, (
                    "DeepSeek export should replace quantization_config with compression_config"
                )
                assert compression_config is not None, (
                    "DeepSeek export should write a compressed-tensors compression_config"
                )
                assert compression_config["quant_method"] == "compressed-tensors"
                assert compression_config["format"] == "float-quantized"
                assert compression_config["quantization_status"] == "compressed"
                assert compression_config["config_groups"]["group_0"]["targets"] == ["Linear"]
                assert (
                    compression_config["config_groups"]["group_0"]["weights"]["strategy"]
                    == "block"
                )
                assert compression_config["config_groups"]["group_0"]["weights"][
                    "block_structure"
                ] == [128, 128]
                assert compression_config["config_groups"]["group_0"]["input_activations"][
                    "dynamic"
                ], "compressed-tensors config should keep dynamic input activations enabled"

                ignore = set(compression_config["ignore"])
                assert "model.layers.0.self_attn.q_a_proj" in ignore, (
                    "Dense non-expert q_a_proj should be explicitly ignored by the compression "
                    "config"
                )
                assert "model.layers.0.mlp.shared_experts.gate_proj" in ignore, (
                    "Shared expert weights should be explicitly ignored by the compression config"
                )
                assert "model.layers.0.mlp.experts.0.gate_proj" not in ignore, (
                    "Routed expert weights should not be ignored by the compression config"
                )

    def test_resume_skips_completed_shards(self, monkeypatch):
        with _local_hf_cache_env():
            config = _make_tiny_deepseek_v31_config()

            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                model_path = root / "model"
                adapter_path = root / "adapter"
                output_path = root / "merged"

                _save_model_to_disk(config, model_path, tokenizer_name="deepseek-ai/DeepSeek-V3.1")
                _copy_hf_files(
                    "deepseek-ai/DeepSeek-V3.1",
                    model_path,
                    ("configuration_deepseek.py", "modeling_deepseek.py"),
                )
                orig = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
                num_experts = 2
                gate_shape = orig.state_dict()["model.layers.0.mlp.experts.0.gate_proj.weight"].shape
                expert_out_dim, expert_in_dim = gate_shape
                dense_shape = orig.state_dict()["model.layers.0.self_attn.q_a_proj.weight"].shape
                dense_out_dim, dense_in_dim = dense_shape

                _reshard_saved_model(
                    model_path,
                    shard_assignments={
                        "model.layers.0.self_attn.q_a_proj.weight": "model-00001-of-00002.safetensors",
                        "model.layers.0.mlp.experts.0.gate_proj.weight": "model-00001-of-00002.safetensors",
                        "model.layers.0.mlp.experts.0.up_proj.weight": "model-00002-of-00002.safetensors",
                        "model.layers.0.mlp.experts.1.gate_proj.weight": "model-00002-of-00002.safetensors",
                        "model.layers.0.mlp.experts.1.up_proj.weight": "model-00001-of-00002.safetensors",
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

                original_save_shard_atomic = deepseek_export._save_shard_atomic
                write_count = {"value": 0}

                def fail_after_first_shard(path, shard_dict):
                    original_save_shard_atomic(path, shard_dict)
                    write_count["value"] += 1
                    if write_count["value"] == 1:
                        raise RuntimeError("forced interruption after first shard")

                monkeypatch.setattr(
                    deepseek_export,
                    "_save_shard_atomic",
                    fail_after_first_shard,
                )
                with pytest.raises(RuntimeError, match="forced interruption"):
                    build_hf_model(
                        base_model=str(model_path),
                        adapter_path=str(adapter_path),
                        output_path=str(output_path),
                    )

                partial_state = _load_merge_state(output_path)
                assert partial_state["status"] == "in_progress"
                completed_shards_on_disk = sorted(path.name for path in output_path.glob("*.safetensors"))
                assert len(completed_shards_on_disk) == 1
                completed_shard = completed_shards_on_disk[0]
                assert (output_path / completed_shard).exists(), (
                    "Interrupted DeepSeek exports should preserve completed shards for resume"
                )

                monkeypatch.setattr(
                    deepseek_export,
                    "_save_shard_atomic",
                    original_save_shard_atomic,
                )
                original_load_raw_shard = deepseek_export._load_raw_shard

                def fail_if_completed_shard_is_reloaded(shard_path):
                    shard_name = Path(shard_path).name
                    if shard_name == completed_shard:
                        raise AssertionError("resume should skip already completed shards")
                    return original_load_raw_shard(shard_path)

                monkeypatch.setattr(
                    deepseek_export,
                    "_load_raw_shard",
                    fail_if_completed_shard_is_reloaded,
                )
                build_hf_model(
                    base_model=str(model_path),
                    adapter_path=str(adapter_path),
                    output_path=str(output_path),
                )

                final_state = _load_merge_state(output_path)
                assert final_state["status"] == "completed"
                assert len(final_state["completed_shards"]) == 2
                assert (output_path / "model.safetensors.index.json").exists()


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
