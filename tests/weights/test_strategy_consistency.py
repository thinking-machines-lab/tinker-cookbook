"""Verify that merge_strategy='full' and merge_strategy='shard' produce identical output.

Uses a tiny real Qwen3 dense model to exercise the full pipeline including
tokenizer saving, config handling, and actual HF model loading.

Requires network access to download Qwen3-8B config + tokenizer on first run
(cached by HF Hub afterwards).
"""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from tinker_cookbook.weights import build_hf_model

FILL = 0.01


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


def _save_model_to_disk(config: PretrainedConfig, path: Path) -> None:
    # Save in bfloat16 to match real-world models. This ensures both full
    # (which loads as bfloat16 by default) and shard (which preserves on-disk
    # dtype) paths work with the same precision.
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.bfloat16)
    model.save_pretrained(path)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    tok.save_pretrained(path)


def _save_adapter(path: Path, *, model_path: Path) -> None:
    """Create adapter with LoRA weights matching the model's actual dimensions."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16
    )
    sd = model.state_dict()

    rank = 1
    gate_shape = sd["model.layers.0.mlp.gate_proj.weight"].shape  # (out, in)
    q_shape = sd["model.layers.0.self_attn.q_proj.weight"].shape  # (out, in)

    weights = {
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": (
            torch.ones(rank, gate_shape[1]) * FILL
        ),
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.ones(
            gate_shape[0], rank
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (
            torch.ones(rank, q_shape[1]) * FILL
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
            q_shape[0], rank
        ),
    }
    path.mkdir(parents=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


def _load_all_tensors(output_dir: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors from output directory."""
    single = output_dir / "model.safetensors"
    if single.exists():
        return load_file(str(single))
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        tensors.update(load_file(str(output_dir / shard_name)))
    return tensors


class TestStrategyConsistency:
    """Verify full and shard strategies produce identical merged weights."""

    def test_full_and_shard_produce_identical_weights(self):
        config = _make_tiny_qwen3_dense_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model"
            adapter_path = root / "adapter"
            output_full = root / "merged_full"
            output_shard = root / "merged_shard"

            _save_model_to_disk(config, model_path)
            _save_adapter(adapter_path, model_path=model_path)

            # Run both strategies
            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(output_full),
                merge_strategy="full",
            )
            build_hf_model(
                base_model=str(model_path),
                adapter_path=str(adapter_path),
                output_path=str(output_shard),
                merge_strategy="shard",
            )

            # Load both outputs
            full_tensors = _load_all_tensors(output_full)
            shard_tensors = _load_all_tensors(output_shard)

            # Same keys
            assert set(full_tensors.keys()) == set(shard_tensors.keys()), (
                f"Key mismatch: "
                f"full_only={set(full_tensors.keys()) - set(shard_tensors.keys())}, "
                f"shard_only={set(shard_tensors.keys()) - set(full_tensors.keys())}"
            )

            # Same values (bit-identical)
            mismatches = []
            for key in sorted(full_tensors.keys()):
                if not torch.equal(full_tensors[key], shard_tensors[key]):
                    max_diff = (full_tensors[key].float() - shard_tensors[key].float()).abs().max()
                    mismatches.append(f"{key}: max_diff={max_diff:.6e}")

            assert not mismatches, (
                "Weight mismatches between full and shard strategies:\n" + "\n".join(mismatches)
            )

            # Both should have config.json
            assert (output_full / "config.json").exists()
            assert (output_shard / "config.json").exists()

            # Verify the merge actually changed something
            orig = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.bfloat16
            )
            orig_gate = orig.state_dict()["model.layers.0.mlp.gate_proj.weight"]
            merged_gate = full_tensors["model.layers.0.mlp.gate_proj.weight"]
            delta = (merged_gate.float() - orig_gate.float()).abs().sum()
            assert delta > 0, "Merge did not modify gate_proj"
