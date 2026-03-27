"""Unit tests for build_hf_model strategy dispatch and the sharded export path.

Uses synthetic safetensors files and adapter weights to test the shard-by-shard
pipeline end-to-end without requiring real HF models or network access.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinker_cookbook.weights._export import build_hf_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_model(model_dir: Path, config_dict: dict, state_dict: dict) -> None:
    """Create a minimal synthetic HF model directory.

    Writes config.json, a single safetensors shard, and a minimal tokenizer.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config_dict))
    save_file(state_dict, str(model_dir / "model.safetensors"))
    # Minimal tokenizer files so AutoTokenizer doesn't fail
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
                "added_tokens": [],
            }
        )
    )


def _create_sharded_model(
    model_dir: Path, config_dict: dict, shards: dict[str, dict[str, torch.Tensor]]
) -> None:
    """Create a synthetic model with multiple safetensors shards."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config_dict))

    weight_map: dict[str, str] = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, str(model_dir / shard_name))
        for key in tensors:
            weight_map[key] = shard_name

    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
                "added_tokens": [],
            }
        )
    )


def _create_adapter(adapter_dir: Path, weights: dict[str, torch.Tensor], config: dict) -> None:
    """Create a synthetic adapter directory."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------


class TestBuildHfModelDispatch:
    def test_invalid_strategy_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="merge_strategy"):
            build_hf_model(
                base_model=str(tmp_path),
                adapter_path=str(tmp_path),
                output_path=str(tmp_path / "out"),
                merge_strategy="invalid",
            )

    def test_dequantize_raises_not_implemented(self, tmp_path: Path):
        with pytest.raises(NotImplementedError, match="dequantize"):
            build_hf_model(
                base_model=str(tmp_path),
                adapter_path=str(tmp_path),
                output_path=str(tmp_path / "out"),
                dequantize=True,
            )

    def test_invalid_dtype_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="dtype"):
            build_hf_model(
                base_model=str(tmp_path),
                adapter_path=str(tmp_path),
                output_path=str(tmp_path / "out"),
                dtype="float8",
            )


# ---------------------------------------------------------------------------
# Shard-by-shard end-to-end (single shard)
# ---------------------------------------------------------------------------


class TestBuildShardedSingleShard:
    """End-to-end test of shard-by-shard merge with a single-shard synthetic model."""

    def test_merges_adapter_into_single_shard(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        # Create a synthetic model with one linear layer
        config = {"architectures": ["TestModel"], "model_type": "test"}
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 4, dtype=torch.float32),
            "model.layers.0.mlp.gate_proj.weight": torch.zeros(8, 4, dtype=torch.float32),
        }
        _create_synthetic_model(model_dir, config, state_dict)

        # Create adapter targeting q_proj
        adapter_weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 1),
        }
        _create_adapter(adapter_dir, adapter_weights, {"lora_alpha": 1, "r": 1})

        # Run sharded merge
        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            merge_strategy="shard",
        )

        # Verify output structure
        assert (output_dir / "config.json").exists()
        assert (output_dir / "model.safetensors").exists() or (
            output_dir / "model.safetensors.index.json"
        ).exists()

        # Load and verify merged weights
        out_tensors = _load_output_tensors(output_dir)
        q_proj = out_tensors["model.layers.0.self_attn.q_proj.weight"]
        gate_proj = out_tensors["model.layers.0.mlp.gate_proj.weight"]

        # q_proj should have LoRA delta applied (all ones from B @ A)
        assert q_proj.abs().sum() > 0
        assert torch.allclose(q_proj, torch.ones(8, 4))

        # gate_proj should be unchanged (no adapter targeting it)
        assert gate_proj.abs().sum() == 0


# ---------------------------------------------------------------------------
# Shard-by-shard end-to-end (multiple shards)
# ---------------------------------------------------------------------------


class TestBuildShardedMultiShard:
    """End-to-end test of shard-by-shard merge with a multi-shard synthetic model."""

    def test_merges_across_shards(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        # Create a model with weights split across two shards
        config = {"architectures": ["TestModel"], "model_type": "test"}
        shards = {
            "model-00001-of-00002.safetensors": {
                "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 4, dtype=torch.float32),
            },
            "model-00002-of-00002.safetensors": {
                "model.layers.0.mlp.gate_proj.weight": torch.zeros(8, 4, dtype=torch.float32),
            },
        }
        _create_sharded_model(model_dir, config, shards)

        # Adapter targets weights in both shards
        adapter_weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 4)
            * 0.5,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 1),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": torch.ones(1, 4) * 0.3,
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.ones(8, 1),
        }
        _create_adapter(adapter_dir, adapter_weights, {"lora_alpha": 1, "r": 1})

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            merge_strategy="shard",
        )

        out_tensors = _load_output_tensors(output_dir)

        # Both should have their respective deltas
        q_proj = out_tensors["model.layers.0.self_attn.q_proj.weight"]
        gate_proj = out_tensors["model.layers.0.mlp.gate_proj.weight"]

        assert torch.allclose(q_proj, torch.full((8, 4), 0.5), atol=1e-6)
        assert torch.allclose(gate_proj, torch.full((8, 4), 0.3), atol=1e-6)


# ---------------------------------------------------------------------------
# Shard-by-shard with separate experts
# ---------------------------------------------------------------------------


class TestBuildShardedSeparateExperts:
    """Shard-by-shard merge with per-expert weights (no fused gate_up_proj)."""

    def test_merges_per_expert_weights(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        num_experts = 2
        config = {"architectures": ["TestMoEModel"], "model_type": "test"}
        state_dict = {
            f"model.layers.0.mlp.experts.{i}.gate_proj.weight": torch.zeros(
                8, 4, dtype=torch.float32
            )
            for i in range(num_experts)
        }
        _create_synthetic_model(model_dir, config, state_dict)

        adapter_weights = {
            "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.ones(
                num_experts, 1, 4
            )
            * 0.1,
            "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.ones(
                num_experts, 8, 1
            ),
        }
        _create_adapter(adapter_dir, adapter_weights, {"lora_alpha": 1, "r": 1})

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            merge_strategy="shard",
        )

        out_tensors = _load_output_tensors(output_dir)
        for i in range(num_experts):
            w = out_tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"]
            assert torch.allclose(w, torch.full((8, 4), 0.1), atol=1e-6), f"Expert {i} incorrect"


# ---------------------------------------------------------------------------
# Shard-by-shard with fused experts (concatenated layout)
# ---------------------------------------------------------------------------


class TestBuildShardedFusedExperts:
    """Shard-by-shard merge with fused gate_up_proj (concatenated layout)."""

    NUM_EXPERTS = 2
    IN_DIM = 4
    OUT_DIM = 4
    FUSED_DIM = OUT_DIM * 2

    def test_merges_gate_and_up_into_correct_halves(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        config = {"architectures": ["TestMoEModel"], "model_type": "test"}
        state_dict = {
            "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(
                self.NUM_EXPERTS, self.IN_DIM, self.FUSED_DIM, dtype=torch.float32
            ),
        }
        _create_synthetic_model(model_dir, config, state_dict)

        # Adapter for gate (w1) and up (w3) projections
        prefix = "base_model.model.model.layers.0.mlp.experts"
        gate_fill, up_fill = 0.02, 0.07
        adapter_weights = {
            f"{prefix}.w1.lora_A.weight": torch.ones(self.NUM_EXPERTS, 1, self.IN_DIM) * gate_fill,
            f"{prefix}.w1.lora_B.weight": torch.ones(self.NUM_EXPERTS, self.OUT_DIM, 1),
            f"{prefix}.w3.lora_A.weight": torch.ones(self.NUM_EXPERTS, 1, self.IN_DIM) * up_fill,
            f"{prefix}.w3.lora_B.weight": torch.ones(self.NUM_EXPERTS, self.OUT_DIM, 1),
        }
        _create_adapter(adapter_dir, adapter_weights, {"lora_alpha": 1, "r": 1})

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            merge_strategy="shard",
        )

        out_tensors = _load_output_tensors(output_dir)
        fused = out_tensors["model.layers.0.mlp.experts.gate_up_proj"]
        sz = self.FUSED_DIM // 2
        gate_half = fused[:, :, :sz]
        up_half = fused[:, :, sz:]

        assert torch.allclose(gate_half, torch.full_like(gate_half, gate_fill), atol=1e-6)
        assert torch.allclose(up_half, torch.full_like(up_half, up_fill), atol=1e-6)

    def test_single_shard_output_has_no_index(self, tmp_path: Path):
        """Single-shard models should produce model.safetensors without an index."""
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        config = {"architectures": ["TestModel"], "model_type": "test"}
        state_dict = {"model.layers.0.proj.weight": torch.zeros(4, 4, dtype=torch.float32)}
        _create_synthetic_model(model_dir, config, state_dict)

        adapter_weights = {
            "base_model.model.model.layers.0.proj.lora_A.weight": torch.ones(1, 4),
            "base_model.model.model.layers.0.proj.lora_B.weight": torch.ones(4, 1),
        }
        _create_adapter(adapter_dir, adapter_weights, {"lora_alpha": 1, "r": 1})

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            merge_strategy="shard",
        )

        assert (output_dir / "model.safetensors").exists()
        assert not (output_dir / "model.safetensors.index.json").exists()


# ---------------------------------------------------------------------------
# Shard-by-shard with Nemotron (Mamba fused in_proj + MoE experts)
# ---------------------------------------------------------------------------


class TestBuildShardedNemotron:
    """Shard-by-shard merge for Nemotron: Mamba gate_proj/x_proj → in_proj + MoE experts."""

    HIDDEN = 8
    INTERMEDIATE = 12
    IN_PROJ_DIM = INTERMEDIATE * 2 + 6  # gate + x + B + C + dt
    NUM_EXPERTS = 2

    def test_mamba_fused_and_moe_experts(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        adapter_dir = tmp_path / "adapter"
        output_dir = tmp_path / "output"

        # Nemotron-like model: backbone prefix, in_proj (fused), per-expert up/down
        config = {"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h"}
        state_dict: dict[str, torch.Tensor] = {
            # Mamba layer (layer 0)
            "backbone.layers.0.mixer.in_proj.weight": torch.zeros(
                self.IN_PROJ_DIM, self.HIDDEN, dtype=torch.float32
            ),
            "backbone.layers.0.mixer.out_proj.weight": torch.zeros(
                self.HIDDEN, self.INTERMEDIATE, dtype=torch.float32
            ),
            # MoE layer (layer 1)
            "backbone.layers.1.mixer.shared_experts.up_proj.weight": torch.zeros(
                self.INTERMEDIATE, self.HIDDEN, dtype=torch.float32
            ),
        }
        for e in range(self.NUM_EXPERTS):
            state_dict[f"backbone.layers.1.mixer.experts.{e}.up_proj.weight"] = torch.zeros(
                self.INTERMEDIATE, self.HIDDEN, dtype=torch.float32
            )
            state_dict[f"backbone.layers.1.mixer.experts.{e}.down_proj.weight"] = torch.zeros(
                self.HIDDEN, self.INTERMEDIATE, dtype=torch.float32
            )
        _create_synthetic_model(model_dir, config, state_dict)

        # Adapter: Mamba gate_proj/x_proj + MoE w1/w2 + empty w3
        prefix = "base_model.model.backbone"
        gate_fill, x_fill, expert_fill = 0.5, 0.3, 0.1
        adapter_weights: dict[str, torch.Tensor] = {
            # Mamba gate_proj/x_proj (will be merged into in_proj)
            f"{prefix}.layers.0.mixer.gate_proj.lora_A.weight": (
                torch.ones(1, self.HIDDEN) * gate_fill
            ),
            f"{prefix}.layers.0.mixer.gate_proj.lora_B.weight": torch.ones(self.INTERMEDIATE, 1),
            f"{prefix}.layers.0.mixer.x_proj.lora_A.weight": (torch.ones(1, self.HIDDEN) * x_fill),
            f"{prefix}.layers.0.mixer.x_proj.lora_B.weight": torch.ones(self.INTERMEDIATE, 1),
            # MoE w1 (up_proj)
            f"{prefix}.layers.1.mixer.experts.w1.lora_A.weight": (
                torch.ones(self.NUM_EXPERTS, 1, self.HIDDEN) * expert_fill
            ),
            f"{prefix}.layers.1.mixer.experts.w1.lora_B.weight": torch.ones(
                self.NUM_EXPERTS, self.INTERMEDIATE, 1
            ),
            # MoE w2 (down_proj)
            f"{prefix}.layers.1.mixer.experts.w2.lora_A.weight": (
                torch.ones(self.NUM_EXPERTS, 1, self.INTERMEDIATE) * expert_fill
            ),
            f"{prefix}.layers.1.mixer.experts.w2.lora_B.weight": torch.ones(
                self.NUM_EXPERTS, self.HIDDEN, 1
            ),
            # Empty w3 placeholder
            f"{prefix}.layers.1.mixer.experts.w3.lora_A.weight": torch.empty(0),
            f"{prefix}.layers.1.mixer.experts.w3.lora_B.weight": torch.empty(0),
        }
        _create_adapter(adapter_dir, adapter_weights, {"lora_alpha": 1, "r": 1})

        build_hf_model(
            base_model=str(model_dir),
            adapter_path=str(adapter_dir),
            output_path=str(output_dir),
            merge_strategy="shard",
        )

        out_tensors = _load_output_tensors(output_dir)

        # Verify Mamba in_proj: gate in [0:INTERMEDIATE], x in [INTERMEDIATE:2*INTERMEDIATE]
        in_proj = out_tensors["backbone.layers.0.mixer.in_proj.weight"]
        gate_slice = in_proj[: self.INTERMEDIATE]
        x_slice = in_proj[self.INTERMEDIATE : 2 * self.INTERMEDIATE]
        rest = in_proj[2 * self.INTERMEDIATE :]

        assert torch.allclose(gate_slice, torch.full_like(gate_slice, gate_fill), atol=1e-6)
        assert torch.allclose(x_slice, torch.full_like(x_slice, x_fill), atol=1e-6)
        assert rest.abs().sum() == 0, "B/C/dt rows should be unchanged"

        # Verify MoE expert up_proj and down_proj merged
        for e in range(self.NUM_EXPERTS):
            up = out_tensors[f"backbone.layers.1.mixer.experts.{e}.up_proj.weight"]
            down = out_tensors[f"backbone.layers.1.mixer.experts.{e}.down_proj.weight"]
            assert torch.allclose(up, torch.full_like(up, expert_fill), atol=1e-6)
            assert torch.allclose(down, torch.full_like(down, expert_fill), atol=1e-6)

        # Verify out_proj unchanged (no adapter targeting it)
        out_proj = out_tensors["backbone.layers.0.mixer.out_proj.weight"]
        assert out_proj.abs().sum() == 0


# ---------------------------------------------------------------------------
# Cleanup on failure
# ---------------------------------------------------------------------------


class TestBuildShardedCleanup:
    def test_cleans_up_on_failure(self, tmp_path: Path):
        output_dir = tmp_path / "output"
        adapter_dir = tmp_path / "adapter"

        # Create adapter but no model — will fail when trying to resolve model dir
        _create_adapter(adapter_dir, {"x.lora_A.weight": torch.zeros(1)}, {"lora_alpha": 1, "r": 1})

        with pytest.raises(Exception):  # noqa: B017
            build_hf_model(
                base_model=str(tmp_path / "nonexistent_model"),
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
                merge_strategy="shard",
            )

        # Output dir should not exist after cleanup
        assert not output_dir.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_output_tensors(output_dir: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from an output directory (single or sharded)."""
    single = output_dir / "model.safetensors"
    if single.exists():
        return load_file(str(single))

    index_path = output_dir / "model.safetensors.index.json"
    assert index_path.exists(), f"No model.safetensors or index.json in {output_dir}"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    tensors: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        tensors.update(load_file(str(output_dir / shard_name)))
    return tensors
