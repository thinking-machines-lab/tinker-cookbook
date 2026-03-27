"""E2e export tests for Nemotron-3: Mamba fused in_proj + MoE experts.

Tests run build_hf_model (shard strategy) on the real Nemotron-3-Nano model
with a real Tinker adapter and verify that:
- Mamba gate_proj/x_proj are merged into the correct rows of in_proj
- MoE expert up_proj/down_proj are merged correctly
- Attention q_proj is merged correctly
- Non-targeted weights are unchanged

Requires:
- TINKER_API_KEY for adapter download (or pre-downloaded adapter)
- ~60GB disk for model weights
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from tinker_cookbook.weights._export import build_hf_model

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ADAPTER_DIR = "/tmp/nemotron_adapter_dl"


def _have_adapter() -> bool:
    """Check if the real Tinker adapter is available."""
    p = Path(ADAPTER_DIR)
    return (p / "adapter_model.safetensors").exists() and (p / "adapter_config.json").exists()


@pytest.mark.skipif(not _have_adapter(), reason="Real Tinker adapter not available")
class TestNemotronNanoExport:
    """Full weight merge on real Nemotron-3-Nano (30B) with real Tinker adapter."""

    # mamba_num_heads (64) * mamba_head_dim (64) for Nano
    MAMBA_INTERMEDIATE = 4096

    @pytest.fixture(scope="class")
    def merged_output(self, tmp_path_factory):
        """Run build_hf_model once for the class, return output path."""
        output_dir = tmp_path_factory.mktemp("nemotron_merged") / "output"
        build_hf_model(
            base_model=MODEL,
            adapter_path=ADAPTER_DIR,
            output_path=str(output_dir),
            merge_strategy="shard",
        )
        return output_dir

    @pytest.fixture(scope="class")
    def merged_index(self, merged_output):
        with open(merged_output / "model.safetensors.index.json") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def orig_index(self):
        path = hf_hub_download(MODEL, "model.safetensors.index.json")
        with open(path) as f:
            return json.load(f)

    def _load_delta(self, key, merged_output, merged_index, orig_index):
        """Load merged and original weights and return the delta."""
        merged = load_file(str(merged_output / merged_index["weight_map"][key]))[key]
        orig = load_file(hf_hub_download(MODEL, orig_index["weight_map"][key]))[key]
        return merged.float() - orig.float()

    def test_mamba_in_proj_gate_slice_has_delta(self, merged_output, merged_index, orig_index):
        """gate_proj LoRA should appear in first intermediate_size rows of in_proj."""
        delta = self._load_delta(
            "backbone.layers.0.mixer.in_proj.weight",
            merged_output,
            merged_index,
            orig_index,
        )
        intermediate = self.MAMBA_INTERMEDIATE
        gate_delta = delta[:intermediate]
        assert gate_delta.norm() > 0, "gate_proj slice should have non-zero delta"

    def test_mamba_in_proj_x_slice_has_delta(self, merged_output, merged_index, orig_index):
        """x_proj LoRA should appear in rows [intermediate:2*intermediate] of in_proj."""
        delta = self._load_delta(
            "backbone.layers.0.mixer.in_proj.weight",
            merged_output,
            merged_index,
            orig_index,
        )
        intermediate = self.MAMBA_INTERMEDIATE
        x_delta = delta[intermediate : 2 * intermediate]
        assert x_delta.norm() > 0, "x_proj slice should have non-zero delta"

    def test_mamba_in_proj_bcd_rows_unchanged(self, merged_output, merged_index, orig_index):
        """B/C/dt rows of in_proj should be unchanged (no LoRA targets them)."""
        delta = self._load_delta(
            "backbone.layers.0.mixer.in_proj.weight",
            merged_output,
            merged_index,
            orig_index,
        )
        intermediate = self.MAMBA_INTERMEDIATE
        rest_delta = delta[2 * intermediate :]
        assert rest_delta.norm() == 0, "B/C/dt rows should have zero delta"

    def test_attention_q_proj_has_delta(self, merged_output, merged_index, orig_index):
        """Attention q_proj should be merged."""
        # Layer 5 is the first attention layer (pattern char '*')
        delta = self._load_delta(
            "backbone.layers.5.mixer.q_proj.weight",
            merged_output,
            merged_index,
            orig_index,
        )
        assert delta.norm() > 0, "Attention q_proj should have non-zero delta"

    def test_lm_head_has_delta(self, merged_output, merged_index, orig_index):
        """lm_head should be merged."""
        delta = self._load_delta(
            "lm_head.weight",
            merged_output,
            merged_index,
            orig_index,
        )
        assert delta.norm() > 0, "lm_head should have non-zero delta"

    def test_shared_expert_has_delta(self, merged_output, merged_index, orig_index):
        """Shared expert weights should be merged."""
        delta = self._load_delta(
            "backbone.layers.1.mixer.shared_experts.up_proj.weight",
            merged_output,
            merged_index,
            orig_index,
        )
        assert delta.norm() > 0, "Shared expert up_proj should have non-zero delta"

    def test_routed_expert_has_delta(self, merged_output, merged_index, orig_index):
        """At least one routed expert should have a non-zero delta.

        With 1-step training, most experts aren't activated by the router.
        We check that at least one of the 128 experts per layer was updated.
        """
        # Check experts in layer 1 (first MoE layer in Nano's pattern)
        any_nonzero = False
        for exp_idx in range(128):
            key = f"backbone.layers.1.mixer.experts.{exp_idx}.up_proj.weight"
            if key not in merged_index["weight_map"]:
                continue
            delta = self._load_delta(key, merged_output, merged_index, orig_index)
            if delta.norm() > 0:
                any_nonzero = True
                break
        assert any_nonzero, "At least one routed expert should have non-zero delta"

    def test_output_has_config_and_shards(self, merged_output, merged_index):
        """Verify output directory structure."""
        assert (merged_output / "config.json").exists()
        assert (merged_output / "model.safetensors.index.json").exists()
        num_shards = len(set(merged_index["weight_map"].values()))
        assert num_shards > 0
