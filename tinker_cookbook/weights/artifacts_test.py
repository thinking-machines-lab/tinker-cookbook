"""Unit tests for model artifact utilities.

Uses temporary directories and synthetic safetensors files — no network or
GPU required.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tinker_cookbook.weights._artifacts import (
    ShardWriter,
    copy_model_code_files,
    get_model_state_keys,
    get_model_state_shapes,
    get_shard_files,
    load_adapter_weights,
)

# ---------------------------------------------------------------------------
# get_model_state_keys
# ---------------------------------------------------------------------------


class TestGetModelStateKeys:
    def test_reads_keys_from_single_shard(self, tmp_path: Path):
        tensors = {"layer.0.weight": torch.zeros(4, 4), "layer.0.bias": torch.zeros(4)}
        save_file(tensors, str(tmp_path / "model.safetensors"))

        keys = get_model_state_keys(tmp_path)
        assert keys == {"layer.0.weight", "layer.0.bias"}

    def test_reads_keys_from_multiple_shards(self, tmp_path: Path):
        save_file({"a.weight": torch.zeros(2)}, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file({"b.weight": torch.zeros(3)}, str(tmp_path / "model-00002-of-00002.safetensors"))

        keys = get_model_state_keys(tmp_path)
        assert keys == {"a.weight", "b.weight"}

    def test_raises_if_no_safetensors(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match=r"No \.safetensors files"):
            get_model_state_keys(tmp_path)


# ---------------------------------------------------------------------------
# get_model_state_shapes
# ---------------------------------------------------------------------------


class TestGetModelStateShapes:
    def test_reads_shapes(self, tmp_path: Path):
        tensors = {"weight": torch.zeros(8, 4), "bias": torch.zeros(8)}
        save_file(tensors, str(tmp_path / "model.safetensors"))

        shapes = get_model_state_shapes(tmp_path)
        assert shapes == {"weight": (8, 4), "bias": (8,)}

    def test_reads_shapes_across_shards(self, tmp_path: Path):
        save_file({"a": torch.zeros(2, 3)}, str(tmp_path / "shard-1.safetensors"))
        save_file({"b": torch.zeros(4)}, str(tmp_path / "shard-2.safetensors"))

        shapes = get_model_state_shapes(tmp_path)
        assert shapes == {"a": (2, 3), "b": (4,)}

    def test_helpful_error_for_bin_files(self, tmp_path: Path):
        (tmp_path / "pytorch_model.bin").write_bytes(b"fake")
        with pytest.raises(FileNotFoundError, match=r"\.bin file.*merge_strategy='full'"):
            get_model_state_shapes(tmp_path)

    def test_helpful_error_for_empty_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match=r"merge_strategy='full'"):
            get_model_state_shapes(tmp_path)


# ---------------------------------------------------------------------------
# get_shard_files
# ---------------------------------------------------------------------------


class TestGetShardFiles:
    def test_reads_from_index_json(self, tmp_path: Path):
        index = {
            "weight_map": {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00002-of-00002.safetensors",
                "c.weight": "model-00001-of-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        files = get_shard_files(tmp_path)
        assert files == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

    def test_falls_back_to_glob(self, tmp_path: Path):
        save_file({"a": torch.zeros(1)}, str(tmp_path / "model.safetensors"))

        files = get_shard_files(tmp_path)
        assert files == ["model.safetensors"]

    def test_raises_if_no_files(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            get_shard_files(tmp_path)


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------


class TestShardWriter:
    def test_single_shard_named_without_index(self, tmp_path: Path):
        writer = ShardWriter(tmp_path)
        writer.add_tensor("a.weight", torch.zeros(4))
        writer.add_tensor("b.weight", torch.ones(4))
        weight_map = writer.finalize()

        assert (tmp_path / "model.safetensors").exists()
        assert weight_map == {
            "a.weight": "model.safetensors",
            "b.weight": "model.safetensors",
        }

    def test_multiple_shards_when_exceeding_max_size(self, tmp_path: Path):
        # Each float32 tensor of 1024 elements = 4096 bytes
        writer = ShardWriter(tmp_path, max_shard_size=4096)
        writer.add_tensor("a.weight", torch.zeros(1024))  # 4096 bytes, fits
        writer.add_tensor("b.weight", torch.zeros(1024))  # triggers flush of a, then b pending

        weight_map = writer.finalize()
        assert len(set(weight_map.values())) == 2
        assert "model-00001-of-00002.safetensors" in weight_map.values()
        assert "model-00002-of-00002.safetensors" in weight_map.values()

    def test_total_size_tracks_bytes(self, tmp_path: Path):
        writer = ShardWriter(tmp_path)
        writer.add_tensor("x", torch.zeros(100, dtype=torch.float32))  # 400 bytes
        assert writer.total_size == 400

    def test_temp_files_cleaned_up(self, tmp_path: Path):
        writer = ShardWriter(tmp_path)
        writer.add_tensor("a", torch.zeros(4))
        writer.finalize()

        # No .tmp files should remain
        tmp_files = list(tmp_path.glob("*.tmp.*"))
        assert tmp_files == []

    def test_empty_writer_produces_no_files(self, tmp_path: Path):
        writer = ShardWriter(tmp_path)
        weight_map = writer.finalize()
        assert weight_map == {}
        assert list(tmp_path.glob("*.safetensors")) == []


# ---------------------------------------------------------------------------
# load_adapter_weights
# ---------------------------------------------------------------------------


class TestLoadAdapterWeights:
    def test_loads_weights_and_config(self, tmp_path: Path):
        weights = {"lora_A": torch.ones(2, 4), "lora_B": torch.ones(8, 2)}
        save_file(weights, str(tmp_path / "adapter_model.safetensors"))
        (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": 2}))

        loaded_weights, config = load_adapter_weights(tmp_path)
        assert set(loaded_weights.keys()) == {"lora_A", "lora_B"}
        assert config["r"] == 2

    def test_missing_safetensors_raises(self, tmp_path: Path):
        (tmp_path / "adapter_config.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match=r"adapter_model\.safetensors"):
            load_adapter_weights(tmp_path)

    def test_missing_config_raises(self, tmp_path: Path):
        save_file({"x": torch.zeros(1)}, str(tmp_path / "adapter_model.safetensors"))
        with pytest.raises(FileNotFoundError, match=r"adapter_config\.json"):
            load_adapter_weights(tmp_path)


# ---------------------------------------------------------------------------
# copy_model_code_files
# ---------------------------------------------------------------------------


class TestCopyModelCodeFiles:
    def test_copies_only_py_files(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()

        (src / "modeling_custom.py").write_text("# model code")
        (src / "configuration_custom.py").write_text("# config code")
        # Non-py files should NOT be copied
        (src / "config.json").write_text('{"model_type": "test"}')
        (src / "tokenizer.model").write_text("tokenizer data")
        save_file({"x": torch.zeros(1)}, str(src / "model.safetensors"))

        copy_model_code_files(src, dst)

        assert (dst / "modeling_custom.py").exists()
        assert (dst / "configuration_custom.py").exists()
        assert not (dst / "config.json").exists()
        assert not (dst / "tokenizer.model").exists()
        assert not (dst / "model.safetensors").exists()

    def test_does_not_overwrite_existing(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()

        (src / "modeling.py").write_text("source")
        (dst / "modeling.py").write_text("existing")

        copy_model_code_files(src, dst)

        assert (dst / "modeling.py").read_text() == "existing"
