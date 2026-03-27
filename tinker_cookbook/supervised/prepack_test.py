"""Tests for the pre-packing pipeline (prepack.py): dataset, serialization, CLI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import tinker

from tinker_cookbook.supervised.packing import pack_to_datums
from tinker_cookbook.supervised.prepack import (
    PrepackConfig,
    PrepackedDatasetBuilder,
    PrepackedSupervisedDataset,
    load_packed_datums,
    prepack,
    render_parallel,
    save_packed_datums,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(token_ids: list[int], weight_val: float = 1.0) -> tuple[list[int], list[float]]:
    """Create a (tokens, weights) pair for testing."""
    return (token_ids, [weight_val] * len(token_ids))


def _make_jsonl(conversations: list[list[dict]], path: Path) -> None:
    """Write conversations to a JSONL file."""
    with open(path, "w") as f:
        for msgs in conversations:
            f.write(json.dumps({"messages": msgs}) + "\n")


def _simple_conversations(n: int) -> list[list[dict]]:
    """Create n simple user/assistant conversations."""
    return [
        [
            {"role": "user", "content": f"Question {i}"},
            {"role": "assistant", "content": f"Answer {i}"},
        ]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip(self):
        """Datums survive a save/load cycle."""
        items = [
            _make_item([10, 20, 30, 40]),
            _make_item([50, 60, 70, 80]),
        ]
        datums = pack_to_datums(items, max_length=100)
        assert len(datums) > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {"source_file": "test.jsonl", "max_packed_length": 100}
            save_packed_datums(datums, tmpdir, metadata=metadata, shard_size=1)

            loaded_datums, loaded_metadata = load_packed_datums(tmpdir)

            assert len(loaded_datums) == len(datums)
            assert loaded_metadata["source_file"] == "test.jsonl"
            assert loaded_metadata["num_datums"] == len(datums)
            assert loaded_metadata["num_shards"] >= 1

            # Verify content matches.
            for orig, loaded in zip(datums, loaded_datums, strict=True):
                assert list(orig.model_input.to_ints()) == list(loaded.model_input.to_ints())

    def test_roundtrip_empty(self):
        """Empty dataset roundtrips correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_packed_datums([], tmpdir, metadata={"empty": True})
            loaded, meta = load_packed_datums(tmpdir)
            assert loaded == []
            assert meta["num_datums"] == 0

    def test_missing_metadata_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="metadata"):
                load_packed_datums(tmpdir)

    def test_multiple_shards(self):
        """Many Datums are spread across multiple shard files."""
        items = [_make_item(list(range(i, i + 5))) for i in range(20)]
        datums = pack_to_datums(items, max_length=12)
        assert len(datums) > 3, f"Expected >3 datums but got {len(datums)}"

        with tempfile.TemporaryDirectory() as tmpdir:
            save_packed_datums(datums, tmpdir, metadata={}, shard_size=3)
            shard_files = list(Path(tmpdir).glob("shard_*.jsonl"))
            assert len(shard_files) > 1

            loaded, meta = load_packed_datums(tmpdir)
            assert len(loaded) == len(datums)


# ---------------------------------------------------------------------------
# PrepackedSupervisedDataset
# ---------------------------------------------------------------------------


class TestPrepackedSupervisedDataset:
    def test_len_exact(self):
        """__len__ returns num_datums // batch_size."""
        items = [_make_item([i, i + 1]) for i in range(10)]
        datums = pack_to_datums(items, max_length=5)
        # Each item is 2 tokens. At max 5, we get 2 items per pack (4 tokens).
        # 10 items -> 5 packs.
        assert len(datums) == 5

        with tempfile.TemporaryDirectory() as tmpdir:
            save_packed_datums(datums, tmpdir, metadata={})
            ds = PrepackedSupervisedDataset(tmpdir, batch_size=2)
            assert len(ds) == 2  # 5 // 2 = 2

    def test_get_batch(self):
        items = [_make_item([i, i + 1]) for i in range(10)]
        datums = pack_to_datums(items, max_length=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_packed_datums(datums, tmpdir, metadata={})
            ds = PrepackedSupervisedDataset(tmpdir, batch_size=2)
            batch = ds.get_batch(0)
            assert len(batch) == 2
            assert all(isinstance(d, tinker.Datum) for d in batch)

    def test_get_batch_random_access(self):
        items = [_make_item([i, i + 1]) for i in range(10)]
        datums = pack_to_datums(items, max_length=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_packed_datums(datums, tmpdir, metadata={})
            ds = PrepackedSupervisedDataset(tmpdir, batch_size=2)
            # Access batches in any order.
            b1 = ds.get_batch(1)
            b0 = ds.get_batch(0)
            assert len(b0) == 2
            assert len(b1) == 2
            # They should be different batches.
            assert list(b0[0].model_input.to_ints()) != list(b1[0].model_input.to_ints())

    def test_set_epoch_shuffles_deterministically(self):
        items = [_make_item([i * 10, i * 10 + 1]) for i in range(20)]
        datums = pack_to_datums(items, max_length=5)
        assert len(datums) >= 4

        with tempfile.TemporaryDirectory() as tmpdir:
            save_packed_datums(datums, tmpdir, metadata={})
            ds = PrepackedSupervisedDataset(tmpdir, batch_size=2)

            # Epoch 0
            ds.set_epoch(seed=0)
            batch_e0 = ds.get_batch(0)

            # Epoch 1 should differ
            ds.set_epoch(seed=1)
            batch_e1 = ds.get_batch(0)

            # Same seed should produce same order
            ds.set_epoch(seed=0)
            batch_e0_again = ds.get_batch(0)

            e0_tokens = [list(d.model_input.to_ints()) for d in batch_e0]
            e1_tokens = [list(d.model_input.to_ints()) for d in batch_e1]
            e0_again_tokens = [list(d.model_input.to_ints()) for d in batch_e0_again]

            assert e0_tokens == e0_again_tokens, "Same seed should produce same order"
            # With enough datums, different seeds should produce different orders
            # (extremely unlikely to be the same by chance).
            assert e0_tokens != e1_tokens, "Different seeds should produce different orders"

    def test_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_packed_datums([], tmpdir, metadata={})
            ds = PrepackedSupervisedDataset(tmpdir, batch_size=4)
            assert len(ds) == 0


# ---------------------------------------------------------------------------
# PrepackedDatasetBuilder
# ---------------------------------------------------------------------------


class TestPrepackedDatasetBuilder:
    def test_builder_returns_dataset(self):
        items = [_make_item([i, i + 1]) for i in range(10)]
        datums = pack_to_datums(items, max_length=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_packed_datums(datums, tmpdir, metadata={})
            builder = PrepackedDatasetBuilder(packed_dir=tmpdir, batch_size=2)
            train_ds, eval_ds = builder()
            assert isinstance(train_ds, PrepackedSupervisedDataset)
            assert eval_ds is None
            assert len(train_ds) == 2


# ---------------------------------------------------------------------------
# Parallel rendering (integration test with mock renderer)
# ---------------------------------------------------------------------------


class TestParallelRendering:
    @patch("tinker_cookbook.supervised.prepack._worker_init")
    @patch("tinker_cookbook.supervised.prepack._render_one")
    def test_single_worker_rendering(self, mock_render, mock_init):
        """Single-worker mode processes all conversations."""
        mock_render.return_value = ([1, 2, 3], [1.0, 1.0, 1.0])

        conversations = _simple_conversations(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.jsonl"
            _make_jsonl(conversations, source)

            config = PrepackConfig(
                source_file=str(source),
                output_dir=str(Path(tmpdir) / "packed"),
                model_name="test-model",
                renderer_name="test-renderer",
                max_packed_length=100,
                num_workers=1,
            )

            prepack(config)

            # Should have called render for each conversation
            assert mock_render.call_count == 5


# ---------------------------------------------------------------------------
# Custom render_fn
# ---------------------------------------------------------------------------


class TestCustomRenderFn:
    def test_prepack_with_custom_render_fn(self):
        """prepack() accepts a custom render_fn instead of using the renderer."""
        conversations = _simple_conversations(5)

        def custom_render(convos: list[list[dict]]) -> list[tuple[list[int], list[float]]]:
            return [([1, 2, 3], [1.0, 1.0, 1.0]) for _ in convos]

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.jsonl"
            _make_jsonl(conversations, source)

            config = PrepackConfig(
                source_file=str(source),
                output_dir=str(Path(tmpdir) / "packed"),
                model_name="unused",
                renderer_name="unused",
                max_packed_length=100,
                num_workers=1,
            )

            prepack(config, render_fn=custom_render)

            datums, metadata = load_packed_datums(Path(tmpdir) / "packed")
            assert len(datums) > 0
            assert metadata["num_raw_conversations"] == 5


# ---------------------------------------------------------------------------
# End-to-end CLI test (uses a real renderer)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture
    def sample_data_dir(self, tmp_path: Path) -> Path:
        """Create a temp directory with a small JSONL file."""
        conversations = _simple_conversations(10)
        source = tmp_path / "data.jsonl"
        _make_jsonl(conversations, source)
        return tmp_path

    def test_end_to_end_with_role_colon(self, sample_data_dir: Path):
        """Full pipeline: load -> render -> pack -> save -> load."""
        output_dir = sample_data_dir / "packed"
        source_file = sample_data_dir / "data.jsonl"

        config = PrepackConfig(
            source_file=str(source_file),
            output_dir=str(output_dir),
            model_name="Qwen/Qwen3-0.6B",
            renderer_name="role_colon",
            # Small max_packed_length to force multiple packed sequences.
            max_packed_length=40,
            num_workers=2,
            train_on_what="all_assistant_messages",
            shard_size=5,
            imap_chunksize=2,
        )

        prepack(config)

        # Verify output structure.
        assert (output_dir / "metadata.json").exists()
        shard_files = list(output_dir.glob("shard_*.jsonl"))
        assert len(shard_files) > 0

        # Load and verify.
        datums, metadata = load_packed_datums(output_dir)
        assert len(datums) > 0
        assert metadata["num_raw_conversations"] == 10
        assert metadata["renderer_name"] == "role_colon"

        # Use as a dataset.
        batch_size = min(2, len(datums))
        ds = PrepackedSupervisedDataset(output_dir, batch_size=batch_size)
        assert len(ds) >= 1
        batch = ds.get_batch(0)
        assert len(batch) == batch_size
        for datum in batch:
            assert isinstance(datum, tinker.Datum)
            assert datum.model_input.length > 0
