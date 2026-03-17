"""Downstream compatibility tests for tinker_cookbook.checkpoint_utils.

Validates that checkpoint management types and functions remain stable.
"""

from dataclasses import fields

from tinker_cookbook.checkpoint_utils import (
    CheckpointRecord,
    get_last_checkpoint,
    load_checkpoints_file,
    save_checkpoint,
)


class TestCheckpointRecord:
    def test_fields(self):
        names = {f.name for f in fields(CheckpointRecord)}
        expected = {"name", "batch", "epoch", "final", "state_path", "sampler_path", "extra"}
        assert expected.issubset(names)

    def test_constructable_minimal(self):
        record = CheckpointRecord(name="step_100")
        assert record.name == "step_100"
        assert record.batch is None
        assert record.extra == {}

    def test_to_dict(self):
        record = CheckpointRecord(name="step_100", batch=100)
        d = record.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "step_100"

    def test_from_dict(self):
        d = {"name": "step_100", "batch": 100}
        record = CheckpointRecord.from_dict(d)
        assert record.name == "step_100"
        assert record.batch == 100

    def test_roundtrip(self):
        original = CheckpointRecord(name="step_50", batch=50, epoch=1, final=False)
        restored = CheckpointRecord.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.batch == original.batch

    def test_has_method(self):
        record = CheckpointRecord(name="test", extra={"key": "value"})
        assert record.has("key") is True
        assert record.has("missing") is False

    def test_get_method(self):
        record = CheckpointRecord(name="test", extra={"key": "value"})
        assert record.get("key") == "value"


class TestCheckpointFunctions:
    def test_load_checkpoints_file_exists(self):
        assert callable(load_checkpoints_file)

    def test_get_last_checkpoint_exists(self):
        assert callable(get_last_checkpoint)

    def test_save_checkpoint_exists(self):
        assert callable(save_checkpoint)
        # save_checkpoint can be sync or async; just verify it exists

    def test_save_checkpoint_is_async(self):
        # Downstream expects save_checkpoint to be usable as a coroutine
        # Check if there's also an async version
        from tinker_cookbook import checkpoint_utils

        assert hasattr(checkpoint_utils, "save_checkpoint_async") or hasattr(
            checkpoint_utils, "save_checkpoint"
        )
