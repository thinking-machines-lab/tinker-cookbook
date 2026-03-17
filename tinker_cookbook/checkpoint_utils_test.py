"""Tests for checkpoint_utils path handling."""

import json
import tempfile
from pathlib import Path

from tinker_cookbook.checkpoint_utils import (
    CheckpointRecord,
    get_last_checkpoint,
    load_checkpoints_file,
)


def _write_checkpoints_jsonl(log_dir: str, records: list[dict]) -> None:
    path = Path(log_dir) / "checkpoints.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_load_checkpoints_file_missing_dir():
    """load_checkpoints_file returns [] when the directory doesn't exist."""
    result = load_checkpoints_file("/tmp/nonexistent_dir_abc123")
    assert result == []


def test_load_checkpoints_file_missing_file():
    """load_checkpoints_file returns [] when checkpoints.jsonl is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_checkpoints_file(tmpdir)
        assert result == []


def test_load_checkpoints_file_reads_records():
    """load_checkpoints_file reads and deserializes checkpoint records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoints_jsonl(
            tmpdir,
            [
                {"name": "000005", "batch": 5, "state_path": "tinker://state/5"},
                {"name": "000010", "batch": 10, "state_path": "tinker://state/10"},
            ],
        )
        result = load_checkpoints_file(tmpdir)
        assert len(result) == 2
        assert isinstance(result[0], CheckpointRecord)
        assert result[0].name == "000005"
        assert result[1].batch == 10


def test_get_last_checkpoint_returns_last():
    """get_last_checkpoint returns the last record with the required key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoints_jsonl(
            tmpdir,
            [
                {"name": "000005", "batch": 5, "state_path": "tinker://state/5"},
                {"name": "000010", "batch": 10, "sampler_path": "tinker://sampler/10"},
                {"name": "000015", "batch": 15, "state_path": "tinker://state/15"},
            ],
        )
        result = get_last_checkpoint(tmpdir, required_key="state_path")
        assert result is not None
        assert result.name == "000015"


def test_get_last_checkpoint_returns_none_when_empty():
    """get_last_checkpoint returns None when no checkpoints exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = get_last_checkpoint(tmpdir)
        assert result is None


def test_get_last_checkpoint_returns_none_when_key_missing():
    """get_last_checkpoint returns None when no record has the required key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoints_jsonl(
            tmpdir,
            [{"name": "000005", "batch": 5, "sampler_path": "tinker://sampler/5"}],
        )
        result = get_last_checkpoint(tmpdir, required_key="state_path")
        assert result is None


def test_load_checkpoints_file_without_batch():
    """Entries without 'batch' should deserialize without error (backward compat)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoints_jsonl(
            tmpdir,
            [
                {"name": "000000", "step": 0},
                {"name": "000010", "step": 10, "state_path": "tinker://state/10"},
            ],
        )
        result = load_checkpoints_file(tmpdir)
        assert len(result) == 2
        assert result[0].batch is None
        assert result[0].extra["step"] == 0
        assert result[1].state_path == "tinker://state/10"


def test_checkpoint_record_extra_round_trips():
    """Unknown keys land in extra and survive to_dict/from_dict round-trip."""
    record = CheckpointRecord.from_dict(
        {"name": "000005", "batch": 5, "step": 5, "custom_key": "val"}
    )
    assert record.extra == {"step": 5, "custom_key": "val"}
    d = record.to_dict()
    assert d["step"] == 5
    assert d["custom_key"] == "val"
    restored = CheckpointRecord.from_dict(d)
    assert restored.extra == {"step": 5, "custom_key": "val"}


def test_checkpoint_record_name_only():
    """A minimal entry with only 'name' should deserialize (batch None)."""
    record = CheckpointRecord.from_dict({"name": "000000"})
    assert record.name == "000000"
    assert record.batch is None


def test_checkpoint_record_get_known_field():
    """get() returns known field values, including None for unset optional fields."""
    record = CheckpointRecord(name="test", batch=5, state_path="tinker://state/5")
    assert record.get("batch") == 5
    assert record.get("state_path") == "tinker://state/5"
    # Known fields always return the attribute value, even when None.
    # This distinguishes "field exists but is unset" from "key is unknown".
    assert record.get("epoch") is None
    assert record.get("epoch", -1) is None


def test_checkpoint_record_get_extra_field():
    """get() falls through to extra for unknown keys."""
    record = CheckpointRecord(name="test", extra={"step": 10, "custom": "val"})
    assert record.get("step") == 10
    assert record.get("custom") == "val"
    assert record.get("missing") is None
    assert record.get("missing", "default") == "default"


def test_checkpoint_record_has_extra_field():
    """has() works for both known fields and extra keys."""
    record = CheckpointRecord(name="test", batch=5, extra={"step": 10})
    assert record.has("batch")
    assert not record.has("epoch")
    assert record.has("step")
    assert not record.has("missing")


def test_checkpoint_record_extra_overlap_with_known_keys():
    """Known keys in extra are dropped defensively to prevent to_dict() conflicts."""
    record = CheckpointRecord(name="test", batch=5, extra={"batch": 99, "custom": "val"})
    # "batch" should be stripped from extra; the attribute value (5) wins
    assert record.batch == 5
    assert "batch" not in record.extra
    assert record.extra == {"custom": "val"}
    # to_dict() should have batch=5, not 99
    d = record.to_dict()
    assert d["batch"] == 5
