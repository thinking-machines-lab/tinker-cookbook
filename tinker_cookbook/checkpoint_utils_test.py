"""Tests for checkpoint_utils path handling."""

import json
import tempfile
from pathlib import Path

from tinker_cookbook.checkpoint_utils import (
    CheckpointRecord,
    load_checkpoints_file,
    get_last_checkpoint,
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
