"""Tests for cli_utils path handling."""

import tempfile
import uuid
from pathlib import Path

import pytest

from tinker_cookbook.cli_utils import check_log_dir
from tinker_cookbook.stores.storage import storage_from_uri


def test_check_log_dir_nonexistent_is_noop():
    """check_log_dir does nothing when the directory doesn't exist."""
    check_log_dir("/tmp/nonexistent_dir_abc123", "raise")


def test_check_log_dir_resume_keeps_directory():
    """check_log_dir with 'resume' leaves the directory intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / "keep_me.txt"
        marker.write_text("hello")
        check_log_dir(tmpdir, "resume")
        assert marker.exists()


def test_check_log_dir_delete_removes_directory():
    """check_log_dir with 'delete' removes the directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "subdir"
        target.mkdir()
        (target / "file.txt").write_text("hello")
        check_log_dir(str(target), "delete")
        assert not target.exists()


def test_check_log_dir_raise_raises():
    """check_log_dir with 'raise' raises ValueError when directory exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="already exists"):
            check_log_dir(tmpdir, "raise")


def test_check_log_dir_cloud_nonexistent_is_noop():
    """A cloud prefix with no objects is treated as nonexistent (no raise)."""
    base = f"memory://bucket/{uuid.uuid4()}"
    check_log_dir(f"{base}/run", "raise")


def test_check_log_dir_cloud_raise_when_objects_exist():
    """A cloud prefix with objects (no directory marker) is detected via exists_tree."""
    base = f"memory://bucket/{uuid.uuid4()}"
    storage_from_uri(f"{base}/run").write("metrics.jsonl", b"{}\n")
    with pytest.raises(ValueError, match="already exists"):
        check_log_dir(f"{base}/run", "raise")


def test_check_log_dir_cloud_delete_prefix_safety():
    """'delete' clears the run prefix but leaves a sibling that shares its name."""
    base = f"memory://bucket/{uuid.uuid4()}"
    run = storage_from_uri(f"{base}/run")
    sibling = storage_from_uri(f"{base}/run-sibling")
    run.write("subdir/file.txt", b"x")
    sibling.write("g.txt", b"keep")

    check_log_dir(f"{base}/run", "delete")

    assert not run.exists_tree("")
    assert sibling.read("g.txt") == b"keep"
