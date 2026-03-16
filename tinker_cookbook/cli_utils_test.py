"""Tests for cli_utils path handling."""

import tempfile
from pathlib import Path

import pytest

from tinker_cookbook.cli_utils import check_log_dir


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
