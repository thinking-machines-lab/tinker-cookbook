"""Tests for cli_utils path handling."""

import tempfile
from pathlib import Path

import pytest

from tinker_cookbook.cli_utils import RUNS_DIR_ENV_VAR, check_log_dir, runs_path


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


def test_runs_path_defaults_to_home(monkeypatch: pytest.MonkeyPatch):
    """runs_path falls back to ~/tinker-runs, expanded."""
    monkeypatch.delenv(RUNS_DIR_ENV_VAR, raising=False)
    assert runs_path("math_rl", "run1") == str(Path.home() / "tinker-runs" / "math_rl" / "run1")


def test_runs_path_uses_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """runs_path honors TINKER_COOKBOOK_RUNS_DIR."""
    monkeypatch.setenv(RUNS_DIR_ENV_VAR, str(tmp_path))
    assert runs_path("sl_basic") == str(tmp_path / "sl_basic")
