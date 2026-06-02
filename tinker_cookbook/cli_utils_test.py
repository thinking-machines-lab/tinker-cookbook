"""Tests for cli_utils path handling."""

import tempfile
import uuid
from pathlib import Path

import fsspec
import pytest

from tinker_cookbook.cli_utils import check_log_dir


def test_check_log_dir_uri_nonexistent_is_noop(tmp_path):
    check_log_dir(f"file://{tmp_path / 'missing'}", behavior_if_exists="raise")


def test_check_log_dir_uri_resume_keeps_contents(tmp_path):
    marker = tmp_path / "keep_me.txt"
    marker.write_text("hello")

    check_log_dir(f"file://{tmp_path}", behavior_if_exists="resume")

    assert marker.exists()


def test_check_log_dir_uri_delete_removes_contents(tmp_path):
    nested = tmp_path / "subdir"
    nested.mkdir()
    (nested / "file.txt").write_text("hello")
    (tmp_path / "root.txt").write_text("hello")

    check_log_dir(f"file://{tmp_path}", behavior_if_exists="delete")

    assert list(tmp_path.iterdir()) == []


def test_check_log_dir_cloud_uri_delete_removes_prefix_contents():
    run_uri = f"memory://test-bucket/{uuid.uuid4()}/run"
    storage = fsspec.filesystem("memory")
    storage.pipe(f"{run_uri}/subdir/file.txt", b"hello")
    storage.pipe(f"{run_uri}/root.txt", b"hello")
    storage.pipe(f"{run_uri}-sibling/file.txt", b"keep")

    check_log_dir(run_uri, behavior_if_exists="delete")

    assert storage.find(run_uri) == []
    sibling_files = storage.find(f"{run_uri}-sibling")
    assert len(sibling_files) == 1
    assert sibling_files[0].endswith("/run-sibling/file.txt")


def test_check_log_dir_uri_raise_raises(tmp_path):
    (tmp_path / "file.txt").write_text("hello")

    with pytest.raises(ValueError, match="already exists"):
        check_log_dir(f"file://{tmp_path}", behavior_if_exists="raise")


def test_check_log_dir_uri_ask_resume_keeps_contents(tmp_path, monkeypatch):
    marker = tmp_path / "keep_me.txt"
    marker.write_text("hello")
    monkeypatch.setattr("builtins.input", lambda _: "resume")

    check_log_dir(f"file://{tmp_path}", behavior_if_exists="ask")

    assert marker.exists()


def test_check_log_dir_uri_ask_delete_removes_contents(tmp_path, monkeypatch):
    marker = tmp_path / "delete_me.txt"
    marker.write_text("hello")
    monkeypatch.setattr("builtins.input", lambda _: "delete")

    check_log_dir(f"file://{tmp_path}", behavior_if_exists="ask")

    assert list(tmp_path.iterdir()) == []


def test_check_log_dir_uri_invalid_behavior_raises(tmp_path):
    (tmp_path / "file.txt").write_text("hello")

    with pytest.raises(AssertionError, match="Invalid behavior_if_exists"):
        check_log_dir(f"file://{tmp_path}", behavior_if_exists="invalid")  # type: ignore[arg-type]


def test_check_log_dir_nonexistent_is_noop():
    """check_log_dir does nothing when the directory doesn't exist."""
    missing = Path("/tmp/nonexistent_dir_abc123")
    check_log_dir(str(missing), "raise")
    assert not missing.exists()


def test_check_log_dir_resume_keeps_directory():
    """check_log_dir with 'resume' leaves the directory intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / "keep_me.txt"
        marker.write_text("hello")
        check_log_dir(tmpdir, "resume")
        assert marker.exists()


def test_check_log_dir_delete_removes_directory():
    """check_log_dir with 'delete' removes directory contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "subdir"
        target.mkdir()
        (target / "file.txt").write_text("hello")
        check_log_dir(str(target), "delete")
        assert target.exists()
        assert list(target.iterdir()) == []


def test_check_log_dir_raise_raises():
    """check_log_dir with 'raise' raises ValueError when directory exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="already exists"):
            check_log_dir(tmpdir, "raise")


def test_check_log_dir_invalid_behavior_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError, match="Invalid behavior_if_exists"):
            check_log_dir(tmpdir, "invalid")  # type: ignore[arg-type]
