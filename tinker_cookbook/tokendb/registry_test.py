"""Tests for the run registry (per-run JSON records + liveness probes)."""

import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from tinker_cookbook.tokendb.registry import (
    DEFAULT_REGISTRY_DIR,
    list_runs,
    load_run_record,
    register_run,
    resolve_registry_dir,
    run_status,
)
from tinker_cookbook.tokendb.schema import TokenRow
from tinker_cookbook.tokendb.writer import TokenDbWriter


def make_row(**overrides) -> TokenRow:
    defaults: dict = {
        "split": "train",
        "iteration": 0,
        "group_idx": 0,
        "traj_idx": 0,
        "step_idx": 0,
        "ob_tokens": [1, 2, 3],
        "ac_tokens": [4, 5],
    }
    defaults.update(overrides)
    return TokenRow(**defaults)


def registry_dir() -> Path:
    resolved = resolve_registry_dir()
    assert resolved is not None
    return Path(resolved)


def test_resolve_registry_dir_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Explicit arg wins over the env var.
    assert resolve_registry_dir(str(tmp_path)) == str(tmp_path)
    # Env var (set by the conftest fixture) wins over the default.
    assert resolve_registry_dir() == os.environ["TINKER_TOKENDB_REGISTRY"]
    # Empty string disables at either level.
    assert resolve_registry_dir("") is None
    monkeypatch.setenv("TINKER_TOKENDB_REGISTRY", "")
    assert resolve_registry_dir() is None
    # No env var: the default location, tilde-expanded.
    monkeypatch.delenv("TINKER_TOKENDB_REGISTRY")
    assert resolve_registry_dir() == str(Path(DEFAULT_REGISTRY_DIR).expanduser())
    # Cloud URIs pass through unchanged (no tilde/Path mangling).
    assert resolve_registry_dir("gs://bucket/prefix/runs") == "gs://bucket/prefix/runs"


def test_writer_registers_run(tmp_path: Path):
    log_path = tmp_path / "run"
    with TokenDbWriter(
        log_path, context={"model_name": "test-model", "recipe_name": "test-recipe"}
    ) as writer:
        run_id = writer.run_id
    records = list(registry_dir().glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["run_id"] == run_id
    assert record["run_attempt"] == 1
    assert record["log_path"] == str(log_path.resolve())
    assert record["model_name"] == "test-model"
    assert record["recipe_name"] == "test-recipe"
    assert record["pid"] == os.getpid()
    assert record["hostname"]
    assert record["writer_id"] == writer.writer_id
    assert record["started_at"]

    # A resume overwrites the same record (keyed by run_id) with the new attempt.
    with TokenDbWriter(log_path) as writer2:
        assert writer2.run_id == run_id
    records = list(registry_dir().glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["run_attempt"] == 2

    loaded = load_run_record(None, run_id)
    assert loaded is not None and loaded["run_id"] == run_id
    assert load_run_record(None, "nope") is None


def test_worker_context_does_not_register(tmp_path: Path):
    with TokenDbWriter(tmp_path / "run", context={"run_id": "abc", "run_attempt": 1}):
        pass
    assert not registry_dir().exists() or not list(registry_dir().glob("*.json"))


def test_registry_dir_empty_disables(tmp_path: Path):
    with TokenDbWriter(tmp_path / "run", registry_dir=""):
        pass
    assert not registry_dir().exists() or not list(registry_dir().glob("*.json"))


def test_env_var_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    other = tmp_path / "other-registry"
    monkeypatch.setenv("TINKER_TOKENDB_REGISTRY", str(other))
    with TokenDbWriter(tmp_path / "run"):
        pass
    assert len(list(other.glob("*.json"))) == 1

    monkeypatch.setenv("TINKER_TOKENDB_REGISTRY", "")
    with TokenDbWriter(tmp_path / "run2"):
        pass
    assert len(list(other.glob("*.json"))) == 1  # nothing new anywhere


def test_register_run_best_effort(tmp_path: Path):
    # Registry dir path collides with an existing file: mkdir fails, but
    # register_run must swallow it (training must not break).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    assert (
        register_run(
            log_path=str(tmp_path / "run"),
            run_id="r1",
            run_attempt=1,
            registry_dir=str(blocker),
        )
        is None
    )


def test_run_status_live_and_stale(tmp_path: Path):
    log_path = tmp_path / "run"
    with TokenDbWriter(log_path) as writer:
        writer.append_rows([make_row(iteration=3), make_row(iteration=7, step_idx=1)])

    status = run_status(str(log_path))
    assert status["live"] is True
    assert status["n_segments"] >= 1
    assert status["latest_iteration"] == 7
    assert status["last_activity_ts"] == pytest.approx(time.time(), abs=30)

    # Age the manifests past the live window.
    old = time.time() - 3600
    for manifest in (log_path / "tokens").glob("manifest-*.jsonl"):
        os.utime(manifest, (old, old))
    stale = run_status(str(log_path))
    assert stale["live"] is False
    assert stale["latest_iteration"] == 7

    # But a wider window counts it as live again.
    assert run_status(str(log_path), live_window_s=7200)["live"] is True


def test_run_status_missing_store(tmp_path: Path):
    status = run_status(str(tmp_path / "nope"))
    assert status == {
        "live": False,
        "last_activity_ts": None,
        "n_segments": 0,
        "latest_iteration": None,
    }


def test_list_runs(tmp_path: Path):
    with TokenDbWriter(tmp_path / "run-a", context={"model_name": "model-a"}) as writer:
        writer.append_rows([make_row(iteration=1)])
        run_a = writer.run_id
    with TokenDbWriter(tmp_path / "run-b", context={"model_name": "model-b"}) as writer:
        writer.append_rows([make_row(iteration=5)])
        run_b = writer.run_id
    # Age run-a's manifests so it reads as stale.
    old = time.time() - 3600
    for manifest in (tmp_path / "run-a" / "tokens").glob("manifest-*.jsonl"):
        os.utime(manifest, (old, old))

    runs = {r["run_id"]: r for r in list_runs()}
    assert set(runs) == {run_a, run_b}
    assert runs[run_a]["status"]["live"] is False
    assert runs[run_b]["status"]["live"] is True
    assert runs[run_a]["status"]["latest_iteration"] == 1
    assert runs[run_b]["status"]["latest_iteration"] == 5
    assert runs[run_a]["model_name"] == "model-a"

    # Malformed records are skipped, not fatal.
    (registry_dir() / "broken.json").write_text("{nope")
    assert {r["run_id"] for r in list_runs()} == {run_a, run_b}


def test_list_runs_disabled_or_missing(monkeypatch: pytest.MonkeyPatch):
    assert list_runs("") == []
    monkeypatch.setenv("TINKER_TOKENDB_REGISTRY", "/nonexistent/definitely/not/here")
    assert list_runs() == []


def test_registry_on_non_local_storage(tmp_path: Path):
    """Registry I/O goes through Storage: a cloud-shaped (fsspec memory://)
    registry supports the full register + load + list + status roundtrip."""
    fsspec = pytest.importorskip("fsspec")
    registry_uri = f"memory://tokendb-registry-{tmp_path.name}"

    # A real (local) store so run_status has manifests to probe.
    log_path = tmp_path / "run"
    with TokenDbWriter(log_path, registry_dir="") as writer:
        writer.append_rows([make_row(iteration=4)])

    assert (
        register_run(
            log_path=str(log_path),
            run_id="mem-run",
            run_attempt=2,
            model_name="mem-model",
            registry_dir=registry_uri,
        )
        is not None
    )
    try:
        loaded = load_run_record(registry_uri, "mem-run")
        assert loaded is not None
        assert loaded["model_name"] == "mem-model"
        assert loaded["log_path"] == str(log_path.resolve())

        runs = list_runs(registry_uri)
        assert [r["run_id"] for r in runs] == ["mem-run"]
        assert runs[0]["run_attempt"] == 2
        assert runs[0]["status"]["live"] is True
        assert runs[0]["status"]["latest_iteration"] == 4

        # Re-registering overwrites the same record.
        register_run(
            log_path=str(log_path), run_id="mem-run", run_attempt=3, registry_dir=registry_uri
        )
        assert len(list_runs(registry_uri)) == 1
        record = load_run_record(registry_uri, "mem-run")
        assert record is not None and record["run_attempt"] == 3
    finally:
        # The fsspec memory filesystem is process-global; clean up.
        fs = fsspec.filesystem("memory")
        fs.rm(f"/tokendb-registry-{tmp_path.name}", recursive=True)
