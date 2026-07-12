"""Tests for the TokenStoreBackend seam and storage flush semantics.

Covers: the default parquet backend satisfying the (enlarged) protocol,
``TokenDB`` over a minimal custom backend, and the "appends must be followed
by ``Storage.flush()``" contract that keeps staged cloud backends
(``FsspecStorage``) from silently holding back manifests, labels, and chat
transcripts.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("pyarrow")

from tinker_cookbook.stores.storage import LocalStorage, Storage
from tinker_cookbook.tokendb.agent import ChatStore, VisualStore
from tinker_cookbook.tokendb.interface import TokenStoreBackend, TokenWriter
from tinker_cookbook.tokendb.reader import ParquetSegmentReader, TokenDB
from tinker_cookbook.tokendb.schema import TokenRow
from tinker_cookbook.tokendb.writer import ParquetSegmentBackend, TokenDbWriter


def make_row(**overrides: Any) -> TokenRow:
    defaults: dict[str, Any] = {
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


# --- Protocol conformance ---


def test_parquet_backend_satisfies_protocol(tmp_path: Path):
    # The typed assignment is the real check: pyright verifies that
    # ParquetSegmentBackend structurally satisfies the enlarged protocol.
    backend: TokenStoreBackend = ParquetSegmentBackend(tmp_path)
    assert isinstance(backend, TokenStoreBackend)


def test_parquet_backend_structured_methods_roundtrip(tmp_path: Path):
    """The promoted methods work end-to-end through the protocol seam."""
    with TokenDbWriter(tmp_path, registry_dir="") as writer:
        writer.append_rows(
            [
                make_row(iteration=1, traj_idx=0, total_reward=1.0),
                make_row(iteration=1, traj_idx=0, step_idx=1, total_reward=1.0),
                make_row(iteration=1, traj_idx=1, total_reward=0.0),
            ]
        )
    backend: TokenStoreBackend = ParquetSegmentBackend(tmp_path)
    assert backend.refresh() != []
    trajs = backend.trajectories(split="train")
    assert {(t["traj_idx"], t["n_steps"]) for t in trajs} == {(0, 2), (1, 1)}
    stats = backend.dashboard_stats(recent_k=5, series_len=50)
    assert stats["n_rows"] == 3
    assert stats["latest_iteration"] == 1
    assert stats["mean_recent_reward"] == pytest.approx(0.5)
    assert backend.group_traj_idxs("train", 1, 0) == [0, 1]
    assert backend.runs() and backend.schema_card() is not None


# --- TokenDB over a custom (non-parquet) backend ---


class FakeWriter:
    def __init__(self) -> None:
        self.rows: list[TokenRow] = []

    def append_rows(self, rows: Sequence[TokenRow]) -> None:
        self.rows.extend(rows)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeBackend:
    """Minimal in-memory TokenStoreBackend (no SQL escape hatch)."""

    def __init__(self) -> None:
        self.labels_added: list[tuple[Any, ...]] = []

    def open_writer(self, run_context: Mapping[str, Any]) -> TokenWriter:
        return FakeWriter()

    def refresh(self) -> list[str]:
        return []

    def query(self, **filters: Any) -> list[dict[str, Any]]:
        return [{"split": "train", "iteration": 0}]

    def trajectories(
        self, *, latest_only: bool = False, limit: int = 500, offset: int = 0, **filters: Any
    ) -> list[dict[str, Any]]:
        return [{"split": "train", "iteration": 0, "n_steps": 1}]

    def get_rollout(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        run_attempt: int | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [{"split": split, "iteration": iteration, "step_idx": 0}]

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def search_hit_counts(self, **kwargs: Any) -> dict[int, int]:
        return {}

    def runs(self) -> list[dict[str, Any]]:
        return [{"run_id": "fake", "run_attempt": 1}]

    def schema_card(self) -> dict[str, Any]:
        return {"metrics_keys": [], "keys_truncated": False}

    def dashboard_stats(self, *, recent_k: int = 5, series_len: int = 50) -> dict[str, Any]:
        return {
            "n_rows": 1,
            "n_filtered_rows": 0,
            "latest_iteration": 0,
            "mean_recent_reward": None,
            "reward_series": [],
        }

    def group_traj_idxs(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        run_attempt: int | None = None,
        run_id: str | None = None,
    ) -> list[int]:
        return [0]

    async def subscribe(self, **filters: Any) -> AsyncIterator[dict[str, Any]]:
        return
        yield {}  # pragma: no cover - makes this an async generator

    def add_label(
        self,
        key: Mapping[str, Any],
        label_key: str,
        label_value: Any,
        *,
        author: str,
        note: str | None = None,
    ) -> None:
        self.labels_added.append((dict(key), label_key, label_value, author, note))

    def labels(self, **filters: Any) -> list[dict[str, Any]]:
        return []


def test_tokendb_accepts_custom_backend():
    fake = FakeBackend()
    db = TokenDB(fake)
    assert db.query() == [{"split": "train", "iteration": 0}]
    assert db.trajectories()[0]["n_steps"] == 1
    assert db.get_rollout("train", 0, 0, 0)[0]["step_idx"] == 0
    assert db.runs()[0]["run_id"] == "fake"
    assert db.dashboard_stats()["n_rows"] == 1
    assert db.group_traj_idxs("train", 0, 0) == [0]
    db.add_label({"split": "train"}, "quality", 1, author="tester")
    assert fake.labels_added == [({"split": "train"}, "quality", 1, "tester", None)]
    # sql() is backend-specific and not part of the protocol.
    with pytest.raises(NotImplementedError):
        db.sql("SELECT 1")


def test_tokendb_still_accepts_paths_and_storage(tmp_path: Path):
    with TokenDbWriter(tmp_path, registry_dir="") as writer:
        writer.append_rows([make_row()])
    assert len(TokenDB(tmp_path).query()) == 1
    assert len(TokenDB(str(tmp_path)).query()) == 1
    assert len(TokenDB(LocalStorage(tmp_path)).query()) == 1


# --- Storage flush semantics ---


class RecordingStorage:
    """Wraps a Storage and counts flush() calls (delegates everything else)."""

    def __init__(self, inner: Storage) -> None:
        self._inner = inner
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1
        self._inner.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_writer_flush_flushes_storage(tmp_path: Path):
    storage = RecordingStorage(LocalStorage(tmp_path))
    writer = TokenDbWriter(cast(Storage, storage), registry_dir="")
    baseline = storage.flush_calls  # run-attempts append flushes at construction
    assert baseline >= 1
    writer.append_rows([make_row()])
    writer.flush()
    # Every segment+manifest cycle must push staged appends to the store, so
    # cloud backends see manifests mid-run (not only at close()).
    assert storage.flush_calls > baseline
    after_flush = storage.flush_calls
    writer.flush()  # empty buffer: no segment, no extra storage flush
    assert storage.flush_calls == after_flush
    writer.close()
    assert storage.flush_calls > after_flush


def test_add_label_flushes_storage(tmp_path: Path):
    with TokenDbWriter(tmp_path, registry_dir="") as writer:
        writer.append_rows([make_row()])
    storage = RecordingStorage(LocalStorage(tmp_path))
    reader = ParquetSegmentReader(cast(Storage, storage))
    reader.add_label({"split": "train", "iteration": 0}, "quality", 1, author="tester")
    assert storage.flush_calls == 1
    assert len(reader.labels(label_key="quality")) == 1


def test_chat_store_append_flushes_storage(tmp_path: Path):
    storage = RecordingStorage(LocalStorage(tmp_path))
    store = ChatStore(cast(Storage, storage))
    store.append_event("20240101-000000-abcd1234", {"type": "ping"})
    assert storage.flush_calls == 1
    assert len(store.load_records("20240101-000000-abcd1234")) == 1


def test_visual_store_publish_flushes_storage(tmp_path: Path):
    storage = RecordingStorage(LocalStorage(tmp_path))
    store = VisualStore(cast(Storage, storage), url_base="/visuals")
    store.publish("t", "d", "<html></html>")
    assert storage.flush_calls == 1
    assert len(store.list()) == 1
