"""Tests for the token DB schema and parquet segment writer."""

import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.tokendb.schema import SCHEMA_VERSION, TokenRow, compute_ob_delta
from tinker_cookbook.tokendb.writer import (
    RUN_JSON_PATH,
    SEGMENTS_DIR,
    TOKENS_DIR,
    TokenDbWriter,
)


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


def read_all_segments(log_path: Path) -> pa.Table:
    segments_dir = log_path / SEGMENTS_DIR
    tables = [pq.read_table(p) for p in sorted(segments_dir.glob("*.parquet"))]
    assert tables, f"no segments under {segments_dir}"
    return pa.concat_tables(tables)


def list_segments(log_path: Path) -> list[Path]:
    return sorted((log_path / SEGMENTS_DIR).glob("*.parquet"))


class TestComputeObDelta:
    def test_first_step_is_full_ob(self):
        stored, is_delta = compute_ob_delta([], [1, 2, 3])
        assert stored == [1, 2, 3]
        assert is_delta is False

    def test_prefix_extension_stores_suffix(self):
        stored, is_delta = compute_ob_delta([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7])
        assert stored == [6, 7]
        assert is_delta is True

    def test_exact_match_stores_empty_delta(self):
        stored, is_delta = compute_ob_delta([1, 2, 3], [1, 2, 3])
        assert stored == []
        assert is_delta is True

    def test_non_prefix_reset_stores_full_ob(self):
        stored, is_delta = compute_ob_delta([1, 2, 3], [9, 8, 7, 6])
        assert stored == [9, 8, 7, 6]
        assert is_delta is False

    def test_shorter_ob_is_not_a_prefix_extension(self):
        stored, is_delta = compute_ob_delta([1, 2, 3], [1, 2])
        assert stored == [1, 2]
        assert is_delta is False


class TestRoundtrip:
    def test_parquet_roundtrip_fidelity(self, tmp_path: Path):
        row = make_row(
            iteration=7,
            group_idx=2,
            traj_idx=1,
            step_idx=3,
            ob_tokens=[10, 20, 30],
            ob_is_delta=True,
            ac_tokens=[40, 50, 60],
            ac_logprobs=[-0.5, -1.25, -2.0],
            stop_reason="stop",
            reward=1.5,
            episode_done=True,
            total_reward=2.5,
            final_reward=1.0,
            ob_text="hello",
            ac_text="world",
            metrics={"format_ok": 1},
            logs={"env/row_id": "gsm8k-42"},
            tags=["gsm", "math"],
            env_row_id="gsm8k-42",
            sampling_client_step=12,
        )
        with TokenDbWriter(tmp_path) as writer:
            writer.append_rows([row])
        table = read_all_segments(tmp_path)
        assert table.num_rows == 1
        got = table.to_pylist()[0]
        assert got["ob_tokens"] == [10, 20, 30]
        assert got["ob_is_delta"] is True
        assert got["ac_tokens"] == [40, 50, 60]
        assert got["ac_logprobs"] == pytest.approx([-0.5, -1.25, -2.0])
        assert got["stop_reason"] == "stop"
        assert got["has_images"] is False
        assert got["reward"] == pytest.approx(1.5)
        assert got["episode_done"] is True
        assert got["total_reward"] == pytest.approx(2.5)
        assert got["final_reward"] == pytest.approx(1.0)
        assert got["ob_text"] == "hello"
        assert got["ac_text"] == "world"
        assert json.loads(got["metrics"]) == {"format_ok": 1}
        assert json.loads(got["logs"]) == {"env/row_id": "gsm8k-42"}
        assert json.loads(got["extra"]) == {}
        assert got["tags"] == ["gsm", "math"]
        assert got["env_row_id"] == "gsm8k-42"
        assert got["sampling_client_step"] == 12
        assert got["split"] == "train"
        assert got["iteration"] == 7
        assert (got["group_idx"], got["traj_idx"], got["step_idx"]) == (2, 1, 3)
        assert got["source"] == "rollout"
        assert got["filtered_reason"] is None
        # Writer identity stamped on every row
        assert got["run_id"] == writer.run_id
        assert got["run_attempt"] == 1
        assert got["writer_id"] == writer.writer_id

    def test_nullable_columns_roundtrip_as_none(self, tmp_path: Path):
        row = make_row(ac_logprobs=None, ob_text=None, ac_text=None)
        with TokenDbWriter(tmp_path) as writer:
            writer.append_rows([row])
        got = read_all_segments(tmp_path).to_pylist()[0]
        assert got["ac_logprobs"] is None
        assert got["ob_text"] is None
        assert got["ac_text"] is None
        assert got["env_row_id"] is None
        assert got["sampling_client_step"] is None


class TestFlushTriggers:
    def test_flush_on_buffer_rows(self, tmp_path: Path):
        writer = TokenDbWriter(tmp_path, buffer_rows=3, flush_interval_s=3600.0)
        try:
            writer.append_rows([make_row(step_idx=i) for i in range(2)])
            assert not list_segments(tmp_path)
            writer.append_rows([make_row(step_idx=2)])
            segments = list_segments(tmp_path)
            assert len(segments) == 1
            assert pq.read_table(segments[0]).num_rows == 3
        finally:
            writer.close()

    def test_flush_on_close(self, tmp_path: Path):
        writer = TokenDbWriter(tmp_path, flush_interval_s=3600.0)
        writer.append_rows([make_row(), make_row(step_idx=1)])
        assert not list_segments(tmp_path)
        writer.close()
        assert read_all_segments(tmp_path).num_rows == 2

    def test_flush_on_interval(self, tmp_path: Path):
        writer = TokenDbWriter(tmp_path, flush_interval_s=0.05)
        try:
            writer.append_rows([make_row()])
            deadline = time.monotonic() + 5.0
            while not list_segments(tmp_path):
                assert time.monotonic() < deadline, "interval flush never happened"
                time.sleep(0.01)
        finally:
            writer.close()
        assert read_all_segments(tmp_path).num_rows == 1

    def test_close_is_idempotent(self, tmp_path: Path):
        writer = TokenDbWriter(tmp_path)
        writer.append_rows([make_row()])
        writer.close()
        writer.close()
        assert read_all_segments(tmp_path).num_rows == 1
        with pytest.raises(RuntimeError):
            writer.append_rows([make_row()])


class _RecordingStorage(LocalStorage):
    """LocalStorage that records the order of write/append calls."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.calls: list[tuple[str, str]] = []

    def write(self, path: str, data: bytes) -> None:
        self.calls.append(("write", path))
        super().write(path, data)

    def append(self, path: str, data: bytes) -> None:
        self.calls.append(("append", path))
        super().append(path, data)


class TestSegmentManifestOrdering:
    def test_segment_written_before_manifest_line(self, tmp_path: Path):
        storage = _RecordingStorage(tmp_path)
        with TokenDbWriter(storage, flush_interval_s=3600.0) as writer:
            writer.append_rows([make_row()])
            writer.flush()
            writer.append_rows([make_row(iteration=1)])
        segment_calls = [
            (idx, path)
            for idx, (op, path) in enumerate(storage.calls)
            if op == "write" and path.startswith(SEGMENTS_DIR)
        ]
        manifest_calls = [
            (idx, path)
            for idx, (op, path) in enumerate(storage.calls)
            if op == "append" and "manifest-" in path
        ]
        assert len(segment_calls) == 2
        assert len(manifest_calls) == 2
        for (seg_idx, _), (man_idx, _) in zip(segment_calls, manifest_calls):
            assert seg_idx < man_idx

    def test_manifest_lines_reference_existing_segments(self, tmp_path: Path):
        with TokenDbWriter(tmp_path, flush_interval_s=3600.0) as writer:
            writer.append_rows([make_row(iteration=3), make_row(iteration=5, step_idx=1)])
        manifest_path = tmp_path / TOKENS_DIR / f"manifest-{writer.writer_id}.jsonl"
        lines = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        assert len(lines) == 1
        entry = lines[0]
        segment_path = tmp_path / TOKENS_DIR / entry["path"]
        assert segment_path.exists()
        assert pq.read_table(segment_path).num_rows == entry["n_rows"] == 2
        assert entry["min_iteration"] == 3
        assert entry["max_iteration"] == 5
        assert entry["writer_id"] == writer.writer_id
        assert entry["schema_version"] == SCHEMA_VERSION


class TestMultiWriter:
    def test_distinct_writer_ids_do_not_collide(self, tmp_path: Path):
        w1 = TokenDbWriter(tmp_path, flush_interval_s=3600.0)
        w2 = TokenDbWriter(
            tmp_path,
            flush_interval_s=3600.0,
            context={"run_id": w1.run_id, "run_attempt": w1.run_attempt},
        )
        try:
            assert w1.writer_id != w2.writer_id
            w1.append_rows([make_row(group_idx=0)])
            w2.append_rows([make_row(group_idx=1)])
        finally:
            w1.close()
            w2.close()
        segments = list_segments(tmp_path)
        assert len(segments) == 2
        assert read_all_segments(tmp_path).num_rows == 2
        manifests = sorted((tmp_path / TOKENS_DIR).glob("manifest-*.jsonl"))
        assert {m.name for m in manifests} == {
            f"manifest-{w1.writer_id}.jsonl",
            f"manifest-{w2.writer_id}.jsonl",
        }

    def test_worker_context_does_not_touch_run_json(self, tmp_path: Path):
        worker = TokenDbWriter(
            tmp_path, flush_interval_s=3600.0, context={"run_id": "abc123", "run_attempt": 4}
        )
        try:
            worker.append_rows([make_row()])
        finally:
            worker.close()
        assert not (tmp_path / RUN_JSON_PATH).exists()
        got = read_all_segments(tmp_path).to_pylist()[0]
        assert got["run_id"] == "abc123"
        assert got["run_attempt"] == 4


class TestRunAttempt:
    def test_run_attempt_increments_on_reconstruction(self, tmp_path: Path):
        w1 = TokenDbWriter(tmp_path, flush_interval_s=3600.0)
        w1.append_rows([make_row()])
        w1.close()
        w2 = TokenDbWriter(tmp_path, flush_interval_s=3600.0)
        w2.append_rows([make_row()])
        w2.close()
        assert w1.run_attempt == 1
        assert w2.run_attempt == 2
        assert w1.run_id == w2.run_id
        run_json = json.loads((tmp_path / RUN_JSON_PATH).read_text())
        assert run_json["run_id"] == w1.run_id
        assert run_json["run_attempt"] == 2
        attempts = sorted(row["run_attempt"] for row in read_all_segments(tmp_path).to_pylist())
        assert attempts == [1, 2]


class TestConcurrency:
    def test_concurrent_appends_lose_no_rows(self, tmp_path: Path):
        n_threads = 8
        rows_per_thread = 50
        writer = TokenDbWriter(tmp_path, buffer_rows=16, flush_interval_s=0.02)

        def worker(thread_idx: int) -> None:
            for i in range(rows_per_thread):
                writer.append_rows(
                    [make_row(group_idx=thread_idx, step_idx=i, env_row_id=f"{thread_idx}-{i}")]
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        writer.close()

        table = read_all_segments(tmp_path)
        assert table.num_rows == n_threads * rows_per_thread
        row_ids = {row["env_row_id"] for row in table.to_pylist()}
        assert len(row_ids) == n_threads * rows_per_thread
