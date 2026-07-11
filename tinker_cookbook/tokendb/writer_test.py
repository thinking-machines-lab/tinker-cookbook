"""Tests for the token DB schema and parquet segment writer."""

import json
import logging
import math
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.tokendb.schema import (
    SCHEMA_VERSION,
    TokenRow,
    coerce_attrs,
    coerce_metrics,
    coerce_token_metrics,
    compute_ob_delta,
)
from tinker_cookbook.tokendb.writer import (
    MANIFEST_MAX_KEYS,
    RUN_ATTEMPTS_PATH,
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
        assert dict(got["metrics"]) == {"format_ok": 1.0}
        assert dict(got["attrs"]) == {}
        assert dict(got["token_metrics"]) == {}
        assert got["tool_calls"] is None
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

    def test_v2_map_columns_roundtrip(self, tmp_path: Path):
        row = make_row(
            metrics={"reward/format": 0.5, "acc": float("nan")},
            attrs={"dataset": "gsm8k", "task_name": "algebra"},
            # Parallel to ac_tokens (len 2); NaN entries must survive parquet.
            token_metrics={"teacher/logprobs": [-1.5, float("nan")]},
            tool_calls=[
                {
                    "name": "bash",
                    "args_json": '{"cmd": "ls"}',
                    "error_type": None,
                    "should_stop": False,
                },
                {
                    "name": "submit",
                    "args_json": "{}",
                    "error_type": "timeout",
                    "should_stop": True,
                },
            ],
        )
        with TokenDbWriter(tmp_path) as writer:
            writer.append_rows([row])
        got = read_all_segments(tmp_path).to_pylist(maps_as_pydicts="strict")[0]
        assert got["metrics"]["reward/format"] == pytest.approx(0.5)
        # A literal NaN metrics value survives parquet as a real float NaN.
        assert math.isnan(got["metrics"]["acc"])
        assert got["attrs"] == {"dataset": "gsm8k", "task_name": "algebra"}
        assert got["token_metrics"]["teacher/logprobs"][0] == pytest.approx(-1.5)
        assert math.isnan(got["token_metrics"]["teacher/logprobs"][1])
        assert got["tool_calls"] == [
            {
                "name": "bash",
                "args_json": '{"cmd": "ls"}',
                "error_type": None,
                "should_stop": False,
            },
            {"name": "submit", "args_json": "{}", "error_type": "timeout", "should_stop": True},
        ]

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


class TestCoercion:
    def test_metrics_coerced_to_float(self):
        out = coerce_metrics({"a": 1, "b": 2.5, "c": True, "d": "3.5"})
        assert out == {"a": 1.0, "b": 2.5, "c": 1.0, "d": 3.5}
        assert all(isinstance(v, float) for v in out.values())

    def test_nan_passes_through(self):
        out = coerce_metrics({"acc": float("nan")})
        assert math.isnan(out["acc"])

    def test_non_coercible_dropped_with_single_warning(self, caplog):
        key = "coercion_test/unique_bad_key"
        with caplog.at_level(logging.WARNING, logger="tinker_cookbook.tokendb.schema"):
            assert coerce_metrics({key: object(), "ok": 1}) == {"ok": 1.0}
            assert coerce_metrics({key: [1, 2]}) == {}
        warnings = [r for r in caplog.records if key in r.getMessage()]
        assert len(warnings) == 1  # warn once per key, not per value

    def test_large_exact_int_warns_once_but_is_stored(self, caplog):
        key = "coercion_test/unique_big_int"
        big = 2**24 + 1
        with caplog.at_level(logging.WARNING, logger="tinker_cookbook.tokendb.schema"):
            assert coerce_metrics({key: big}) == {key: float(big)}
            coerce_metrics({key: big + 1})
        warnings = [r for r in caplog.records if key in r.getMessage()]
        assert len(warnings) == 1

    def test_attrs_values_stringified(self):
        assert coerce_attrs({"dataset": "math", "level": 5, "flag": True}) == {
            "dataset": "math",
            "level": "5",
            "flag": "True",
        }

    def test_token_metrics_coerced_with_nan(self):
        out = coerce_token_metrics(
            {"teacher/logprobs": [-0.5, float("nan")], "kl": (0.1, 0.2)}, n_ac_tokens=2
        )
        assert out["teacher/logprobs"][0] == pytest.approx(-0.5)
        assert math.isnan(out["teacher/logprobs"][1])
        assert out["kl"] == [pytest.approx(0.1), pytest.approx(0.2)]
        assert all(isinstance(v, float) for arr in out.values() for v in arr)

    def test_token_metrics_length_mismatch_dropped_with_single_warning(self, caplog):
        key = "token_metrics_test/unique_short_key"
        with caplog.at_level(logging.WARNING, logger="tinker_cookbook.tokendb.schema"):
            out = coerce_token_metrics({key: [0.1], "ok": [0.1, 0.2]}, n_ac_tokens=2)
            assert out == {"ok": [pytest.approx(0.1), pytest.approx(0.2)]}
            coerce_token_metrics({key: [0.1, 0.2, 0.3]}, n_ac_tokens=2)
        warnings = [r for r in caplog.records if key in r.getMessage()]
        assert len(warnings) == 1  # warn once per key, not per array
        # The first warning names the key and both lengths.
        assert "1" in warnings[0].getMessage() and "2" in warnings[0].getMessage()

    def test_token_metrics_non_coercible_element_drops_whole_array(self, caplog):
        key = "token_metrics_test/unique_bad_element"
        with caplog.at_level(logging.WARNING, logger="tinker_cookbook.tokendb.schema"):
            out = coerce_token_metrics({key: [0.1, object()]}, n_ac_tokens=2)
        assert out == {}
        assert any(key in r.getMessage() for r in caplog.records)

    def test_token_row_coerces_at_construction(self):
        # Coercion happens when the row is built (caller's thread), so a bad
        # value never reaches the flush thread's parquet encode.
        row = make_row(
            metrics={"good": "1.5", "bad": object()},
            attrs={"n": 3},
        )
        assert row.metrics == {"good": 1.5}
        assert row.attrs == {"n": "3"}


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


def read_manifest_lines(log_path: Path, writer_id: str) -> list[dict]:
    manifest_path = log_path / TOKENS_DIR / f"manifest-{writer_id}.jsonl"
    return [json.loads(line) for line in manifest_path.read_text().splitlines()]


class TestManifestObservedKeys:
    def test_manifest_records_observed_keys(self, tmp_path: Path):
        with TokenDbWriter(tmp_path, flush_interval_s=3600.0) as writer:
            writer.append_rows(
                [
                    make_row(metrics={"acc": 1.0, "group/win": 0.5}, tags=["gsm"]),
                    make_row(
                        step_idx=1,
                        metrics={"acc": 0.0},
                        attrs={"dataset": "math"},
                        token_metrics={"teacher/logprobs": [-0.1, -0.2]},
                        tags=["gsm", "hard"],
                    ),
                ]
            )
        (entry,) = read_manifest_lines(tmp_path, writer.writer_id)
        assert entry["metrics_keys"] == ["acc", "group/win"]
        assert entry["attrs_keys"] == ["dataset"]
        assert entry["token_metrics_keys"] == ["teacher/logprobs"]
        assert entry["tags"] == ["gsm", "hard"]
        assert entry["keys_truncated"] is False

    def test_manifest_key_lists_are_capped(self, tmp_path: Path):
        n_keys = MANIFEST_MAX_KEYS + 50
        rows = [make_row(step_idx=i, metrics={f"metric_{i:04d}": float(i)}) for i in range(n_keys)]
        with TokenDbWriter(tmp_path, flush_interval_s=3600.0) as writer:
            writer.append_rows(rows)
        (entry,) = read_manifest_lines(tmp_path, writer.writer_id)
        assert len(entry["metrics_keys"]) == MANIFEST_MAX_KEYS
        assert entry["metrics_keys"] == sorted(entry["metrics_keys"])
        assert entry["keys_truncated"] is True
        # Even a truncated line stays small enough for atomic appends.
        assert len(json.dumps(entry)) < 4096


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

    def test_run_attempts_jsonl_appends_one_line_per_attempt(self, tmp_path: Path):
        # run.json holds only the latest attempt (overwritten); the
        # append-per-attempt record preserves each attempt's context.
        w1 = TokenDbWriter(tmp_path, flush_interval_s=3600.0, context={"learning_rate": 1e-4})
        w1.append_rows([make_row()])
        w1.close()
        w2 = TokenDbWriter(tmp_path, flush_interval_s=3600.0, context={"learning_rate": 5e-5})
        w2.append_rows([make_row()])
        w2.close()
        lines = [
            json.loads(line)
            for line in (tmp_path / RUN_ATTEMPTS_PATH).read_text().splitlines()
            if line.strip()
        ]
        assert [line["run_attempt"] for line in lines] == [1, 2]
        assert all(line["run_id"] == w1.run_id for line in lines)
        assert [line["context"]["learning_rate"] for line in lines] == [1e-4, 5e-5]
        # run.json matches the LAST attempts line.
        assert json.loads((tmp_path / RUN_JSON_PATH).read_text()) == lines[-1]

    def test_worker_context_does_not_touch_run_attempts(self, tmp_path: Path):
        worker = TokenDbWriter(
            tmp_path, flush_interval_s=3600.0, context={"run_id": "abc123", "run_attempt": 4}
        )
        try:
            worker.append_rows([make_row()])
        finally:
            worker.close()
        assert not (tmp_path / RUN_ATTEMPTS_PATH).exists()


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
