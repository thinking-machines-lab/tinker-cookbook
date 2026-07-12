"""Tests for the cross-run registry reader (RegistryBackend)."""

import asyncio
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from tinker_cookbook.stores.storage import LocalStorage, Storage
from tinker_cookbook.tokendb.agent_prompt import format_schema_card
from tinker_cookbook.tokendb.interface import TokenStoreBackend
from tinker_cookbook.tokendb.reader import ParquetSegmentReader
from tinker_cookbook.tokendb.reader_test import make_row, write_v1_segment
from tinker_cookbook.tokendb.registry import register_run
from tinker_cookbook.tokendb.registry_backend import RegistryBackend, resolve_segcache_dir
from tinker_cookbook.tokendb.writer import RUN_JSON_PATH, TokenDbWriter

V1_RUN_ID = "v1run"


def make_v1_run(log_path: Path) -> None:
    """A v1-shaped run written with raw pyarrow, as an old writer would.

    One segment with dotted-key and NaN metrics (the payloads SQL-level
    normalization cannot handle), plus a hand-written ``run.json`` and a
    registry record.
    """
    write_v1_segment(
        log_path,
        [
            {
                "iteration": 2,
                "metrics": json.dumps({"time/total.ms": 12.5, "acc": float("nan")}),
                "ac_text": "v1 row",
                "total_reward": 3.0,
            },
            {"iteration": 2, "traj_idx": 1, "metrics": "{}", "total_reward": 1.0},
        ],
    )
    run_json = {
        "schema_version": 1,
        "run_id": V1_RUN_ID,
        "run_attempt": 1,
        "context": {"model_name": "old-model", "config": {"temperature": 0.2}},
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path = log_path / RUN_JSON_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_json) + "\n")
    register_run(log_path=str(log_path), run_id=V1_RUN_ID, run_attempt=1, model_name="old-model")


@pytest.fixture
def two_runs(tmp_path: Path) -> dict:
    """Two registered runs: a v2 run (two attempts) and a raw-pyarrow v1 run."""
    v2_path = tmp_path / "run-v2"
    context = {"model_name": "new-model", "config": {"temperature": 0.7}}
    with TokenDbWriter(v2_path, context=context) as writer:
        writer.append_rows(
            [
                make_row(iteration=0, total_reward=1.0, ac_text="v2 first"),
                make_row(iteration=2, total_reward=0.5, ac_text="attempt one answer"),
            ]
        )
        v2_id = writer.run_id
    # Resume: attempt 2 re-runs iteration 2 (superseding within this run only).
    with TokenDbWriter(v2_path, context=context) as writer:
        assert writer.run_attempt == 2
        writer.append_rows([make_row(iteration=2, total_reward=2.0, ac_text="attempt two answer")])
    v1_path = tmp_path / "run-v1"
    make_v1_run(v1_path)
    return {"v2_id": v2_id, "v2_path": v2_path, "v1_path": v1_path}


def make_backend(**kwargs: Any) -> RegistryBackend:
    kwargs.setdefault("refresh_ttl_s", 0.0)
    return RegistryBackend(**kwargs)


class TestCrossRunSql:
    def test_rollouts_span_both_runs(self, two_runs: dict):
        backend = make_backend()
        rows = backend.query()
        # Sum of the per-run readers' row counts.
        n_v2 = len(ParquetSegmentReader(two_runs["v2_path"]).query())
        n_v1 = len(ParquetSegmentReader(two_runs["v1_path"]).query())
        assert len(rows) == n_v2 + n_v1 == 5
        assert {row["run_id"] for row in rows} == {two_runs["v2_id"], V1_RUN_ID}

    def test_join_trajectories_with_runs_config(self, two_runs: dict):
        backend = make_backend()
        rows = backend.sql(
            """
            SELECT r.temperature, avg(t.total_reward) AS mean_reward, count(*) AS n
            FROM trajectories t JOIN runs r USING (run_id, run_attempt)
            GROUP BY 1 ORDER BY 1
            """
        )
        by_temp = {round(row["temperature"], 3): row for row in rows}
        assert by_temp[0.2]["mean_reward"] == pytest.approx(2.0)  # (3 + 1) / 2
        assert by_temp[0.7]["n"] == 3  # attempts 1 (2 trajs) + 2 (1 traj)

    def test_group_by_run_id(self, two_runs: dict):
        backend = make_backend()
        rows = backend.sql(
            "SELECT run_id, count(*) AS n FROM rollouts GROUP BY run_id ORDER BY run_id"
        )
        assert {row["run_id"]: row["n"] for row in rows} == {two_runs["v2_id"]: 3, V1_RUN_ID: 2}

    def test_select_only_guard(self, two_runs: dict):
        backend = make_backend()
        with pytest.raises(ValueError, match="sql\\(\\)"):
            backend.sql("DROP VIEW rollouts")
        with pytest.raises(ValueError, match="exactly one"):
            backend.sql("SELECT 1; SELECT 2")

    def test_open_writer_raises(self, two_runs: dict):
        with pytest.raises(NotImplementedError):
            make_backend().open_writer({})


class TestSuperseded:
    def test_partitioned_by_run(self, two_runs: dict):
        """Same (split, iteration) in different runs never supersede each other;
        within one run the flag matches the single-run reader."""
        backend = make_backend()
        rows = backend.query(iteration=2)
        flags = {(row["run_id"], row["run_attempt"]): row["superseded"] for row in rows}
        # v2 run: attempt 1 superseded by attempt 2 (single-run semantics)...
        assert flags[(two_runs["v2_id"], 1)] is True
        assert flags[(two_runs["v2_id"], 2)] is False
        single = ParquetSegmentReader(two_runs["v2_path"]).query(iteration=2)
        assert {(r["run_attempt"], r["superseded"]) for r in single} == {(1, True), (2, False)}
        # ...while the v1 run's attempt-1 rows at the same (split, iteration)
        # stay live despite the other run's attempt 2.
        assert flags[(V1_RUN_ID, 1)] is False
        latest = backend.query(iteration=2, latest_only=True)
        assert {row["run_id"] for row in latest} == {two_runs["v2_id"], V1_RUN_ID}


class TestV1Upgrade:
    def test_v1_metrics_survive_with_python_semantics(self, two_runs: dict):
        backend = make_backend()
        rows = backend.query(run_id=V1_RUN_ID, iteration=2, traj_idx=0)
        (row,) = rows
        assert row["metrics"]["time/total.ms"] == pytest.approx(12.5)  # dotted key intact
        assert math.isnan(row["metrics"]["acc"])  # bare-NaN v1 JSON parsed leniently
        assert row["attrs"] == {}
        assert row["tool_calls"] is None

    def test_original_file_untouched_and_upgrade_cached(self, two_runs: dict, tmp_path: Path):
        segments_dir = two_runs["v1_path"] / "tokens" / "segments"
        (original,) = segments_dir.iterdir()
        before = original.read_bytes()
        backend = make_backend()
        backend.query()
        assert original.read_bytes() == before  # store stays append-only
        cache_root = resolve_segcache_dir(None)
        upgraded = list(cache_root.rglob(f"upgraded/{original.name}"))
        assert len(upgraded) == 1
        # A second refresh does not re-upgrade the cached copy.
        mtime = upgraded[0].stat().st_mtime_ns
        backend.refresh(force=True)
        backend.query()
        assert upgraded[0].stat().st_mtime_ns == mtime

    def test_local_v2_segments_are_scanned_in_place(self, two_runs: dict):
        backend = make_backend()
        backend.query()
        cache_root = resolve_segcache_dir(None)
        cached = [p.name for p in cache_root.rglob("seg-*.parquet")]
        # Only the v1 segment was staged; v2 files stay in the store (no copy).
        assert cached == ["seg-v1host-1-000000.parquet"]


class TestCorruptSegment:
    def test_bad_file_is_skipped_others_stay_queryable(self, two_runs: dict):
        segments_dir = two_runs["v2_path"] / "tokens" / "segments"
        (segments_dir / "seg-corrupt-000000.parquet").write_bytes(b"not a parquet file")
        backend = make_backend()
        rows = backend.query()
        assert len(rows) == 5  # both runs' real segments, corrupt one skipped
        # The skip is remembered: a forced refresh reports nothing new.
        assert backend.refresh(force=True) == []


class CountingStorage:
    """Wraps a Storage and counts read() calls per path (delegates the rest)."""

    def __init__(self, inner: Storage) -> None:
        self._inner = inner
        self.reads: dict[str, int] = {}

    def read(self, path: str) -> bytes:
        self.reads[path] = self.reads.get(path, 0) + 1
        return self._inner.read(path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TestSegcacheFillOnce:
    def test_each_segment_fetched_exactly_once(self, two_runs: dict):
        # A non-LocalStorage Storage exercises the cloud staging path: the
        # backend must fetch each segment through Storage exactly once and
        # answer every later query from the cache.
        stores: list[CountingStorage] = []

        def factory(log_path: str) -> Storage:
            store = CountingStorage(LocalStorage(log_path))
            stores.append(store)
            return cast(Storage, store)

        backend = make_backend(storage_factory=factory)
        first = backend.sql("SELECT count(*) AS n FROM rollouts")[0]["n"]
        for _ in range(5):
            backend.refresh(force=True)
            assert backend.sql("SELECT count(*) AS n FROM rollouts")[0]["n"] == first
        segment_reads = {
            path: count
            for store in stores
            for path, count in store.reads.items()
            if path.startswith("tokens/segments/")
        }
        assert segment_reads  # the staging path was actually exercised
        assert all(count == 1 for count in segment_reads.values()), segment_reads


class TestRefreshTtl:
    def test_new_run_discovered_after_ttl_not_before(self, two_runs: dict, tmp_path: Path):
        gated = RegistryBackend(refresh_ttl_s=3600.0)
        assert len({r["run_id"] for r in gated.runs()}) == 2
        # Register a third run after backend construction.
        with TokenDbWriter(tmp_path / "run-late", context={"model_name": "late"}) as writer:
            writer.append_rows([make_row(iteration=0, ac_text="late row")])
            late_id = writer.run_id
        # Within the TTL window the rescan is a no-op: not visible yet.
        assert late_id not in {r["run_id"] for r in gated.runs()}
        # Once the gate opens (forced here instead of sleeping), it appears.
        new_pairs = gated.refresh(force=True)
        assert any(run_id == late_id for run_id, _ in new_pairs)
        assert late_id in {r["run_id"] for r in gated.runs()}
        # A TTL-0 backend sees it on the next call with no forcing.
        assert late_id in {r["run_id"] for r in make_backend().runs()}


class TestPerRolloutMethods:
    def test_get_rollout_requires_run_id_with_multiple_runs(self, two_runs: dict):
        backend = make_backend()
        with pytest.raises(ValueError, match="run_id"):
            backend.get_rollout("train", 2, 0, 0)
        rows = backend.get_rollout("train", 2, 0, 0, run_id=two_runs["v2_id"])
        assert [row["ac_text"] for row in rows] == ["attempt two answer"]  # latest attempt
        rows = backend.get_rollout("train", 2, 0, 0, 1, two_runs["v2_id"])
        assert [row["ac_text"] for row in rows] == ["attempt one answer"]
        with pytest.raises(ValueError, match="run_id"):
            backend.group_traj_idxs("train", 2, 0)
        assert backend.group_traj_idxs("train", 2, 0, run_id=V1_RUN_ID) == [0, 1]

    def test_single_run_registry_needs_no_run_id(self, tmp_path: Path):
        with TokenDbWriter(tmp_path / "only-run", context={}) as writer:
            writer.append_rows([make_row(iteration=1, ac_text="solo")])
        backend = make_backend()
        assert [r["ac_text"] for r in backend.get_rollout("train", 1, 0, 0)] == ["solo"]
        assert backend.group_traj_idxs("train", 1, 0) == [0]


class TestLabels:
    def test_add_label_requires_and_routes_by_run_id(self, two_runs: dict):
        backend = make_backend()
        key = {"split": "train", "iteration": 2, "group_idx": 0, "traj_idx": 0}
        with pytest.raises(ValueError, match="run_id"):
            backend.add_label(key, "quality", "good", author="tester")
        backend.add_label({**key, "run_id": V1_RUN_ID}, "quality", "good", author="tester")
        # The append landed in the owning run's store.
        assert (two_runs["v1_path"] / "tokens" / "labels.jsonl").exists()
        assert not (two_runs["v2_path"] / "tokens" / "labels.jsonl").exists()
        (label,) = backend.labels(label_key="quality")
        assert label["run_id"] == V1_RUN_ID
        assert label["label_value"] == "good"
        with pytest.raises(ValueError, match="unknown run_id"):
            backend.add_label({**key, "run_id": "nope"}, "k", 1, author="a")

    def test_labels_union_backfills_run_id(self, two_runs: dict):
        # A record written per-run without a run_id (older label files) gets
        # the owning run's id in the cross-run view.
        reader = ParquetSegmentReader(two_runs["v2_path"])
        reader.add_label({"split": "train", "iteration": 0}, "flag", True, author="a")
        backend = make_backend()
        (label,) = backend.labels(label_key="flag")
        assert label["run_id"] == two_runs["v2_id"]


class TestSubscribe:
    def test_requires_run_id(self, two_runs: dict):
        backend = make_backend()
        with pytest.raises(ValueError, match="run_id"):
            backend.subscribe()
        with pytest.raises(ValueError, match="unknown run_id"):
            backend.subscribe(run_id="nope")

    def test_yields_rows_written_after_subscription(self, two_runs: dict):
        backend = make_backend()
        run_id = two_runs["v2_id"]

        async def main() -> list[dict]:
            received: list[dict] = []

            async def consume() -> None:
                async for row in backend.subscribe(poll_interval_s=0.05, run_id=run_id):
                    received.append(row)
                    if len(received) >= 2:
                        return

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.15)  # let the subscriber take its baseline
            with TokenDbWriter(two_runs["v2_path"], context={}) as writer:
                writer.append_rows(
                    [
                        make_row(iteration=20, ac_text="live row 1"),
                        make_row(iteration=21, ac_text="live row 2"),
                    ]
                )
            await asyncio.wait_for(task, timeout=5.0)
            return received

        received = asyncio.run(main())
        assert [row["ac_text"] for row in received] == ["live row 1", "live row 2"]
        assert all(row["run_id"] == run_id for row in received)


class TestDashboardStats:
    def test_per_run_matches_single_run_readers(self, two_runs: dict):
        backend = make_backend()
        stats = backend.dashboard_stats()
        per_run = stats["per_run"]
        assert set(per_run) == {two_runs["v2_id"], V1_RUN_ID}
        for run_id, path in [(two_runs["v2_id"], "v2_path"), (V1_RUN_ID, "v1_path")]:
            single = ParquetSegmentReader(two_runs[path]).dashboard_stats()
            cross = per_run[run_id]
            assert cross["n_rows"] == single["n_rows"]
            assert cross["n_filtered_rows"] == single["n_filtered_rows"]
            assert cross["latest_iteration"] == single["latest_iteration"]
            assert cross["mean_recent_reward"] == pytest.approx(single["mean_recent_reward"])
            assert cross["reward_series"] == single["reward_series"]
        # Top-level aggregates span both runs.
        assert stats["n_rows"] == 5
        assert stats["latest_iteration"] == 2
        # With run_id: identical shape to the single-run reader's numbers.
        one = backend.dashboard_stats(run_id=V1_RUN_ID)
        assert "per_run" not in one
        assert one["n_rows"] == 2

    def test_unreadable_run_reports_error_row(self, two_runs: dict):
        def factory(log_path: str) -> Storage:
            if "run-v1" in log_path:
                raise OSError("store unreachable")
            return LocalStorage(log_path)

        backend = make_backend(storage_factory=factory)
        per_run = backend.dashboard_stats()["per_run"]
        assert per_run[V1_RUN_ID]["n_rows"] is None
        assert "unreachable" in per_run[V1_RUN_ID]["error"]
        assert per_run[two_runs["v2_id"]]["n_rows"] == 3  # isolation: the other run is fine


class TestSchemaCard:
    @pytest.fixture
    def keyed_runs(self, tmp_path: Path) -> dict:
        with TokenDbWriter(tmp_path / "run-a", context={}) as writer:
            writer.append_rows(
                [make_row(metrics={"acc": 1.0, "shared": 0.5}, attrs={"dataset": "gsm8k"})]
            )
            id_a = writer.run_id
        with TokenDbWriter(tmp_path / "run-b", context={}) as writer:
            writer.append_rows([make_row(metrics={"shared": 0.1}, tags=["hard"])])
            id_b = writer.run_id
        return {"a": id_a, "b": id_b}

    def test_union_with_per_run_attribution(self, keyed_runs: dict):
        card = make_backend().schema_card()
        assert card["metrics_keys"] == ["acc", "shared"]
        assert card["attrs_keys"] == ["dataset"]
        assert card["tags"] == ["hard"]
        assert card["keys_truncated"] is False
        assert set(card["runs"]) == {keyed_runs["a"], keyed_runs["b"]}
        assert card["runs"][keyed_runs["a"]]["metrics_keys"] == ["acc", "shared"]
        assert card["runs"][keyed_runs["b"]]["metrics_keys"] == ["shared"]

    def test_prompt_rendering_attributes_partial_keys(self, keyed_runs: dict):
        text = format_schema_card(make_backend().schema_card())
        assert "Observed keys across runs (2 runs)" in text
        # `acc` exists only in run a; `shared` everywhere (no suffix).
        assert f"`acc` (only: {keyed_runs['a']})" in text
        assert "`shared` (only:" not in text
        assert "`shared`" in text


class TestProtocolAndConcurrency:
    def test_satisfies_token_store_backend(self, two_runs: dict):
        # The typed assignment is the real check: pyright verifies that
        # RegistryBackend structurally satisfies the enlarged protocol.
        backend: TokenStoreBackend = make_backend()
        assert isinstance(backend, TokenStoreBackend)

    def test_concurrent_queries_on_per_thread_cursors(self, two_runs: dict):
        backend = make_backend()
        backend.refresh()

        async def main() -> list[Any]:
            def query(i: int) -> int:
                # Interleave refreshes with queries across worker threads.
                backend.refresh(force=(i % 3 == 0))
                return backend.sql("SELECT count(*) AS n FROM rollouts")[0]["n"]

            return await asyncio.gather(*[asyncio.to_thread(query, i) for i in range(8)])

        results = asyncio.run(main())
        assert results == [5] * 8
