"""Tests for the token DB read path (DuckDB reader + TokenDB facade)."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.tokendb.reader import (
    LABELS_PATH,
    ParquetSegmentReader,
    TokenDB,
    reconstruct_full_ob,
)
from tinker_cookbook.tokendb.schema import TokenRow, compute_ob_delta
from tinker_cookbook.tokendb.writer import ParquetSegmentBackend, TokenDbWriter


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


def write_label_line(log_path: Path, **fields) -> None:
    record = {
        "run_id": None,
        "split": None,
        "iteration": None,
        "group_idx": None,
        "traj_idx": None,
        "step_idx": None,
        "label_key": None,
        "label_value": None,
        "author": None,
        "ts": datetime.now(UTC).isoformat(),
        "note": None,
    }
    record.update(fields)
    path = log_path / LABELS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def make_multistep_trajectory(
    split: str, iteration: int, group_idx: int, traj_idx: int, **overrides
) -> list[TokenRow]:
    """A 3-step trajectory with delta-encoded obs (step 2 is a non-prefix reset)."""
    obs_full = [
        [10, 11, 12],
        [10, 11, 12, 20, 21, 30],  # prefix extension of step-0 ob + ac
        [99, 98],  # non-prefix reset
    ]
    acs = [[20, 21], [40, 41], [50]]
    rows = []
    prev_sequence: list[int] = []
    for step_idx, (ob, ac) in enumerate(zip(obs_full, acs)):
        stored_ob, is_delta = compute_ob_delta(prev_sequence, ob)
        rows.append(
            make_row(
                split=split,
                iteration=iteration,
                group_idx=group_idx,
                traj_idx=traj_idx,
                step_idx=step_idx,
                ob_tokens=stored_ob,
                ob_is_delta=is_delta,
                ac_tokens=ac,
                **overrides,
            )
        )
        prev_sequence = ob + ac
    return rows


@pytest.fixture
def populated_store(tmp_path: Path) -> Path:
    """A store with two writers, two run attempts, delta obs, filtered rows, labels."""
    log_path = tmp_path / "run"
    # Coordinator writer, attempt 1.
    with TokenDbWriter(log_path, context={"model_name": "test-model"}) as writer_a:
        writer_a.append_rows(
            [
                make_row(
                    iteration=0,
                    group_idx=0,
                    traj_idx=0,
                    ob_tokens=[1, 2],
                    ac_tokens=[3, 4, 5],
                    ac_logprobs=[-0.1, -0.2, -0.3],
                    stop_reason="stop_token",
                    total_reward=1.0,
                    final_reward=1.0,
                    tags=["easy", "math"],
                    env_row_id="row-42",
                    ob_text="what is 2+2",
                    ac_text="the answer is 4",
                ),
                make_row(
                    iteration=0,
                    group_idx=0,
                    traj_idx=1,
                    ac_tokens=[7, 8],
                    stop_reason="length",
                    total_reward=-1.0,
                    final_reward=-1.0,
                    ac_text="i give up",
                ),
                make_row(
                    iteration=1,
                    group_idx=0,
                    traj_idx=0,
                    source="filtered",
                    filtered_reason="constant_reward",
                    ac_text="all same reward",
                ),
                make_row(
                    iteration=2,
                    group_idx=0,
                    traj_idx=0,
                    total_reward=0.5,
                    ac_text="attempt one answer",
                ),
            ]
            + make_multistep_trajectory("train", 0, 1, 0, total_reward=2.0, final_reward=2.0)
        )
        run_id = writer_a.run_id
        # Second writer (worker): explicit identity, same attempt.
        with TokenDbWriter(
            log_path,
            context={"run_id": run_id, "run_attempt": writer_a.run_attempt},
        ) as writer_b:
            writer_b.append_rows(
                [
                    make_row(
                        iteration=0,
                        group_idx=2,
                        traj_idx=0,
                        split="test",
                        ac_text="eval rollout",
                        total_reward=3.0,
                    )
                ]
            )
    # Coordinator resume: attempt 2 re-runs iteration 2.
    with TokenDbWriter(log_path, context={"model_name": "test-model"}) as writer_c:
        assert writer_c.run_attempt == 2
        writer_c.append_rows(
            [make_row(iteration=2, group_idx=0, traj_idx=0, ac_text="attempt two answer")]
        )
    # Labels: one overwritten value, one tombstoned key.
    base_key = {"run_id": run_id, "split": "train", "iteration": 0, "group_idx": 0, "traj_idx": 0}
    write_label_line(
        log_path,
        **base_key,
        label_key="quality",
        label_value="bad",
        author="alice",
        ts="2026-07-11T00:00:00+00:00",
    )
    write_label_line(
        log_path,
        **base_key,
        label_key="quality",
        label_value="good",
        author="bob",
        ts="2026-07-11T01:00:00+00:00",
    )
    write_label_line(
        log_path,
        **base_key,
        label_key="flagged",
        label_value=True,
        author="alice",
        ts="2026-07-11T00:30:00+00:00",
    )
    write_label_line(
        log_path,
        **base_key,
        label_key="flagged",
        label_value=None,  # tombstone
        author="alice",
        ts="2026-07-11T02:00:00+00:00",
    )
    return log_path


class TestIncrementalRegistration:
    def test_picks_up_segments_written_after_construction(self, populated_store: Path):
        reader = ParquetSegmentReader(populated_store)
        n_before = len(reader.query())
        assert n_before > 0
        with TokenDbWriter(populated_store, context={}) as writer:
            writer.append_rows([make_row(iteration=10, ac_text="late row")])
        rows = reader.query(iteration=10)
        assert len(rows) == 1
        assert rows[0]["ac_text"] == "late row"
        assert len(reader.query()) == n_before + 1

    def test_refresh_registers_each_file_once(self, populated_store: Path):
        reader = ParquetSegmentReader(populated_store)
        first = reader.refresh()
        assert len(first) > 0
        assert reader.refresh() == []


class TestQueryFilters:
    @pytest.fixture
    def db(self, populated_store: Path) -> TokenDB:
        return TokenDB(populated_store)

    def test_multi_writer_rows_merge(self, db: TokenDB):
        writer_ids = {row["writer_id"] for row in db.query()}
        assert len(writer_ids) == 3  # coordinator, worker, resume

    def test_split_filter(self, db: TokenDB):
        rows = db.query(split="test")
        assert len(rows) == 1
        assert rows[0]["ac_text"] == "eval rollout"

    def test_iteration_range(self, db: TokenDB):
        rows = db.query(min_iteration=1, max_iteration=2)
        assert rows
        assert all(1 <= row["iteration"] <= 2 for row in rows)

    def test_exact_iteration_and_group_traj(self, db: TokenDB):
        rows = db.query(iteration=0, group_idx=0, traj_idx=1)
        assert len(rows) == 1
        assert rows[0]["ac_text"] == "i give up"

    def test_reward_range(self, db: TokenDB):
        rows = db.query(min_reward=0.9, max_reward=2.5)
        assert rows
        assert all(0.9 <= row["total_reward"] <= 2.5 for row in rows)

    def test_stop_reason(self, db: TokenDB):
        rows = db.query(stop_reason="length")
        assert len(rows) == 1
        assert rows[0]["traj_idx"] == 1

    def test_source_and_filtered_reason(self, db: TokenDB):
        rows = db.query(source="filtered")
        assert len(rows) == 1
        assert rows[0]["filtered_reason"] == "constant_reward"
        assert db.query(filtered_reason="constant_reward") == rows

    def test_tag_contains(self, db: TokenDB):
        rows = db.query(tag="math")
        assert len(rows) == 1
        assert rows[0]["env_row_id"] == "row-42"

    def test_env_row_id(self, db: TokenDB):
        rows = db.query(env_row_id="row-42")
        assert len(rows) == 1

    def test_run_attempt_filter(self, db: TokenDB):
        rows = db.query(iteration=2, run_attempt=1)
        assert [row["ac_text"] for row in rows] == ["attempt one answer"]

    def test_text_regex(self, db: TokenDB):
        rows = db.query(text_regex=r"answer is \d")
        assert len(rows) == 1
        assert rows[0]["ac_text"] == "the answer is 4"
        # Regex also matches over ob_text.
        assert len(db.query(text_regex=r"2\+2")) == 1

    def test_limit_offset_and_ordering(self, db: TokenDB):
        rows = db.query()
        assert [r["iteration"] for r in rows] == sorted(r["iteration"] for r in rows)
        page = db.query(limit=2, offset=1)
        assert page == rows[1:3]

    def test_run_id_filter(self, db: TokenDB):
        run_id = db.query()[0]["run_id"]
        assert len(db.query(run_id=run_id)) == len(db.query())
        assert db.query(run_id="nonexistent") == []


class TestSuperseded:
    def test_superseded_flag_and_latest_view(self, populated_store: Path):
        db = TokenDB(populated_store)
        rows = db.query(iteration=2)
        assert {row["ac_text"]: row["superseded"] for row in rows} == {
            "attempt one answer": True,
            "attempt two answer": False,
        }
        # Rows at (train, 0) were only produced by attempt 1: not superseded.
        assert all(not row["superseded"] for row in db.query(iteration=0))
        latest = db.query(iteration=2, latest_only=True)
        assert [row["ac_text"] for row in latest] == ["attempt two answer"]

    def test_trajectories_view(self, populated_store: Path):
        db = TokenDB(populated_store)
        rows = db.sql(
            "SELECT * FROM trajectories WHERE split='train' AND iteration=0 AND group_idx=1"
        )
        assert len(rows) == 1
        traj = rows[0]
        assert traj["n_steps"] == 3
        assert traj["n_ac_tokens"] == 5  # 2 + 2 + 1
        assert traj["total_reward"] == pytest.approx(2.0)


class TestGetRollout:
    def test_ordering_and_delta_reconstruction(self, populated_store: Path):
        db = TokenDB(populated_store)
        rows = db.get_rollout("train", 0, 1, 0)
        assert [row["step_idx"] for row in rows] == [0, 1, 2]
        assert rows[1]["ob_is_delta"] is True
        assert rows[2]["ob_is_delta"] is False
        full_obs = reconstruct_full_ob(rows)
        assert full_obs == [
            [10, 11, 12],
            [10, 11, 12, 20, 21, 30],
            [99, 98],
        ]

    def test_defaults_to_latest_attempt(self, populated_store: Path):
        db = TokenDB(populated_store)
        rows = db.get_rollout("train", 2, 0, 0)
        assert [row["ac_text"] for row in rows] == ["attempt two answer"]
        rows_v1 = db.get_rollout("train", 2, 0, 0, run_attempt=1)
        assert [row["ac_text"] for row in rows_v1] == ["attempt one answer"]

    def test_missing_rollout_is_empty(self, populated_store: Path):
        db = TokenDB(populated_store)
        assert db.get_rollout("train", 999, 0, 0) == []


class TestSearch:
    def test_regex_search(self, populated_store: Path):
        db = TokenDB(populated_store)
        rows = db.search(regex=r"give up")
        assert len(rows) == 1
        assert rows[0]["traj_idx"] == 1
        # Restricting fields to ob_text misses an ac_text-only match.
        assert db.search(regex=r"give up", fields=["ob_text"]) == []

    def test_bad_field_rejected(self, populated_store: Path):
        with pytest.raises(ValueError, match="Unsupported search fields"):
            TokenDB(populated_store).search(regex="x", fields=["ac_tokens"])

    def test_token_subsequence(self, populated_store: Path):
        db = TokenDB(populated_store)
        rows = db.search(token_subsequence=[4, 5])
        assert len(rows) >= 1
        assert all(
            any(row["ac_tokens"][i : i + 2] == [4, 5] for i in range(len(row["ac_tokens"])))
            for row in rows
        )
        # Present individually but never contiguous in this order.
        assert db.search(token_subsequence=[5, 3]) == []
        # Contiguity check: [3, 5] never adjacent even though both in [3, 4, 5].
        assert db.search(token_subsequence=[3, 5]) == []

    def test_hit_counts_grouped_by_iteration(self, populated_store: Path):
        db = TokenDB(populated_store)
        counts = db.search_hit_counts(regex=r"answer")
        assert counts == {0: 1, 2: 2}


class TestLabels:
    def test_last_write_wins_and_tombstone(self, populated_store: Path):
        db = TokenDB(populated_store)
        labels = db.labels()
        assert len(labels) == 1  # "flagged" tombstoned; "quality" deduped
        assert labels[0]["label_key"] == "quality"
        assert labels[0]["label_value"] == "good"
        assert labels[0]["author"] == "bob"
        assert db.labels(label_key="flagged") == []

    def test_add_label_roundtrip(self, populated_store: Path):
        db = TokenDB(populated_store)
        key = {"split": "train", "iteration": 0, "group_idx": 0, "traj_idx": 1}
        db.add_label(key, "verdict", {"score": 0.9}, author="agent", note="looks wrong")
        labels = db.labels(label_key="verdict")
        assert len(labels) == 1
        assert labels[0]["label_value"] == {"score": 0.9}
        assert labels[0]["author"] == "agent"
        assert labels[0]["note"] == "looks wrong"
        assert labels[0]["traj_idx"] == 1
        # Tombstone through the API deletes it.
        db.add_label(key, "verdict", None, author="agent")
        assert db.labels(label_key="verdict") == []

    def test_add_label_rejects_bad_key_fields(self, populated_store: Path):
        with pytest.raises(ValueError, match="Unsupported label key"):
            TokenDB(populated_store).add_label({"bogus": 1}, "k", "v", author="a")

    def test_labels_before_any_label_file(self, tmp_path: Path):
        log_path = tmp_path / "empty-run"
        with TokenDbWriter(log_path, context={}) as writer:
            writer.append_rows([make_row()])
        assert TokenDB(log_path).labels() == []


class TestSql:
    def test_select_and_with_allowed(self, populated_store: Path):
        db = TokenDB(populated_store)
        rows = db.sql("SELECT count(*) AS n FROM rollouts WHERE split = ?", ["train"])
        assert rows[0]["n"] > 0
        rows = db.sql(
            "WITH t AS (SELECT iteration FROM rollouts) SELECT max(iteration) AS m FROM t"
        )
        assert rows[0]["m"] == 2

    @pytest.mark.parametrize(
        "bad_sql",
        [
            "INSERT INTO segment_rows SELECT * FROM segment_rows",
            "UPDATE segment_rows SET split = 'x'",
            "DELETE FROM segment_rows",
            "ATTACH ':memory:' AS other",
            "COPY segment_rows TO '/tmp/out.parquet'",
            "DROP VIEW rollouts",
            "CREATE TABLE evil (x INT)",
            "SELECT 1; SELECT 2",
            "SELECT 1; DROP VIEW rollouts",
        ],
    )
    def test_non_select_rejected(self, populated_store: Path, bad_sql: str):
        db = TokenDB(populated_store)
        with pytest.raises(ValueError, match="sql\\(\\)"):
            db.sql(bad_sql)

    def test_facade_from_backend(self, populated_store: Path):
        backend = ParquetSegmentBackend(LocalStorage(populated_store))
        db = TokenDB(backend)
        assert db.query(split="test")


class TestSubscribe:
    def test_yields_rows_written_after_subscription(self, populated_store: Path):
        async def run() -> list[dict]:
            db = TokenDB(populated_store)
            received: list[dict] = []

            async def consume() -> None:
                async for row in db.iter_new(poll_interval_s=0.05, split="train"):
                    received.append(row)
                    if len(received) >= 2:
                        return

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.15)  # let the subscriber take its baseline
            with TokenDbWriter(populated_store, context={}) as writer:
                writer.append_rows(
                    [
                        make_row(iteration=20, ac_text="live row 1"),
                        make_row(iteration=21, ac_text="live row 2"),
                        make_row(iteration=21, split="test", ac_text="filtered out"),
                    ]
                )
            await asyncio.wait_for(task, timeout=5.0)
            return received

        received = asyncio.run(run())
        assert [row["ac_text"] for row in received] == ["live row 1", "live row 2"]
        assert all(row["split"] == "train" for row in received)
