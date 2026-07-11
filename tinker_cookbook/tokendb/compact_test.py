"""Tests for token DB segment compaction."""

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from tinker_cookbook.tokendb.compact import ActiveWriterError, compact
from tinker_cookbook.tokendb.writer import SEGMENTS_DIR, TOKENS_DIR, TokenDbWriter
from tinker_cookbook.tokendb.writer_test import make_row, read_all_segments


def _row_key(record: dict) -> tuple:
    return (
        record["writer_id"],
        record["split"],
        record["iteration"],
        record["group_idx"],
        record["traj_idx"],
        record["step_idx"],
    )


def _canonical_records(log_path: Path) -> list[dict]:
    records = read_all_segments(log_path).to_pylist()
    return sorted(records, key=_row_key)


def _segments(log_path: Path) -> list[str]:
    return sorted(p.name for p in (log_path / SEGMENTS_DIR).glob("*.parquet"))


def _manifests(log_path: Path) -> list[str]:
    return sorted(p.name for p in (log_path / TOKENS_DIR).glob("manifest-*.jsonl"))


@pytest.fixture
def multi_writer_store(tmp_path: Path) -> Path:
    """Two writers, several small segments each (buffer_rows=2 forces splits)."""
    for writer_idx in range(2):
        with TokenDbWriter(
            tmp_path, writer_id=f"w{writer_idx}", buffer_rows=2, flush_interval_s=3600
        ) as writer:
            for iteration in range(3):
                writer.append_rows(
                    [
                        make_row(
                            iteration=iteration,
                            group_idx=group_idx,
                            ob_tokens=[writer_idx, iteration, group_idx],
                            ac_tokens=[100 + group_idx],
                        )
                        for group_idx in range(3)
                    ]
                )
    return tmp_path


class TestCompact:
    def test_preserves_all_rows_exactly(self, multi_writer_store: Path):
        before = _canonical_records(multi_writer_store)
        old_segments = _segments(multi_writer_store)
        assert len(old_segments) > 2  # multi-segment, multi-writer fixture

        plan = compact(multi_writer_store, min_quiet_s=0)

        after = _canonical_records(multi_writer_store)
        assert after == before  # every column of every row, exactly
        assert plan.n_rows == len(before)
        new_segments = _segments(multi_writer_store)
        assert new_segments == sorted(plan.new_segments)
        assert len(new_segments) == 1
        assert not set(old_segments) & set(new_segments)
        # Old manifests replaced by the single compacted one.
        manifests = _manifests(multi_writer_store)
        assert len(manifests) == 1
        assert manifests[0].startswith("manifest-compact-")

    def test_rows_sorted_and_split_by_target(self, multi_writer_store: Path):
        plan = compact(multi_writer_store, target_rows_per_segment=4, min_quiet_s=0)
        assert len(plan.new_segments) == (plan.n_rows + 3) // 4
        table = read_all_segments(multi_writer_store)
        iterations = table.column("iteration").to_pylist()
        group_idxs = table.column("group_idx").to_pylist()
        assert sorted(zip(iterations, group_idxs)) == list(zip(iterations, group_idxs))

    def test_reader_identical_after_compaction(self, multi_writer_store: Path):
        pytest.importorskip("duckdb")
        from tinker_cookbook.tokendb.reader import TokenDB

        def snapshot(db: TokenDB):
            return (
                db.query(limit=1000),
                db.query(split="train", min_iteration=1, max_iteration=2, limit=1000),
                db.get_rollout("train", 2, 1, 0),
                db.search(token_subsequence=[101]),
            )

        before = snapshot(TokenDB(multi_writer_store))
        compact(multi_writer_store, min_quiet_s=0)
        after = snapshot(TokenDB(multi_writer_store))
        assert after == before

    def test_dry_run_touches_nothing(self, multi_writer_store: Path):
        before_files = sorted(
            p.relative_to(multi_writer_store) for p in multi_writer_store.rglob("*") if p.is_file()
        )
        before_records = _canonical_records(multi_writer_store)

        plan = compact(multi_writer_store, dry_run=True, min_quiet_s=0)

        assert plan.dry_run is True
        assert plan.n_rows == len(before_records)
        assert plan.old_segments == _segments(multi_writer_store)
        after_files = sorted(
            p.relative_to(multi_writer_store) for p in multi_writer_store.rglob("*") if p.is_file()
        )
        assert after_files == before_files
        assert _canonical_records(multi_writer_store) == before_records

    def test_refuses_recent_manifest(self, multi_writer_store: Path):
        # The fixture's manifests were just written, so a long quiet window
        # must refuse (active-writer heuristic).
        with pytest.raises(ActiveWriterError):
            compact(multi_writer_store, min_quiet_s=3600)

    def test_empty_store_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No token DB segments"):
            compact(tmp_path, min_quiet_s=0)

    def test_recompaction_is_stable(self, multi_writer_store: Path):
        compact(multi_writer_store, min_quiet_s=0)
        before = _canonical_records(multi_writer_store)
        compact(multi_writer_store, min_quiet_s=0)
        assert _canonical_records(multi_writer_store) == before

    def test_mixed_generation_store_compacts_to_v2(self, tmp_path: Path):
        pytest.importorskip("duckdb")
        import json

        import pyarrow as pa

        from tinker_cookbook.tokendb.reader import TokenDB
        from tinker_cookbook.tokendb.reader_test import write_v1_segment

        # One v1 segment (metrics as a JSON string) + one v2 segment.
        write_v1_segment(
            tmp_path, [{"iteration": 0, "metrics": json.dumps({"acc": 0.5}), "ac_text": "old"}]
        )
        with TokenDbWriter(tmp_path, flush_interval_s=3600.0) as writer:
            writer.append_rows(
                [
                    make_row(
                        iteration=1,
                        metrics={"acc": 1.0},
                        attrs={"dataset": "gsm8k"},
                        ac_text="new",
                    )
                ]
            )
        plan = compact(tmp_path, min_quiet_s=0)
        assert plan.n_rows == 2
        # The compacted segment is uniformly v2: metrics is a typed map now.
        table = read_all_segments(tmp_path)
        assert pa.types.is_map(table.schema.field("metrics").type)
        db = TokenDB(tmp_path)
        rows = db.query()
        assert [row["ac_text"] for row in rows] == ["old", "new"]
        assert [row["metrics"]["acc"] for row in rows] == [pytest.approx(0.5), pytest.approx(1.0)]
        assert rows[0]["attrs"] == {} and rows[1]["attrs"] == {"dataset": "gsm8k"}
        agg = db.sql("SELECT avg(metrics['acc']) AS mean_acc FROM rollouts")
        assert agg[0]["mean_acc"] == pytest.approx(0.75)

    def test_compacted_manifest_keeps_observed_keys(self, tmp_path: Path):
        import json

        with TokenDbWriter(tmp_path, flush_interval_s=3600.0) as writer:
            writer.append_rows(
                [
                    make_row(
                        metrics={"acc": 1.0},
                        attrs={"dataset": "gsm8k"},
                        token_metrics={"kl": [0.1, 0.2]},
                        tags=["math"],
                    )
                ]
            )
        compact(tmp_path, min_quiet_s=0)
        # Old manifests (the schema card's source) are gone; the compacted
        # manifest must carry the observed keys forward.
        (manifest,) = _manifests(tmp_path)
        (entry,) = [
            json.loads(line) for line in (tmp_path / TOKENS_DIR / manifest).read_text().splitlines()
        ]
        assert entry["metrics_keys"] == ["acc"]
        assert entry["attrs_keys"] == ["dataset"]
        assert entry["token_metrics_keys"] == ["kl"]
        assert entry["tags"] == ["math"]
