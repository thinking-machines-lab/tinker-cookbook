from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from .data import load_prophet_arena_examples, load_prophet_arena_split, parse_utc_datetime


def _row(
    event_ticker: str,
    submission_id: str,
    snapshot_time: str,
    close_time: str,
    *,
    markets: list[str] | None = None,
    target: str | None = None,
    outcome: int = 1,
    source_summary: str = "Information available before the forecast.",
) -> dict[str, str]:
    markets = markets or ["Alpha", "Beta"]
    target = target or markets[0]
    outcomes = {market: int(market == target and outcome == 1) for market in markets}
    return {
        "submission_id": submission_id,
        "event_ticker": event_ticker,
        "title": f"What will happen in {event_ticker}?",
        "snapshot_time": snapshot_time,
        "close_time": close_time,
        "market_data": json.dumps({market: {"yes_ask": 50} for market in markets}),
        "market_outcome": json.dumps(outcomes),
        "category": "Other",
        # The published file mixes JSON objects with Python-style lists.
        "markets": repr(markets),
        "augmented_title": f"Will {target} happen?",
        "rules": f"If the host says {target}, then the market resolves to Yes.",
        "sources": repr(
            [
                {
                    "ranking": 1,
                    "title": "Reference",
                    "summary": source_summary,
                    "url": "https://example.test/reference",
                }
            ]
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2025-10-20", datetime(2025, 10, 20, tzinfo=UTC)),
        ("2025-10-20T00:00:00Z", datetime(2025, 10, 20, tzinfo=UTC)),
        ("2025-10-20T00:00:00-07:00", datetime(2025, 10, 20, 7, tzinfo=UTC)),
        ("2025-10-20T12:30:00+05:30", datetime(2025, 10, 20, 7, tzinfo=UTC)),
    ],
)
def test_parse_utc_datetime(value: str, expected: datetime) -> None:
    assert parse_utc_datetime(value) == expected


def test_loader_uses_every_market_and_its_earliest_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "prophet.csv"
    _write_csv(
        path,
        [
            _row(
                "event",
                "later",
                "2025-09-02T00:00:00+00:00",
                "2025-09-03T00:00:00+00:00",
                markets=["Speaker", "Target phrase"],
                target="Target phrase",
                source_summary="Later material.",
            ),
            _row(
                "event",
                "earlier",
                "2025-09-01T00:00:00+00:00",
                "2025-09-03T00:00:00+00:00",
                markets=["Speaker", "Target phrase"],
                target="Target phrase",
                source_summary="Earlier material.",
            ),
        ],
    )

    examples = load_prophet_arena_examples(path)

    assert [example.market for example in examples] == ["Speaker", "Target phrase"]
    assert [example.outcome for example in examples] == [0, 1]
    assert {example.submission_id for example in examples} == {"earlier"}
    assert {example.event_title for example in examples} == {"What will happen in event?"}
    assert all("Earlier material." in example.reference_material for example in examples)
    assert all("Later material." not in example.reference_material for example in examples)
    assert examples[0].resolution_criteria == (
        "The market resolves YES if this condition occurs: Speaker."
    )
    assert "host says Target phrase" not in examples[0].resolution_criteria


def test_split_requires_closed_training_events_and_later_validation_snapshots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prophet.csv"
    _write_csv(
        path,
        [
            _row(
                "train",
                "train-id",
                "2025-10-01T00:00:00+00:00",
                "2025-10-19T00:00:00+00:00",
            ),
            _row(
                "crossing",
                "crossing-id",
                "2025-10-01T00:00:00+00:00",
                "2025-10-21T00:00:00+00:00",
            ),
            _row(
                "validation",
                "validation-later",
                "2025-10-22T00:00:00+00:00",
                "2025-10-23T00:00:00+00:00",
            ),
            _row(
                "validation",
                "validation-first",
                "2025-10-20T00:00:00+00:00",
                "2025-10-23T00:00:00+00:00",
            ),
        ],
    )

    split = load_prophet_arena_split(
        path,
        split_time=datetime(2025, 10, 20, tzinfo=UTC),
        max_train_questions=None,
        max_validation_questions=None,
    )

    assert {(example.event_ticker, example.market) for example in split.train} == {
        ("train", "Alpha"),
        ("train", "Beta"),
    }
    assert {(example.event_ticker, example.market) for example in split.validation} == {
        ("validation", "Alpha"),
        ("validation", "Beta"),
    }
    assert {example.submission_id for example in split.validation} == {"validation-first"}
    assert split.excluded_crossing_events == 1


def test_split_caps_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "prophet.csv"
    rows = [
        _row(
            f"train-{index}",
            f"train-id-{index}",
            "2025-10-01T00:00:00+00:00",
            "2025-10-19T00:00:00+00:00",
        )
        for index in range(10)
    ]
    rows.extend(
        _row(
            f"validation-{index}",
            f"validation-id-{index}",
            "2025-10-21T00:00:00+00:00",
            "2025-10-22T00:00:00+00:00",
        )
        for index in range(10)
    )
    _write_csv(path, rows)

    first = load_prophet_arena_split(
        path,
        split_time=datetime(2025, 10, 20, tzinfo=UTC),
        max_train_questions=4,
        max_validation_questions=3,
        seed=7,
    )
    repeated = load_prophet_arena_split(
        path,
        split_time=datetime(2025, 10, 20, tzinfo=UTC),
        max_train_questions=4,
        max_validation_questions=3,
        seed=7,
    )

    assert first == repeated
    assert len(first.train) == 4
    assert len(first.validation) == 3


def test_loader_rejects_snapshots_after_market_close(tmp_path: Path) -> None:
    path = tmp_path / "prophet.csv"
    _write_csv(
        path,
        [
            _row(
                "invalid",
                "invalid-id",
                "2025-10-22T00:00:00+00:00",
                "2025-10-21T00:00:00+00:00",
            )
        ],
    )

    with pytest.raises(ValueError, match="snapshot_time must precede close_time"):
        load_prophet_arena_examples(path)
