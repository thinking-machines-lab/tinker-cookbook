"""Download Prophet Arena and build a leakage-resistant temporal split."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DATASET_REPOSITORY = "prophetarena/Prophet-Arena-Subset-1200"
DEFAULT_DATASET_REVISION = "c94b6f450d7fe3b03688799cce1c8b29838b5d96"
DEFAULT_DATASET_SHA256 = "1e34a5970e515cd06a2e57a955074561308cb8734c8c5712adb1452b977b3984"
DEFAULT_CACHE_DIR = "~/.cache/tinker-cookbook/prophet-arena"
DEFAULT_SPLIT_DATE = "2025-10-20"
DEFAULT_MAX_TRAIN_QUESTIONS = 1024
DEFAULT_MAX_VALIDATION_QUESTIONS = 256

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastExample:
    """One binary market forecast at its earliest recorded snapshot."""

    submission_id: str
    event_ticker: str
    event_title: str
    market: str
    reference_material: str
    resolution_criteria: str
    snapshot_time: datetime
    close_time: datetime
    category: str
    outcome: int


@dataclass(frozen=True)
class ForecastSplit:
    """Training and validation examples separated by an outcome-availability gap."""

    train: tuple[ForecastExample, ...]
    validation: tuple[ForecastExample, ...]
    split_time: datetime
    excluded_crossing_events: int


def _safe_revision(revision: str) -> str:
    if not revision or revision in {".", ".."}:
        raise ValueError("dataset revision must be a non-empty Git revision")
    if any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
        for character in revision
    ):
        raise ValueError(f"unsafe dataset revision: {revision!r}")
    return revision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_prophet_arena(
    revision: str = DEFAULT_DATASET_REVISION,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Path:
    """Download a commit-pinned Prophet Arena CSV and return its path."""
    revision = _safe_revision(revision)
    expected_sha256 = DEFAULT_DATASET_SHA256 if revision == DEFAULT_DATASET_REVISION else None
    target = Path(cache_dir).expanduser() / revision.replace("/", "--") / "subset_data_1200.csv"
    if target.exists():
        if expected_sha256 is not None and _sha256(target) != expected_sha256:
            raise ValueError(f"cached Prophet Arena file has an unexpected checksum: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    quoted_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/{quoted_revision}/subset_data_1200.csv"
    request = urllib.request.Request(url, headers={"User-Agent": "tinker-forecast-cookbook/1"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}-", dir=target.parent, delete=False
        ) as output:
            temporary_path = Path(output.name)
            logger.info("Downloading Prophet Arena revision %s", revision)
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, output)
        if expected_sha256 is not None and _sha256(temporary_path) != expected_sha256:
            raise ValueError("downloaded Prophet Arena file has an unexpected checksum")
        os.replace(temporary_path, target)
        return target
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _required_text(row: Mapping[str, str], key: str, context: str) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"expected non-empty {key!r} in {context}")
    return value


def parse_utc_datetime(value: str, context: str = "datetime") -> datetime:
    """Parse an ISO datetime and normalize it to UTC.

    Values without an explicit timezone are interpreted as UTC. Values with
    an offset are converted to UTC without changing the instant they represent.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO datetime for {context}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_structured(value: str, context: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid structured value for {context}") from exc


def _parse_json_list(value: str, context: str) -> list[object]:
    parsed = _parse_structured(value, context)
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON list for {context}")
    return parsed


def _parse_json_object(value: str, context: str) -> Mapping[str, object]:
    parsed = _parse_structured(value, context)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object for {context}")
    return parsed


def _format_sources(raw_sources: Sequence[object], context: str) -> str:
    sources: list[tuple[int, str, str]] = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ValueError(f"expected {context}.sources[{index}] to be an object")
        title = raw_source.get("title")
        summary = raw_source.get("summary")
        ranking = raw_source.get("ranking", index + 1)
        if not isinstance(title, str) or not isinstance(summary, str):
            raise ValueError(f"expected source title and summary strings in {context}")
        if isinstance(ranking, bool) or not isinstance(ranking, int):
            ranking = index + 1
        if title.strip() or summary.strip():
            sources.append((ranking, title.strip(), summary.strip()))
    sources.sort(key=lambda item: (item[0], item[1]))
    return "\n\n".join(
        f"{position}. {title}\n{summary}"
        for position, (_, title, summary) in enumerate(sources, start=1)
    )


def _row_to_examples(row: Mapping[str, str], row_number: int) -> list[ForecastExample]:
    context = f"row {row_number}"
    submission_id = _required_text(row, "submission_id", context)
    event_ticker = _required_text(row, "event_ticker", context)
    category = _required_text(row, "category", context)
    markets = _parse_json_list(_required_text(row, "markets", context), f"{context}.markets")
    outcomes = _parse_json_object(
        _required_text(row, "market_outcome", context), f"{context}.market_outcome"
    )
    if not markets or not all(isinstance(market, str) and market.strip() for market in markets):
        raise ValueError(f"expected non-empty string markets in {context}")
    market_names = [market for market in markets if isinstance(market, str)]
    if set(market_names) != set(outcomes):
        raise ValueError(f"markets and market_outcome disagree in {context}")

    title = _required_text(row, "title", context)
    raw_sources = _parse_json_list(_required_text(row, "sources", context), f"{context}.sources")
    reference_material = _format_sources(raw_sources, context)
    if not reference_material:
        raise ValueError(f"expected at least one source in {context}")
    snapshot_time = parse_utc_datetime(
        _required_text(row, "snapshot_time", context), f"{context}.snapshot_time"
    )
    close_time = parse_utc_datetime(
        _required_text(row, "close_time", context), f"{context}.close_time"
    )
    if snapshot_time >= close_time:
        raise ValueError(f"snapshot_time must precede close_time in {context}")

    examples: list[ForecastExample] = []
    for market in market_names:
        raw_outcome = outcomes[market]
        if isinstance(raw_outcome, bool) or raw_outcome not in (0, 1):
            raise ValueError(f"expected a binary outcome for {market!r} in {context}")
        examples.append(
            ForecastExample(
                submission_id=submission_id,
                event_ticker=event_ticker,
                event_title=title,
                market=market,
                reference_material=reference_material,
                resolution_criteria=(
                    f"The market resolves YES if this condition occurs: {market}."
                ),
                snapshot_time=snapshot_time,
                close_time=close_time,
                category=category,
                outcome=int(raw_outcome),
            )
        )
    return examples


def load_prophet_arena_examples(csv_path: str | Path) -> list[ForecastExample]:
    """Load the earliest snapshot of each binary market."""
    path = Path(csv_path).expanduser()
    by_market: dict[tuple[str, str], list[ForecastExample]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required_columns = {
            "submission_id",
            "event_ticker",
            "title",
            "snapshot_time",
            "close_time",
            "market_outcome",
            "category",
            "markets",
            "sources",
        }
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            missing = sorted(required_columns.difference(reader.fieldnames or ()))
            raise ValueError(f"Prophet Arena CSV is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            for example in _row_to_examples(row, row_number):
                by_market[(example.event_ticker, example.market)].append(example)

    if not by_market:
        raise ValueError(f"no Prophet Arena examples found in {path}")

    examples: list[ForecastExample] = []
    for (event_ticker, market), snapshots in by_market.items():
        stable_fields = {
            (
                example.event_title,
                example.market,
                example.resolution_criteria,
                example.close_time,
                example.category,
                example.outcome,
            )
            for example in snapshots
        }
        if len(stable_fields) != 1:
            raise ValueError(f"market {event_ticker}:{market} changes across snapshots")
        examples.append(min(snapshots, key=lambda item: (item.snapshot_time, item.submission_id)))
    examples.sort(key=lambda item: (item.snapshot_time, item.event_ticker, item.market))
    return examples


def _sample(examples: list[ForecastExample], cap: int | None, seed: int) -> list[ForecastExample]:
    if cap is not None and cap <= 0:
        raise ValueError("question caps must be positive or None")
    if cap is None or len(examples) <= cap:
        return list(examples)
    return random.Random(seed).sample(examples, cap)


def load_prophet_arena_split(
    csv_path: str | Path,
    *,
    split_time: datetime,
    max_train_questions: int | None = DEFAULT_MAX_TRAIN_QUESTIONS,
    max_validation_questions: int | None = DEFAULT_MAX_VALIDATION_QUESTIONS,
    seed: int = 0,
) -> ForecastSplit:
    """Split events around a boundary without sharing events or future labels."""
    if split_time.tzinfo is None:
        raise ValueError("split_time must be timezone-aware")
    split_time = split_time.astimezone(UTC)
    examples = load_prophet_arena_examples(csv_path)

    # Keep every market from an event on the same side of the split. Training
    # labels come only from events whose markets all closed before validation
    # begins; events already open at the boundary form a gap.
    event_starts: dict[str, datetime] = {}
    event_closes: dict[str, datetime] = {}
    for example in examples:
        event_starts[example.event_ticker] = min(
            event_starts.get(example.event_ticker, example.snapshot_time),
            example.snapshot_time,
        )
        event_closes[example.event_ticker] = max(
            event_closes.get(example.event_ticker, example.close_time),
            example.close_time,
        )

    train_event_tickers = {
        event_ticker for event_ticker, close_time in event_closes.items() if close_time < split_time
    }
    validation_event_tickers = {
        event_ticker
        for event_ticker, snapshot_time in event_starts.items()
        if snapshot_time >= split_time
    }
    crossing_event_tickers = set(event_starts) - train_event_tickers - validation_event_tickers
    train_candidates = [
        example for example in examples if example.event_ticker in train_event_tickers
    ]
    validation_candidates = [
        example for example in examples if example.event_ticker in validation_event_tickers
    ]
    excluded = len(crossing_event_tickers)
    train_events = {example.event_ticker for example in train_candidates}
    validation_events = {example.event_ticker for example in validation_candidates}
    if train_events & validation_events:
        raise ValueError("an event appears in both training and validation")

    train = _sample(train_candidates, max_train_questions, seed)
    random.Random(seed).shuffle(train)
    validation = _sample(validation_candidates, max_validation_questions, seed + 1)
    validation.sort(key=lambda item: (item.snapshot_time, item.event_ticker, item.market))
    if not train or not validation:
        raise ValueError(
            f"empty temporal split: {len(train)} train and {len(validation)} validation"
        )
    return ForecastSplit(tuple(train), tuple(validation), split_time, excluded)


def _yes_rate(examples: Sequence[ForecastExample]) -> float:
    return sum(example.outcome for example in examples) / len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--split-date",
        type=parse_utc_datetime,
        default=parse_utc_datetime(DEFAULT_SPLIT_DATE, "default split date"),
    )
    parser.add_argument("--max-train-questions", type=int, default=DEFAULT_MAX_TRAIN_QUESTIONS)
    parser.add_argument(
        "--max-validation-questions", type=int, default=DEFAULT_MAX_VALIDATION_QUESTIONS
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    csv_path = (
        Path(args.data_path).expanduser()
        if args.data_path is not None
        else fetch_prophet_arena(args.revision, args.cache_dir)
    )
    split = load_prophet_arena_split(
        csv_path,
        split_time=args.split_date,
        max_train_questions=args.max_train_questions,
        max_validation_questions=args.max_validation_questions,
        seed=args.seed,
    )
    print(f"dataset: {csv_path}")
    print(f"split date: {split.split_time.date().isoformat()}")
    print(f"train: {len(split.train)} examples, yes rate {_yes_rate(split.train):.3f}")
    print(
        f"validation: {len(split.validation)} examples, yes rate {_yes_rate(split.validation):.3f}"
    )
    print(f"excluded crossing events: {split.excluded_crossing_events}")


if __name__ == "__main__":
    main()
