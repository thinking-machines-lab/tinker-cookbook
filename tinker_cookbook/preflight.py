"""Fail-closed validation for bounded Tinker training runs.

This module reads the artifacts that cookbook training loops already write. It does
not call the Tinker API or start a training run. Use it after a small, explicit canary
run to decide whether that run produced the evidence required for a larger launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Literal, cast

from tinker_cookbook.checkpoint_utils import CheckpointRecord
from tinker_cookbook.stores.storage import Storage, storage_from_uri

CheckStatus = Literal["passed", "failed"]


@dataclass(frozen=True)
class PreflightCheck:
    """One evidence check in a preflight report."""

    name: str
    status: CheckStatus
    summary: str
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable form of the check."""
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
        }


@dataclass(frozen=True)
class PreflightConfig:
    """Evidence requirements for one bounded training run.

    ``log_path`` may be a local path or another URI supported by
    :func:`tinker_cookbook.stores.storage.storage_from_uri`.
    """

    log_path: str
    required_metric_keys: tuple[str, ...] = ()
    require_state_checkpoint: bool = True
    require_sampler_checkpoint: bool = False
    require_final_checkpoint: bool = True
    minimum_metric_step: int | None = None

    def __post_init__(self) -> None:
        if not self.log_path.strip():
            raise ValueError("log_path must not be empty")
        if any(not key.strip() for key in self.required_metric_keys):
            raise ValueError("required_metric_keys must not contain empty values")
        if self.minimum_metric_step is not None and (
            isinstance(self.minimum_metric_step, bool)
            or not isinstance(self.minimum_metric_step, int)
            or self.minimum_metric_step < 0
        ):
            raise ValueError("minimum_metric_step must be a nonnegative integer")


@dataclass(frozen=True)
class PreflightSnapshot:
    """Artifact prefixes captured before a command appends training evidence."""

    log_path: str
    metric_records: int
    checkpoint_records: int
    metric_prefix_bytes: int
    checkpoint_prefix_bytes: int
    metric_prefix_sha256: str
    checkpoint_prefix_sha256: str


@dataclass(frozen=True)
class PreflightReport:
    """The launch decision and supporting checks for one run."""

    log_path: str
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        """Return ``True`` only when every declared check passed."""
        return all(check.status == "passed" for check in self.checks)

    def failure_summary(self) -> str:
        """Return failed check summaries as one stable message."""
        return "; ".join(
            f"{check.name}: {check.summary}" for check in self.checks if check.status == "failed"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable form of the report."""
        return {
            "status": "passed" if self.passed else "failed",
            "log_path": self.log_path,
            "checks": [check.to_dict() for check in self.checks],
        }


def _non_finite_values(value: object, path: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        failures.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            failures.extend(_non_finite_values(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_non_finite_values(item, f"{path}[{index}]"))
    return failures


def _read_or_empty(storage: Storage, path: str) -> bytes:
    try:
        return storage.read(path)
    except FileNotFoundError:
        return b""


def _jsonl_lines(raw: bytes) -> list[tuple[int, bytes]]:
    return [
        (line_number, line)
        for line_number, line in enumerate(raw.splitlines(), start=1)
        if line.strip()
    ]


def capture_preflight_snapshot(log_path: str) -> PreflightSnapshot:
    """Capture append-only artifact boundaries before running a recipe command."""
    if not log_path.strip():
        raise ValueError("log_path must not be empty")
    storage = storage_from_uri(log_path)
    metric_prefix = _read_or_empty(storage, "metrics.jsonl")
    checkpoint_prefix = _read_or_empty(storage, "checkpoints.jsonl")
    return PreflightSnapshot(
        log_path=log_path,
        metric_records=len(_jsonl_lines(metric_prefix)),
        checkpoint_records=len(_jsonl_lines(checkpoint_prefix)),
        metric_prefix_bytes=len(metric_prefix),
        checkpoint_prefix_bytes=len(checkpoint_prefix),
        metric_prefix_sha256=hashlib.sha256(metric_prefix).hexdigest(),
        checkpoint_prefix_sha256=hashlib.sha256(checkpoint_prefix).hexdigest(),
    )


def _read_jsonl_strict(
    storage: Storage,
    path: str,
    *,
    start_record: int = 0,
    prefix_bytes: int = 0,
    prefix_sha256: str | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    raw = _read_or_empty(storage, path)
    if prefix_sha256 is not None:
        if len(raw) < prefix_bytes:
            return [], [
                f"{path}: artifact was truncated below the captured {prefix_bytes}-byte boundary"
            ]
        observed_prefix_sha256 = hashlib.sha256(raw[:prefix_bytes]).hexdigest()
        if observed_prefix_sha256 != prefix_sha256:
            return [], [f"{path}: artifact changed before the captured append boundary"]

    indexed_lines = _jsonl_lines(raw)
    if len(indexed_lines) < start_record:
        return [], [
            f"{path}: expected at least {start_record} prior record(s), found {len(indexed_lines)}"
        ]

    records: list[dict[str, object]] = []
    failures: list[str] = []
    for line_number, raw_line in indexed_lines[start_record:]:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            failures.append(f"{path}:{line_number}: invalid UTF-8 at byte {error.start}")
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as error:
            failures.append(f"{path}:{line_number}: invalid JSON at column {error.colno}")
            continue
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            failures.append(f"{path}:{line_number}: expected a JSON object")
            continue
        records.append(cast(dict[str, object], value))
    return records, failures


def _checkpoint_records(
    records: list[dict[str, object]], path: str, *, record_offset: int = 0
) -> tuple[list[CheckpointRecord], list[str]]:
    checkpoints: list[CheckpointRecord] = []
    failures: list[str] = []
    for index, record in enumerate(records, start=record_offset + 1):
        try:
            checkpoints.append(CheckpointRecord.from_dict(record))
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{path}:record[{index}]: {type(error).__name__}")
    return checkpoints, failures


def _failure_preview(failures: list[str]) -> str:
    preview = ", ".join(failures[:5])
    if len(failures) > 5:
        preview += f" and {len(failures) - 5} more"
    return preview


def _check_artifact_format(
    failures: list[str], *, metric_record_start: int, checkpoint_record_start: int
) -> PreflightCheck:
    if failures:
        return PreflightCheck(
            name="Artifact format",
            status="failed",
            summary=f"Malformed artifact data at {_failure_preview(failures)}.",
            details={
                "failures": failures,
                "metric_record_start": metric_record_start,
                "checkpoint_record_start": checkpoint_record_start,
            },
        )
    return PreflightCheck(
        name="Artifact format",
        status="passed",
        summary="Selected metric and checkpoint records are valid JSONL objects.",
        details={
            "metric_record_start": metric_record_start,
            "checkpoint_record_start": checkpoint_record_start,
        },
    )


def _check_metrics(
    metrics: list[dict[str, object]],
    required_keys: tuple[str, ...],
    minimum_step: int | None,
) -> PreflightCheck:
    if not metrics:
        return PreflightCheck(
            name="Training metrics",
            status="failed",
            summary="No metric records were found.",
        )

    required_key_set = tuple(sorted(set(required_keys)))
    qualifying_metrics = [
        (index, row)
        for index, row in enumerate(metrics)
        if minimum_step is None
        or (
            isinstance((step := row.get("step")), int)
            and not isinstance(step, bool)
            and step >= minimum_step
        )
    ]
    reached_minimum_step = minimum_step is None or bool(qualifying_metrics)
    missing_keys = (
        sorted(
            key for key in required_key_set if not any(key in row for _, row in qualifying_metrics)
        )
        if reached_minimum_step
        else []
    )
    invalid_required_values = [
        f"record[{index}].{key}"
        for index, row in qualifying_metrics
        for key in required_key_set
        if key in row and (isinstance(row[key], bool) or not isinstance(row[key], (int, float)))
    ]
    non_finite = [
        item
        for index, row in enumerate(metrics)
        for item in _non_finite_values(row, f"record[{index}]")
    ]
    observed_steps = [
        step
        for row in metrics
        if isinstance((step := row.get("step")), int) and not isinstance(step, bool)
    ]
    maximum_step = max(observed_steps, default=None)
    failures: list[str] = []
    if missing_keys:
        failures.append(f"missing required keys: {', '.join(missing_keys)}")
    if invalid_required_values:
        failures.append(
            f"required metric values are not numeric at {_failure_preview(invalid_required_values)}"
        )
    if non_finite:
        failures.append(f"non-finite values at {_failure_preview(non_finite)}")
    if not reached_minimum_step:
        failures.append(f"no metric record reached minimum step {minimum_step}")

    if failures:
        return PreflightCheck(
            name="Training metrics",
            status="failed",
            summary="; ".join(failures) + ".",
            details={
                "records": len(metrics),
                "missing_keys": missing_keys,
                "invalid_required_values": invalid_required_values,
                "non_finite_values": non_finite,
                "minimum_step": minimum_step,
                "maximum_step": maximum_step,
            },
        )

    key_summary = f" Required keys: {', '.join(required_key_set)}." if required_key_set else ""
    evidence_summary = (
        "declared required metrics are numeric and all numeric values are finite"
        if required_key_set
        else "all numeric values are finite"
    )
    return PreflightCheck(
        name="Training metrics",
        status="passed",
        summary=(f"Found {len(metrics)} selected record(s); {evidence_summary}.{key_summary}"),
        details={
            "records": len(metrics),
            "required_keys": list(required_key_set),
            "minimum_step": minimum_step,
            "maximum_step": maximum_step,
        },
    )


def _is_final_checkpoint(record: CheckpointRecord) -> bool:
    # Older cookbook loops used the name "final" before they wrote a final flag.
    return record.final is True or (record.final is None and record.name == "final")


def _checkpoint_with_key(
    records: list[CheckpointRecord],
    key: Literal["state_path", "sampler_path"],
    require_final: bool,
) -> CheckpointRecord | None:
    eligible = [
        record
        for record in records
        if record.has(key) and (not require_final or _is_final_checkpoint(record))
    ]
    return eligible[-1] if eligible else None


def _check_checkpoint(
    records: list[CheckpointRecord],
    *,
    key: Literal["state_path", "sampler_path"],
    label: str,
    require_final: bool,
) -> PreflightCheck:
    checkpoint = _checkpoint_with_key(records, key, require_final)
    if checkpoint is None:
        final_label = " final" if require_final else ""
        return PreflightCheck(
            name=label,
            status="failed",
            summary=f"No{final_label} checkpoint with {key} was found.",
            details={"records": len(records), "required_key": key},
        )

    path = checkpoint.get(key)
    if not isinstance(checkpoint.name, str) or not checkpoint.name.strip():
        return PreflightCheck(
            name=label,
            status="failed",
            summary="Checkpoint record has an invalid name.",
            details={"required_key": key, "reason": "expected a non-empty string name"},
        )
    if (
        not isinstance(path, str)
        or not path.startswith("tinker://")
        or not path.removeprefix("tinker://").strip()
    ):
        return PreflightCheck(
            name=label,
            status="failed",
            summary=f"Checkpoint {checkpoint.name} has an invalid {key}.",
            details={
                "name": checkpoint.name,
                "required_key": key,
                "reason": "expected a non-empty tinker://-prefixed string",
            },
        )
    return PreflightCheck(
        name=label,
        status="passed",
        summary=(
            f"Found checkpoint record {checkpoint.name} with a non-empty tinker://-prefixed string."
        ),
        details={"name": checkpoint.name, "path": path, "final": _is_final_checkpoint(checkpoint)},
    )


def validate_training_run(
    config: PreflightConfig, *, after: PreflightSnapshot | None = None
) -> PreflightReport:
    """Validate selected metrics and checkpoint records without calling Tinker."""
    if after is not None and after.log_path != config.log_path:
        raise ValueError("snapshot log_path must match config log_path")

    metric_record_start = after.metric_records if after is not None else 0
    checkpoint_record_start = after.checkpoint_records if after is not None else 0
    storage = storage_from_uri(config.log_path)
    metrics, metric_format_failures = _read_jsonl_strict(
        storage,
        "metrics.jsonl",
        start_record=metric_record_start,
        prefix_bytes=after.metric_prefix_bytes if after is not None else 0,
        prefix_sha256=after.metric_prefix_sha256 if after is not None else None,
    )
    raw_checkpoints, checkpoint_format_failures = _read_jsonl_strict(
        storage,
        "checkpoints.jsonl",
        start_record=checkpoint_record_start,
        prefix_bytes=after.checkpoint_prefix_bytes if after is not None else 0,
        prefix_sha256=after.checkpoint_prefix_sha256 if after is not None else None,
    )
    checkpoint_records, checkpoint_record_failures = _checkpoint_records(
        raw_checkpoints, "checkpoints.jsonl", record_offset=checkpoint_record_start
    )

    checks = [
        _check_artifact_format(
            metric_format_failures + checkpoint_format_failures + checkpoint_record_failures,
            metric_record_start=metric_record_start,
            checkpoint_record_start=checkpoint_record_start,
        ),
        _check_metrics(metrics, config.required_metric_keys, config.minimum_metric_step),
    ]
    if config.require_state_checkpoint:
        checks.append(
            _check_checkpoint(
                checkpoint_records,
                key="state_path",
                label="Training-state checkpoint",
                require_final=config.require_final_checkpoint,
            )
        )
    if config.require_sampler_checkpoint:
        checks.append(
            _check_checkpoint(
                checkpoint_records,
                key="sampler_path",
                label="Sampler checkpoint",
                require_final=config.require_final_checkpoint,
            )
        )

    return PreflightReport(log_path=config.log_path, checks=tuple(checks))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the artifacts from a bounded Tinker training run."
    )
    parser.add_argument("log_path", help="Training log path or supported storage URI.")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        dest="required_metric_keys",
        help="Metric key that must exist. Repeat for more than one key.",
    )
    parser.add_argument(
        "--state-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a training-state checkpoint (default: true).",
    )
    parser.add_argument(
        "--sampler-checkpoint",
        action="store_true",
        help="Require a recorded non-empty tinker://-prefixed sampler checkpoint string.",
    )
    parser.add_argument(
        "--final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require requested checkpoint types on a final record (default: true).",
    )
    parser.add_argument(
        "--minimum-step",
        type=_nonnegative_int,
        default=None,
        dest="minimum_metric_step",
        help="Require at least one metric record at or above this step.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Run the command-line validator and return a process exit code."""
    args = _build_parser().parse_args(argv)
    report = validate_training_run(
        PreflightConfig(
            log_path=args.log_path,
            required_metric_keys=tuple(args.required_metric_keys),
            require_state_checkpoint=args.state_checkpoint,
            require_sampler_checkpoint=args.sampler_checkpoint,
            require_final_checkpoint=args.final_checkpoint,
            minimum_metric_step=args.minimum_metric_step,
        )
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        label = "PASS" if report.passed else "FAIL"
        print(f"TINKER PREFLIGHT: {label}")
        for check in report.checks:
            print(f"{check.status.upper():6} {check.name}: {check.summary}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
