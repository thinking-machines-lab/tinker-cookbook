"""Persist Tinker audit log entries to disk on a recurring schedule.

The Tinker audit log endpoint returns events for a single UTC day at a time.
To build a durable archive of your organization's audit events, you need to
poll the endpoint regularly and store the results yourself.

This script does that. It writes one JSON Lines file per UTC day to a local
output directory:

    audit_logs/
      audit_log_2026-05-11.jsonl  # complete day, written once and skipped after
      audit_log_2026-05-12.jsonl  # complete day, written once and skipped after
      audit_log_2026-05-13.jsonl  # in-progress day, re-fetched on every tick

Days other than "today (UTC)" are treated as immutable once written. Today's
file is rewritten on every poll because new events are still arriving.

Requirements:
- The caller's API key must have the `VIEW_AUDIT_LOG` capability
  (tinker-admin RBAC role).

Usage:

    # One-shot: fetch today's audit log and exit.
    python -m tinker_cookbook.scripts.save_audit_log --once

    # Backfill the last 7 days, then poll every hour.
    python -m tinker_cookbook.scripts.save_audit_log \\
        --output-dir ./audit_logs \\
        --backfill-days 7 \\
        --interval-seconds 3600
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import tinker
from tinker.types import AuditLogEntry, AuditLogResponse

if TYPE_CHECKING:
    from tinker.lib.public_interfaces.rest_client import RestClient

logger = logging.getLogger(__name__)


def _file_for_day(output_dir: Path, day: date) -> Path:
    return output_dir / f"audit_log_{day.isoformat()}.jsonl"


def _entry_to_jsonl(entry: AuditLogEntry) -> str:
    return json.dumps(
        {
            "timestamp": entry.timestamp.isoformat(),
            "event": entry.event,
            "model_id": entry.model_id,
            "tinker_path": entry.tinker_path,
            "purpose": entry.purpose,
        },
        sort_keys=True,
    )


def _write_atomic(path: Path, response: AuditLogResponse) -> None:
    """Write all entries to `path` atomically via a temp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for entry in response.entries:
            f.write(_entry_to_jsonl(entry))
            f.write("\n")
    os.replace(tmp, path)


def fetch_and_save_day(
    rest_client: RestClient,
    output_dir: Path,
    day: date,
) -> int:
    """Fetch one day's audit log and write it to disk. Returns entry count."""
    response = rest_client.get_audit_log(day=day).result()
    path = _file_for_day(output_dir, day)
    _write_atomic(path, response)
    return len(response.entries)


def days_to_fetch(
    output_dir: Path,
    today: date,
    backfill_days: int,
) -> list[date]:
    """Pick which days to fetch on this tick.

    - Today is always included (it is still in progress, so we always refresh it).
    - For the prior `backfill_days` days, include any day whose file is missing.
      Days that already have a file on disk are considered final and skipped.
    """
    days: list[date] = [today]
    for offset in range(1, backfill_days + 1):
        day = today - timedelta(days=offset)
        if not _file_for_day(output_dir, day).exists():
            days.append(day)
    return days


def run_once(
    rest_client: RestClient,
    output_dir: Path,
    backfill_days: int,
) -> None:
    today = datetime.now(UTC).date()
    for day in days_to_fetch(output_dir, today, backfill_days):
        n = fetch_and_save_day(rest_client, output_dir, day)
        logger.info("Wrote %d audit entries for %s", n, day.isoformat())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audit_logs"),
        help="Directory to write per-day JSONL files into. Default: ./audit_logs",
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=7,
        help=(
            "How many days before today to check for missing files on each "
            "tick. Days already on disk are skipped. Default: 7."
        ),
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=3600,
        help="Polling interval in seconds between ticks. Default: 3600 (1 hour).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch once and exit instead of polling on an interval.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    if args.once:
        run_once(rest_client, args.output_dir, args.backfill_days)
        return

    while True:
        try:
            run_once(rest_client, args.output_dir, args.backfill_days)
        except Exception:
            logger.exception("Audit log fetch failed; will retry next tick")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
