#!/usr/bin/env python3
# ruff: noqa: E402
"""List checkpoints the server knows about for an RLOR trainer job.

Thin wrapper over ``FireworksClient.list_checkpoints(job_id)``. Lists the
authoritative set of checkpoints for a trainer — sampler + DCP — including
ones inherited from a predecessor trainer via hotload. Works for dead
trainers (completed, failed, cancelled) — only the DB record and GCS
blobs need to exist.

Usage:
    export FIREWORKS_API_KEY=...

    # All checkpoints on a trainer:
    python list_checkpoints.py --job-id <job-id>

    # Only rows the server will accept for promote, newest first:
    python list_checkpoints.py --job-id <job-id> --promotable-only

    # Machine-readable:
    python list_checkpoints.py --job-id <job-id> --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

_COOKBOOK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

from fireworks.training.sdk import FireworksClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List checkpoints for an RLOR trainer job.",
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="RLOR trainer job ID (the short ID, not the full resource name).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Max rows per request (the SDK auto-paginates). Default: 200.",
    )
    parser.add_argument(
        "--promotable-only",
        action="store_true",
        help="Filter to rows with promotable=true.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit raw JSON to stdout instead of a human table.",
    )
    return parser.parse_args()


def _sort_newest_first(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r.get("createTime", ""), reverse=True)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no checkpoints)")
        return
    # Widest enum value is CHECKPOINT_TYPE_INFERENCE_ARC_V2 (32 chars).
    header = f"{'name':<40}  {'createTime':<26}  {'type':<35}  promotable"
    print(header)
    print("-" * len(header))
    for r in rows:
        name = r.get("name", "").rsplit("/", 1)[-1]
        created = r.get("createTime", "")
        ckpt_type = r.get("checkpointType", "")
        promo = r.get("promotable", False)
        print(f"{name:<40}  {created:<26}  {ckpt_type:<35}  {promo}")


def main() -> None:
    args = parse_args()

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    client = FireworksClient(api_key=api_key, base_url=base_url)

    rows = client.list_checkpoints(args.job_id, page_size=args.page_size)
    if args.promotable_only:
        rows = [r for r in rows if r.get("promotable")]
    rows = _sort_newest_first(rows)

    if args.as_json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
