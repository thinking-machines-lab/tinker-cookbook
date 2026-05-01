#!/usr/bin/env python3
# ruff: noqa: E402
"""Promote a sampler checkpoint to a deployable Fireworks model.

Queries the control plane for the trainer job's checkpoints
(``FireworksClient.list_checkpoints``), picks the newest promotable
row (or a specific one if ``--checkpoint-name`` is given), and calls
the promotion API. No temporary trainer is needed — promotion is a
lightweight metadata + file-copy operation that works even after the
trainer job has been deleted.

Usage:
    export FIREWORKS_API_KEY=...

    # Promote the newest promotable checkpoint on a job:
    python promote_checkpoint.py \\
        --job-id <trainer-job-id> \\
        --base-model accounts/fireworks/models/qwen3-8b

    # Promote a specific checkpoint. ``step-50`` matches both an exact
    # row stored as ``step-50`` and one stored as ``step-50-a1b2c3d4``
    # (sampler rows get an 8-hex session suffix appended server-side).
    python promote_checkpoint.py \\
        --job-id <trainer-job-id> \\
        --checkpoint-name step-50 \\
        --base-model accounts/fireworks/models/qwen3-8b

    # Override the auto-generated output model ID:
    python promote_checkpoint.py \\
        --job-id <trainer-job-id> \\
        --base-model accounts/fireworks/models/qwen3-8b \\
        --output-model-id my-fine-tuned-qwen3-8b

    # Legacy jobs that used deployment-owned checkpoints:
    python promote_checkpoint.py \\
        --job-id <trainer-job-id> \\
        --base-model accounts/fireworks/models/qwen3-8b \\
        --hot-load-deployment-id my-deployment
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

_COOKBOOK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

from fireworks.training.sdk import FireworksClient, TrainerJobManager
from training.utils.checkpoints import _logical_name, _short_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass(frozen=True)
class PromoteConfig:
    job_id: str
    checkpoint_name: str | None
    base_model: str
    output_model_id: str | None
    hot_load_deployment_id: str | None


@dataclass(frozen=True)
class ResolvedCheckpoint:
    full_name: str             # 4-segment resource name
    short_name: str            # trailing checkpoint id (for log lines only)
    source_job_id: str         # for log lines only


def parse_args() -> PromoteConfig:
    parser = argparse.ArgumentParser(
        description="Promote a sampler checkpoint to a deployable Fireworks model",
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="RLOR trainer job ID that produced the checkpoint (the short ID, "
             "not the full resource name).",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Specific checkpoint name to promote. Matches both the exact "
             "name (e.g. 'step-50-a1b2c3d4') and the logical name "
             "(e.g. 'step-50' matches a row stored as 'step-50-a1b2c3d4'). "
             "Defaults to the newest promotable checkpoint.",
    )
    parser.add_argument(
        "--base-model",
        "--model",
        dest="base_model",
        required=True,
        help="Base model resource name for metadata inheritance "
             "(e.g. accounts/fireworks/models/qwen3-8b).",
    )
    parser.add_argument(
        "--output-model-id",
        default=None,
        help="Promoted model ID. Defaults to an auto-generated value.",
    )
    parser.add_argument(
        "--hot-load-deployment-id",
        default=None,
        help=(
            "[Legacy] Deployment ID for jobs from deployments that predate "
            "the stored-bucket-URL migration. Modern runs (both PER_TRAINER "
            "and PER_DEPLOYMENT bucket scopes) do not need this — the "
            "bucket URL is resolved server-side from the trainer's stored "
            "metadata. Omit for any run from cookbook >= 0.3.0."
        ),
    )
    args = parser.parse_args()
    return PromoteConfig(
        job_id=args.job_id,
        checkpoint_name=args.checkpoint_name,
        base_model=args.base_model,
        output_model_id=args.output_model_id,
        hot_load_deployment_id=args.hot_load_deployment_id,
    )


def _resolve_checkpoint(
    fw_client: FireworksClient,
    job_id: str,
    *,
    checkpoint_name: str | None,
) -> ResolvedCheckpoint:
    rows = fw_client.list_checkpoints(job_id)
    promotable = sorted(
        (r for r in rows if r.get("promotable")),
        key=lambda r: r.get("createTime", ""),
        reverse=True,
    )
    if not promotable:
        raise SystemExit(
            f"ERROR: no promotable checkpoints found on trainer job '{job_id}'.\n"
            "  Run `python list_checkpoints.py --job-id <job-id>` to see all rows.\n"
            "  Promotable rows have checkpointType in (INFERENCE_BASE, INFERENCE_LORA)."
        )

    if checkpoint_name is None:
        chosen = promotable[0]
    else:
        target_logical = _logical_name(checkpoint_name)
        matches = [
            r for r in promotable
            if _short_name(r["name"]) == checkpoint_name
            or _logical_name(_short_name(r["name"])) == target_logical
        ]
        if not matches:
            raise SystemExit(
                f"ERROR: no promotable checkpoint named '{checkpoint_name}' on job '{job_id}'."
            )
        # Newest among matches (handles multiple sampler saves at the same logical name).
        chosen = matches[0]

    return ResolvedCheckpoint(
        full_name=chosen["name"],
        short_name=_short_name(chosen["name"]),
        source_job_id=job_id,
    )


def _sanitize_resource_id(value: str, *, default: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or default


def _default_output_model_id(base_model: str, checkpoint_name: str) -> str:
    model_short = base_model.rsplit("/", 1)[-1]
    suffix = f"{checkpoint_name}-{int(time.time()) % 100000}"
    return _sanitize_resource_id(
        f"{model_short}-promote-{suffix}", default="promoted-model"
    )[:63].rstrip("-")


def main() -> None:
    cfg = parse_args()

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    fw_client = FireworksClient(api_key=api_key, base_url=base_url)
    trainer_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    resolved = _resolve_checkpoint(
        fw_client,
        cfg.job_id,
        checkpoint_name=cfg.checkpoint_name,
    )

    output_model_id = cfg.output_model_id or _default_output_model_id(
        cfg.base_model, resolved.short_name,
    )

    logger.info("Checkpoint:      %s", resolved.full_name)
    logger.info("Base model:      %s", cfg.base_model)
    logger.info("Output model ID: %s", output_model_id)
    if cfg.hot_load_deployment_id:
        logger.warning(
            "--hot-load-deployment-id is only needed for deployments that "
            "predate the stored-bucket-URL migration. Runs from cookbook "
            ">= 0.3.0 (both PER_TRAINER and PER_DEPLOYMENT bucket scopes) "
            "resolve the bucket URL server-side and should omit this flag."
        )
        logger.info("Deployment ID:   %s (legacy)", cfg.hot_load_deployment_id)

    # Pass the 4-segment resource name verbatim — no manual disassembly
    # into (job_id, checkpoint_id). See the public docs page on saving
    # and loading for the full promote API contract.
    model = trainer_mgr.promote_checkpoint(
        name=resolved.full_name,
        output_model_id=output_model_id,
        base_model=cfg.base_model,
        hot_load_deployment_id=cfg.hot_load_deployment_id,
    )

    logger.info(
        "Promoted model: %s",
        model.get("name", f"accounts/{trainer_mgr.account_id}/models/{output_model_id}"),
    )
    logger.info(
        "Model state=%s kind=%s",
        model.get("state", "UNKNOWN"),
        model.get("kind", "UNKNOWN"),
    )


if __name__ == "__main__":
    main()
