#!/usr/bin/env python3
# ruff: noqa: E402
"""Reconnect to a running RLOR trainer.

**Reference example only** — shows how to reconnect to an already-running
trainer job and resume training with a (potentially different) learning rate.
Does NOT run real training (no data, no loss function, no forward/backward).
Plug in your own data pipeline and loss to make it functional.

The trainer job stays alive on the server even if your Python client
disconnects. Use ``reconnect_and_wait()`` to get back to it, then create
a new ``ReconnectableClient`` on the same endpoint.

Usage:
    export FIREWORKS_API_KEY=...

    python reconnect_and_adjust_lr.py \
        --job-id <policy-job-id> \
        --base-model accounts/fireworks/models/qwen3-8b \
        --new-lr 5e-6

    # Also reconnect a reference trainer:
    python reconnect_and_adjust_lr.py \
        --job-id <policy-job-id> \
        --ref-job-id <ref-job-id> \
        --base-model accounts/fireworks/models/qwen3-8b \
        --new-lr 5e-6
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_COOKBOOK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

import tinker
from fireworks.training.sdk import TrainerJobManager
from training.utils import DEFAULT_ADAM, ReconnectableClient

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Reconnect to a running trainer and resume with a new learning rate."
    )
    parser.add_argument("--job-id", required=True, help="Policy RLOR trainer job ID")
    parser.add_argument("--ref-job-id", default=None, help="Reference RLOR trainer job ID (optional)")
    parser.add_argument("--base-model", required=True, help="Base model resource name")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--base-url", default=None, help="Fireworks API base URL")
    parser.add_argument("--new-lr", type=float, required=True, help="Learning rate to use after reconnect")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = args.base_url or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    # -- Reconnect to existing trainer(s) ----------------------------------

    logger.info("Reconnecting to policy trainer %s ...", args.job_id)
    policy_ep = mgr.reconnect_and_wait(args.job_id, timeout_s=300)
    logger.info("Policy connected: %s", policy_ep.base_url)

    policy = ReconnectableClient(
        mgr,
        policy_ep.job_id,
        args.base_model,
        args.lora_rank,
        fw_api_key=api_key,
        endpoint=policy_ep,
    )

    reference = None
    if args.ref_job_id:
        logger.info("Reconnecting to reference trainer %s ...", args.ref_job_id)
        ref_ep = mgr.reconnect_and_wait(args.ref_job_id, timeout_s=300)
        logger.info("Reference connected: %s", ref_ep.base_url)
        reference = ReconnectableClient(
            mgr,
            ref_ep.job_id,
            args.base_model,
            args.lora_rank,
            fw_api_key=api_key,
            endpoint=ref_ep,
        )

    # -- Ready to train ----------------------------------------------------
    # From here you can resume your training loop. Pass the new LR via
    # AdamParams on each optim_step() call — the trainer uses whatever
    # LR you give it.

    adam_params = tinker.AdamParams(learning_rate=args.new_lr, **DEFAULT_ADAM)
    logger.info("Ready. LR=%.2e. Trainer %s is still running.", args.new_lr, args.job_id)

    # Uncomment to actually run training steps:
    # for step in range(num_steps):
    #     policy.forward_backward_custom(datums, loss_fn)
    #     policy.optim_step(adam_params)

    if reference:
        logger.info("Reference trainer %s is also connected.", args.ref_job_id)


if __name__ == "__main__":
    main()
