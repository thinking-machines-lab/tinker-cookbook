#!/usr/bin/env python3
"""Verify train-inference logprob alignment.

Creates an inference deployment and a policy trainer (plus an optional
forward-only reference trainer) from training shapes, samples completions,
runs training forward passes, and compares per-token logprobs on completion
tokens.

Usage:
    export FIREWORKS_API_KEY=...

    # Policy-only (no reference KL):
    python verify.py \\
        --training-shape ts-qwen3-30b-a3b-128k \\
        --base-model accounts/fireworks/models/qwen3-30b-a3b \\
        --tokenizer Qwen/Qwen3-30B-A3B

    # With reference model for KL divergence:
    python verify.py \\
        --training-shape ts-qwen3-30b-a3b-128k \\
        --ref-training-shape ts-qwen3-30b-a3b-128k-ref \\
        --base-model accounts/fireworks/models/qwen3-30b-a3b \\
        --tokenizer Qwen/Qwen3-30B-A3B
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor

import torch
import tinker
import transformers

_COOKBOOK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

from fireworks.training.sdk import (
    TrainerJobManager,
    DeploymentManager,
    DeploymentSampler,
)
from training.utils import (
    InfraConfig,
    DeployConfig,
    ReconnectableClient,
    create_trainer_job,
    setup_deployment,
    load_jsonl_dataset,
    prepare_sampling_messages,
)
from training.utils.infra import ResourceCleanup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("tinker").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_DATASET = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"


# ---------------------------------------------------------------------------
# Logprob comparison (completion tokens only)
# ---------------------------------------------------------------------------


def compare_logprobs(
    training_lps: list[float],
    inference_lps: list[float],
    prompt_len: int,
    label: str = "",
) -> dict:
    """Compare training vs inference logprobs on completion tokens.

    Both inputs are expected to be in training-aligned format: N-1 entries
    for an N-token sequence, where entry *i* is log P(token[i+1] | token[0:i+1]).
    Only positions >= ``prompt_len - 1`` (the completion region) contribute
    to aggregate metrics.

    Returns a dict of summary statistics including KL divergence estimators.
    """
    response_start = max(0, prompt_len - 1)

    diffs, log_ratios = [], []
    for j in range(response_start, min(len(training_lps), len(inference_lps))):
        d = abs(training_lps[j] - inference_lps[j])
        diffs.append(d)
        log_ratios.append(inference_lps[j] - training_lps[j])

    if not diffs:
        return {"label": label, "count": 0}

    t_diffs = torch.tensor(diffs, dtype=torch.float32)
    t_lr = torch.tensor(log_ratios, dtype=torch.float32)

    metrics = {
        "label": label,
        "count": len(diffs),
        "mean_diff": t_diffs.mean().item(),
        "max_diff": t_diffs.max().item(),
        "k1": t_lr.mean().item(),
        "k2": (0.5 * t_lr**2).mean().item(),
        "k3": (t_lr.exp() - 1 - t_lr).mean().item(),
    }

    logger.info(
        "[%s] completion tokens=%d | mean_diff=%.6f max_diff=%.6f | "
        "k1=%.6f k2=%.6f k3=%.6f",
        label, metrics["count"], metrics["mean_diff"], metrics["max_diff"],
        metrics["k1"], metrics["k2"], metrics["k3"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Verify train-inference logprob alignment using cookbook utilities",
    )
    p.add_argument(
        "--training-shape", required=True,
        help="Policy training shape ID (e.g. ts-qwen3-30b-a3b-128k)",
    )
    p.add_argument(
        "--ref-training-shape", default=None,
        help=(
            "Forward-only reference training shape ID. "
            "When set, a second trainer is created to provide reference "
            "logprobs for KL comparison."
        ),
    )
    p.add_argument(
        "--shape-account", default=None,
        help="Account that owns the training shapes (for cross-account resolution)",
    )
    p.add_argument(
        "--base-model", required=True,
        help="Base model resource name (e.g. accounts/fireworks/models/qwen3-30b-a3b)",
    )
    p.add_argument(
        "--tokenizer", required=True,
        help="HuggingFace tokenizer name (e.g. Qwen/Qwen3-30B-A3B)",
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="JSONL dataset path or URL")
    p.add_argument("--max-rows", type=int, default=3, help="Number of prompts to verify")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens per completion")
    p.add_argument("--deployment-id", default=None, help="Reuse an existing deployment")
    p.add_argument("--log-dir", default="./verify_logs", help="Output directory for logs and results")
    p.add_argument("--cleanup", action="store_true", help="Delete resources on exit")
    return p.parse_args()


def main():
    args = parse_args()
    use_reference = args.ref_training_shape is not None

    os.makedirs(args.log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.log_dir, "verify.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    shape_account = args.shape_account
    if shape_account:
        logger.info("Resolving training shapes under account '%s'", shape_account)
        shape_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
        shape_mgr._account_id = shape_account
    else:
        shape_mgr = rlor_mgr

    # -- Resolve training shapes ----------------------------------------------

    logger.info("Resolving policy training shape: %s", args.training_shape)
    profile = shape_mgr.resolve_training_profile(args.training_shape)
    logger.info("  version:          %s", profile.training_shape_version)
    logger.info("  deployment shape: %s", profile.deployment_shape)
    logger.info("  max_seq_len:      %d", profile.max_supported_context_length)
    logger.info("  accelerator:      %s x%d", profile.accelerator_type, profile.accelerator_count)

    max_seq_len = profile.max_supported_context_length
    infra = InfraConfig(training_shape_id=args.training_shape)

    ref_profile = None
    ref_infra = None
    if use_reference:
        logger.info("Resolving reference training shape: %s", args.ref_training_shape)
        ref_profile = shape_mgr.resolve_training_profile(args.ref_training_shape)
        ref_infra = InfraConfig(training_shape_id=args.ref_training_shape)
        logger.info("  version:          %s", ref_profile.training_shape_version)

    # -- Create resources in parallel -----------------------------------------

    dep_id = args.deployment_id or f"verify-{args.base_model.split('/')[-1]}-{int(time.time())}"
    deploy_cfg = DeployConfig(
        deployment_id=dep_id,
        deployment_shape=profile.deployment_shape,
        tokenizer_model=args.tokenizer,
    )

    policy_job_id = None
    ref_job_id = None

    with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        n_workers = 3 if use_reference else 2
        logger.info(
            "\n[1/4] Setting up deployment + trainers in parallel (%s)...",
            "policy + reference" if use_reference else "policy only",
        )

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            dep_fut = pool.submit(
                setup_deployment, deploy_mgr, deploy_cfg, args.base_model, infra,
            )
            pol_fut = pool.submit(
                create_trainer_job, rlor_mgr,
                base_model=args.base_model, infra=infra, profile=profile,
                lora_rank=0, max_seq_len=max_seq_len,
                display_name="verify-policy",
            )
            ref_fut = None
            if use_reference:
                ref_fut = pool.submit(
                    create_trainer_job, rlor_mgr,
                    base_model=args.base_model, infra=ref_infra, profile=ref_profile,
                    lora_rank=0, max_seq_len=max_seq_len,
                    display_name="verify-reference", forward_only=True,
                )

            dep_info = dep_fut.result()
            if args.cleanup and not args.deployment_id:
                cleanup.deployment(dep_id, action="scale_to_zero")

            pol_ep = pol_fut.result()
            policy_job_id = pol_ep.job_id
            if args.cleanup:
                cleanup.trainer(policy_job_id)
            logger.info("  Deployment ready: %s", dep_info.name)
            logger.info("  Policy trainer:   %s", pol_ep.job_id)

            ref_ep = None
            if ref_fut is not None:
                ref_ep = ref_fut.result()
                ref_job_id = ref_ep.job_id
                if args.cleanup:
                    cleanup.trainer(ref_job_id)
                logger.info("  Reference trainer: %s", ref_ep.job_id)

        policy = ReconnectableClient(
            rlor_mgr, pol_ep.job_id, args.base_model, lora_rank=0, fw_api_key=api_key,
        )
        if hasattr(policy, "close"):
            stack.callback(policy.close)
        reference = None
        if ref_ep is not None:
            reference = ReconnectableClient(
                rlor_mgr, ref_ep.job_id, args.base_model, lora_rank=0, fw_api_key=api_key,
            )
            if hasattr(reference, "close"):
                stack.callback(reference.close)

        inference_model = dep_info.inference_model or args.base_model
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True,
        )
        sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=inference_model, api_key=api_key, tokenizer=tokenizer,
        )

        # -- Warmup -----------------------------------------------------------

        logger.info("\n[2/4] Warming up deployment...")
        deploy_mgr.warmup(inference_model, max_retries=30)

        # -- Load dataset -----------------------------------------------------

        logger.info("\n[3/4] Loading dataset (%d rows)...", args.max_rows)
        dataset = load_jsonl_dataset(args.dataset, args.max_rows)
        if not dataset:
            raise RuntimeError("No data loaded from dataset")

        # -- Verification loop ------------------------------------------------

        logger.info(
            "\n[4/4] Verifying logprobs (%d prompts, greedy, completion-only, ref=%s)...",
            len(dataset), "yes" if use_reference else "no",
        )
        all_metrics: list[dict] = []

        for pidx, row in enumerate(dataset):
            messages = row.get("messages", [])
            input_msgs = prepare_sampling_messages(messages)
            if not input_msgs:
                continue

            preview = input_msgs[-1].get("content", "")[:60]
            logger.info(
                "\n--- Prompt %d/%d: %s... ---", pidx + 1, len(dataset), preview,
            )

            # Sample from deployment
            try:
                sampled = sampler.sample_with_tokens(
                    messages=input_msgs, n=1,
                    max_tokens=args.max_new_tokens, temperature=0.0,
                    logprobs=True, echo=True, prompt_cache_max_len=0,
                    max_seq_len=max_seq_len,
                )
            except Exception as e:
                logger.warning("  Sampling failed: %s", e)
                continue

            if not sampled or not sampled[0].inference_logprobs:
                logger.warning("  No completions or logprobs returned")
                continue

            s = sampled[0]
            ft = s.full_tokens
            logger.info(
                "  Sampled: prompt=%d  completion=%d  total=%d",
                s.prompt_len, s.completion_len, len(ft),
            )

            if len(ft) < 2:
                continue

            datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(ft[:-1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=ft[1:], dtype="int64", shape=[len(ft) - 1],
                    ),
                },
            )

            # Policy forward
            try:
                fwd = policy.forward([datum], "cross_entropy")
                policy_lps = list(fwd.loss_fn_outputs[0]["logprobs"].data)
            except Exception as e:
                logger.warning("  Policy forward failed: %s", e)
                continue

            m = compare_logprobs(
                training_lps=policy_lps,
                inference_lps=s.inference_logprobs,
                prompt_len=s.prompt_len,
                label=f"prompt-{pidx}/policy-vs-inference",
            )
            if m.get("count", 0) > 0:
                all_metrics.append(m)

            # Reference forward (if enabled)
            if reference is not None:
                try:
                    ref_fwd = reference.forward([datum], "cross_entropy")
                    ref_lps = list(ref_fwd.loss_fn_outputs[0]["logprobs"].data)
                except Exception as e:
                    logger.warning("  Reference forward failed: %s", e)
                    continue

                ref_m = compare_logprobs(
                    training_lps=ref_lps,
                    inference_lps=s.inference_logprobs,
                    prompt_len=s.prompt_len,
                    label=f"prompt-{pidx}/reference-vs-inference",
                )
                if ref_m.get("count", 0) > 0:
                    all_metrics.append(ref_m)

        # -- Summary ----------------------------------------------------------

        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)

        if all_metrics:
            n = len(all_metrics)
            avg_mean = sum(m["mean_diff"] for m in all_metrics) / n
            worst_max = max(m["max_diff"] for m in all_metrics)
            avg_k1 = sum(m["k1"] for m in all_metrics) / n
            avg_k2 = sum(m["k2"] for m in all_metrics) / n
            avg_k3 = sum(m["k3"] for m in all_metrics) / n

            logger.info("  %d comparisons across %d prompts", n, args.max_rows)
            logger.info(
                "  Completion: avg_mean_diff=%.6f  max_diff=%.6f",
                avg_mean, worst_max,
            )
            logger.info(
                "  KL:         k1=%.6f  k2=%.6f  k3=%.6f",
                avg_k1, avg_k2, avg_k3,
            )

            if worst_max < 0.01:
                logger.info("  RESULT: PASS (max diff < 0.01)")
            elif worst_max < 0.1:
                logger.info(
                    "  RESULT: MARGINAL (max diff %.6f, between 0.01 and 0.1)",
                    worst_max,
                )
            else:
                logger.info("  RESULT: FAIL (max diff %.6f > 0.1)", worst_max)

            with open(os.path.join(args.log_dir, "results.json"), "w") as f:
                json.dump({
                    "training_shape": args.training_shape,
                    "ref_training_shape": args.ref_training_shape,
                    "base_model": args.base_model,
                    "per_prompt": all_metrics,
                    "summary": {
                        "prompts": len(dataset),
                        "comparisons": n,
                        "avg_mean_diff": avg_mean,
                        "max_diff": worst_max,
                        "k1": avg_k1, "k2": avg_k2, "k3": avg_k3,
                    },
                }, f, indent=2)
            logger.info("  Results: %s", os.path.join(args.log_dir, "results.json"))
        else:
            logger.error("  No valid comparisons completed!")


if __name__ == "__main__":
    main()
