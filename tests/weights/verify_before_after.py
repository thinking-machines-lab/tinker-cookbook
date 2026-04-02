#!/usr/bin/env python3
"""Verify before/after bitwise equivalence for the shard engine refactor.

Trains 1-step adapters, saves them to a persistent directory, then runs
build_hf_model with the current code and compares against a baseline
output (generated from main branch).

Usage:
    # Step 1: On the main branch, generate baseline outputs
    git checkout main
    HF_HUB_CACHE=~/huggingface/hub python tests/weights/verify_before_after.py --generate-baseline

    # Step 2: On the PR branch, compare against baseline
    git checkout yujia/weights-gpu-accel
    HF_HUB_CACHE=~/huggingface/hub python tests/weights/verify_before_after.py --compare

    # Or do both in one shot (uses git worktree for baseline):
    HF_HUB_CACHE=~/huggingface/hub python tests/weights/verify_before_after.py --full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import cast

import datasets
import tinker
import torch
from safetensors.torch import load_file

from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.weights import build_hf_model, download

WORK_DIR = Path.home() / "verify_weights_refactor"
ADAPTER_DIR = WORK_DIR / "adapters"
BASELINE_DIR = WORK_DIR / "baseline"
CURRENT_DIR = WORK_DIR / "current"

MODELS = [
    {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "renderer": "qwen3_instruct",
        "label": "qwen3_dense",
        "merge_kwargs": {},
    },
    {
        "name": "Qwen/Qwen3.5-35B-A3B",
        "renderer": "qwen3_5",
        "label": "qwen35_moe",
        "merge_kwargs": {},
    },
    {
        "name": "Qwen/Qwen3.5-35B-A3B",
        "renderer": "qwen3_5",
        "label": "qwen35_moe_fp8",
        "merge_kwargs": {"quantize": "experts-fp8", "serving_format": "vllm"},
    },
    {
        "name": "deepseek-ai/DeepSeek-V3.1",
        "renderer": "deepseekv3",
        "label": "deepseek_fp8",
        "merge_kwargs": {"quantize": "experts-fp8", "serving_format": "vllm"},
    },
]

BATCH_SIZE = 4
MAX_LENGTH = 512
LORA_RANK = 8


def train_adapters():
    """Train 1-step adapters for each model and save to persistent dir."""
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    for model_cfg in MODELS:
        name = model_cfg["name"]
        renderer_name = model_cfg["renderer"]
        label = model_cfg["label"]
        adapter_path = ADAPTER_DIR / label

        # Skip if adapter already exists (reuse across runs)
        if (adapter_path / "adapter_config.json").exists():
            print(f"  [{label}] Adapter exists, skipping training")
            continue

        # Use shared adapter for same model name
        base_label = next(m["label"] for m in MODELS if m["name"] == name)
        base_adapter = ADAPTER_DIR / base_label
        if base_label != label and (base_adapter / "adapter_config.json").exists():
            print(f"  [{label}] Reusing adapter from {base_label}")
            adapter_path.mkdir(parents=True, exist_ok=True)
            for f in base_adapter.iterdir():
                (adapter_path / f.name).symlink_to(f)
            continue

        print(f"  [{label}] Training {name}...")
        tokenizer = get_tokenizer(name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        ds = datasets.load_dataset("allenai/tulu-3-sft-mixture")
        ds = cast(datasets.DatasetDict, ds)
        train_ds = ds["train"].take(BATCH_SIZE)

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(row["messages"], renderer, MAX_LENGTH)

        sft_dataset = SupervisedDatasetFromHFDataset(train_ds, batch_size=BATCH_SIZE, map_fn=map_fn)

        async def _run() -> str:
            sc = tinker.ServiceClient()
            tc = await sc.create_lora_training_client_async(base_model=name, rank=LORA_RANK)
            batch = sft_dataset.get_batch(0)
            fwd_bwd = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            await fwd_bwd.result_async()
            optim = await tc.optim_step_async({"learning_rate": 1e-4})
            await optim.result_async()
            resp = await tc.save_weights_for_sampler_async(f"verify_{label}")
            result = await resp.result_async()
            return result.path

        tinker_path = asyncio.run(_run())
        download(tinker_path=tinker_path, output_dir=str(adapter_path))
        print(f"  [{label}] Adapter saved to {adapter_path}")


def run_merges(output_dir: Path):
    """Run build_hf_model for each model config, save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in MODELS:
        label = model_cfg["label"]
        adapter_path = ADAPTER_DIR / label
        out = output_dir / label

        if out.exists():
            print(f"  [{label}] Output exists, skipping")
            continue

        print(f"  [{label}] Merging {model_cfg['name']}...")
        try:
            build_hf_model(
                base_model=model_cfg["name"],
                adapter_path=str(adapter_path),
                output_path=str(out),
                **model_cfg["merge_kwargs"],
            )
            print(f"  [{label}] Done → {out}")
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")


def compare_outputs():
    """Compare baseline and current outputs tensor-by-tensor."""
    all_match = True

    for model_cfg in MODELS:
        label = model_cfg["label"]
        baseline = BASELINE_DIR / label
        current = CURRENT_DIR / label

        if not baseline.exists():
            print(f"  [{label}] SKIP — no baseline")
            continue
        if not current.exists():
            print(f"  [{label}] SKIP — no current output")
            continue

        print(f"  [{label}] Comparing...")

        # Load all tensors from both
        baseline_tensors = {}
        for sf in sorted(baseline.glob("*.safetensors")):
            baseline_tensors.update(load_file(str(sf)))

        current_tensors = {}
        for sf in sorted(current.glob("*.safetensors")):
            current_tensors.update(load_file(str(sf)))

        # Compare keys
        baseline_keys = set(baseline_tensors.keys())
        current_keys = set(current_tensors.keys())

        if baseline_keys != current_keys:
            only_baseline = baseline_keys - current_keys
            only_current = current_keys - baseline_keys
            print("    KEY MISMATCH:")
            if only_baseline:
                print(f"    Only in baseline: {sorted(only_baseline)[:5]}")
            if only_current:
                print(f"    Only in current: {sorted(only_current)[:5]}")
            all_match = False
            continue

        # Compare tensors
        mismatches = []
        for key in sorted(baseline_keys):
            bt = baseline_tensors[key]
            ct = current_tensors[key]

            if bt.shape != ct.shape:
                mismatches.append(f"    {key}: shape {bt.shape} vs {ct.shape}")
                continue
            if bt.dtype != ct.dtype:
                mismatches.append(f"    {key}: dtype {bt.dtype} vs {ct.dtype}")
                continue

            if bt.dtype == torch.float8_e4m3fn:
                equal = torch.equal(bt.to(torch.float32), ct.to(torch.float32))
            else:
                equal = torch.equal(bt, ct)

            if not equal:
                diff = (bt.float() - ct.float()).abs()
                mismatches.append(
                    f"    {key}: max_diff={diff.max().item():.6e}, "
                    f"mean_diff={diff.mean().item():.6e}"
                )

        if mismatches:
            print(f"    MISMATCH ({len(mismatches)} tensors):")
            for m in mismatches[:10]:
                print(m)
            all_match = False
        else:
            print(f"    MATCH — {len(baseline_keys)} tensors identical")

        # Compare config.json
        baseline_config = json.loads((baseline / "config.json").read_text())
        current_config = json.loads((current / "config.json").read_text())
        if baseline_config != current_config:
            print("    CONFIG MISMATCH")
            all_match = False
        else:
            print("    config.json identical")

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Verify before/after weight export equivalence")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train adapters only")
    group.add_argument("--generate-baseline", action="store_true", help="Run merges as baseline")
    group.add_argument("--generate-current", action="store_true", help="Run merges as current")
    group.add_argument("--compare", action="store_true", help="Compare baseline vs current")
    args = parser.parse_args()

    if args.train:
        print("=== Training adapters ===")
        train_adapters()
    elif args.generate_baseline:
        print("=== Generating baseline outputs ===")
        if not ADAPTER_DIR.exists():
            print("No adapters found. Run --train first.")
            sys.exit(1)
        run_merges(BASELINE_DIR)
    elif args.generate_current:
        print("=== Generating current outputs ===")
        if not ADAPTER_DIR.exists():
            print("No adapters found. Run --train first.")
            sys.exit(1)
        run_merges(CURRENT_DIR)
    elif args.compare:
        print("=== Comparing baseline vs current ===")
        match = compare_outputs()
        if match:
            print("\n ALL OUTPUTS MATCH")
        else:
            print("\n MISMATCHES FOUND")
            sys.exit(1)


if __name__ == "__main__":
    main()
