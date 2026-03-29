"""
Full Nemotron-Cascade-2 RL pipeline on Nemotron-3-Nano-30B.

Runs the cascade sequentially with paper-matched hyperparameters:
  SFT checkpoint -> IF-RL (180 steps) -> Multi-domain RL (70 steps) -> eval

Each stage loads from the previous stage's checkpoint.
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Paper hyperparameters
PAPER_CONFIG = {
    "if_rl": {
        "group_size": 16,
        "groups_per_batch": 128,
        "learning_rate": 3e-6,
        "max_tokens": 49152,  # 49K
        "temperature": 1.0,
        "kl_penalty_coef": 0.0,
        "max_steps": 180,
        "save_every": 20,
        "eval_every": 20,
        "remove_constant_reward_groups": True,
    },
    "mcqa": {
        "group_size": 16,
        "groups_per_batch": 128,
        "learning_rate": 3e-6,
        "max_tokens": 49152,
        "temperature": 1.0,
        "kl_penalty_coef": 0.0,
        "max_steps": 70,
        "save_every": 10,
        "eval_every": 10,
        "remove_constant_reward_groups": True,
    },
    "structured_output": {
        "group_size": 16,
        "groups_per_batch": 128,
        "learning_rate": 3e-6,
        "max_tokens": 49152,
        "temperature": 1.0,
        "kl_penalty_coef": 0.0,
        "max_steps": 70,
        "save_every": 10,
        "eval_every": 10,
        "remove_constant_reward_groups": True,
    },
}

# Smaller config for testing/development
SMALL_CONFIG = {env: {**cfg, "groups_per_batch": 32, "max_tokens": 16384, "max_steps": min(cfg["max_steps"], 50)}
                for env, cfg in PAPER_CONFIG.items()}


def find_latest_checkpoint(log_dir: str) -> str | None:
    """Find the latest checkpoint in a training log directory."""
    cp_file = os.path.join(log_dir, "checkpoints.jsonl")
    if not os.path.exists(cp_file):
        return None
    last = None
    with open(cp_file) as f:
        for line in f:
            last = json.loads(line)
    return last.get("state_path") if last else None


async def run_rl_stage(
    model_name: str,
    env: str,
    checkpoint_path: str,
    config: dict,
    log_path: str,
):
    """Run a single RL stage."""
    from tinker_cookbook.recipes.nemotron_cascade.rl.train import CLIConfig, cli_main

    cli_config = CLIConfig(
        model_name=model_name,
        env=env,
        load_checkpoint_path=checkpoint_path,
        log_path=log_path,
        behavior_if_log_dir_exists="delete",
        **config,
    )
    logger.info(f"Starting {env}: checkpoint={checkpoint_path}, log_path={log_path}")
    await cli_main(cli_config)
    logger.info(f"Completed {env}")

    # Find and return the final checkpoint
    cp = find_latest_checkpoint(log_path)
    if cp:
        logger.info(f"  Final checkpoint: {cp}")
    return cp


async def run_eval_stage(
    model_name: str,
    checkpoint_path: str | None,
    label: str,
    output_dir: str,
    limit: int = 200,
):
    """Run benchmark evaluation."""
    from tinker_cookbook.recipes.nemotron_cascade.eval.run_evals import run_eval
    logger.info(f"Evaluating: {label}")
    results = await run_eval(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        benchmarks=["gsm8k", "ifeval"],
        limit=limit,
        output_dir=os.path.join(output_dir, label),
    )
    return results


async def run_cascade(
    model_name: str,
    sft_checkpoint: str,
    stages: list[str],
    config_size: str = "paper",
    log_base: str | None = None,
    run_evals: bool = True,
    eval_limit: int = 200,
):
    """Run the full cascade: stage1 -> stage2 -> ... -> eval."""
    configs = PAPER_CONFIG if config_size == "paper" else SMALL_CONFIG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_short = model_name.replace("/", "-").replace(":", "-")

    if log_base is None:
        log_base = f"/tmp/tinker-examples/nemotron_cascade_pipeline/{model_short}_{timestamp}"
    os.makedirs(log_base, exist_ok=True)

    current_checkpoint = sft_checkpoint
    all_results = {}

    # Optionally eval the SFT checkpoint first
    if run_evals:
        sampler_cp = sft_checkpoint.replace("/weights/", "/sampler_weights/") if sft_checkpoint else None
        results = await run_eval_stage(model_name, sampler_cp, "sft", log_base, eval_limit)
        all_results["sft"] = results

    # Run each RL stage sequentially
    for stage in stages:
        if stage not in configs:
            logger.warning(f"Unknown stage: {stage}")
            continue

        stage_log = os.path.join(log_base, stage)
        new_checkpoint = await run_rl_stage(
            model_name=model_name,
            env=stage,
            checkpoint_path=current_checkpoint,
            config=configs[stage],
            log_path=stage_log,
        )

        if new_checkpoint:
            current_checkpoint = new_checkpoint
        else:
            logger.error(f"Stage {stage} did not produce a checkpoint!")
            break

        # Eval after each stage
        if run_evals:
            sampler_cp = current_checkpoint.replace("/weights/", "/sampler_weights/")
            results = await run_eval_stage(model_name, sampler_cp, f"after_{stage}", log_base, eval_limit)
            all_results[f"after_{stage}"] = results

    # Print final comparison
    if all_results:
        print("\n" + "=" * 80)
        print("CASCADE RESULTS")
        print("=" * 80)
        metrics = sorted(set(k for r in all_results.values() for k in r.keys()))
        names = list(all_results.keys())
        header = f"{'Metric':<35}" + "".join(f"{n:<20}" for n in names)
        print(header)
        print("-" * len(header))
        for m in metrics:
            row = f"{m:<35}"
            for n in names:
                v = all_results[n].get(m, "N/A")
                row += f"{v:<20.4f}" if isinstance(v, float) else f"{str(v):<20}"
            print(row)

    # Save results
    results_file = os.path.join(log_base, "cascade_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "model": model_name,
            "sft_checkpoint": sft_checkpoint,
            "stages": stages,
            "config_size": config_size,
            "results": {k: {mk: float(mv) if isinstance(mv, float) else mv for mk, mv in v.items()}
                       for k, v in all_results.items()},
        }, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    return current_checkpoint, all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144")
    parser.add_argument("--sft-checkpoint", required=True, help="SFT checkpoint to start from")
    parser.add_argument("--stages", default="if_rl,mcqa,structured_output",
                        help="Comma-separated RL stages")
    parser.add_argument("--config", default="small", choices=["paper", "small"],
                        help="paper=full scale, small=faster iteration")
    parser.add_argument("--log-base", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=200)
    args = parser.parse_args()

    asyncio.run(run_cascade(
        model_name=args.model,
        sft_checkpoint=args.sft_checkpoint,
        stages=args.stages.split(","),
        config_size=args.config,
        log_base=args.log_base,
        run_evals=not args.no_eval,
        eval_limit=args.eval_limit,
    ))


if __name__ == "__main__":
    main()
