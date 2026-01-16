"""Train with tinker-cookbook using Harbor's terminal environment.

This wires Harbor's serverless sandbox + tmux terminal interaction into
`tinker_cookbook.rl.train.main`.

Assumptions:
- You have a running tinker service reachable by `--base-url` (or default).
- You have installed `tinker` + `tinker-cookbook` dependencies.
- Your Harbor environment credentials (e.g. DAYTONA_API_KEY) are set.

Example:
  python tinker_cookbook/recipes/terminal_rl/train.py  \
  --task-dir tinker_cookbook/recipes/terminal_rl/harbor_envs/devops_task \
  --env daytona \
  --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --renderer-name qwen3 \
  --max-tokens 4096 \
  --temperature 0.7 \
  --groups-per-batch 1 \
  --group-size 8 \
  --num-batches 25 \
  --learning-rate 4e-5 \
  --lora-rank 32 \
  --save-every 10 \
  --log-path ./runs/devops_qwen3_lr4e-5_r32_nb25
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
from pathlib import Path

from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from tinker_cookbook.recipes.terminal_rl.wrapped_env import HarborSingleTaskRLDatasetBuilder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-dir", type=str, required=True)
    p.add_argument("--env", type=str, default="daytona", choices=["daytona", "docker"])

    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--groups-per-batch", type=int, default=1)
    p.add_argument("--num-batches", type=int, default=100)

    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--renderer-name", type=str, required=True)

    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--log-path", type=str, required=True)
    p.add_argument("--base-url", type=str, default=None)

    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--lora-rank", type=int, default=32)

    return p.parse_args()


async def main() -> None:
    args = parse_args()

    train_mod = importlib.import_module("tinker_cookbook.rl.train")
    TrainConfig = getattr(train_mod, "Config")

    env_cfg = TrialEnvironmentConfig(type=args.env)

    dataset_builder = HarborSingleTaskRLDatasetBuilder(
        task_dir=Path(args.task_dir),
        group_size=args.group_size,
        environment=env_cfg,
        groups_per_batch=args.groups_per_batch,
        num_batches=args.num_batches,
        renderer_name=args.renderer_name,
        model_name_for_tokenizer=args.model_name,
    )

    cfg = TrainConfig(
        model_name=args.model_name,
        log_path=args.log_path,
        dataset_builder=dataset_builder,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        eval_every=args.eval_every,
        save_every=args.save_every,
        lora_rank=args.lora_rank,
        base_url=args.base_url,
        remove_constant_reward_groups=False,
    )

    await train_mod.main(cfg)


if __name__ == "__main__":
    asyncio.run(main())
