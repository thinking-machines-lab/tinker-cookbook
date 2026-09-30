"""Train on the counting environment used to check RL training numerics.

Example (the three difficulty settings differ only in the ``task.*`` hints):

    python -m tinker_cookbook.recipes.rl_numerics_check.train \
        model_name=Qwen/Qwen3.6-35B-A3B:peft:262144 \
        renderer_name=qwen3_5_disable_thinking \
        task.num_pages_hint=range task.word_hint=exact
"""

import asyncio
from datetime import datetime

import chz
from tinker.types import LossFnType

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.rl_numerics_check.env import (
    DEFAULT_MAX_TRAJECTORY_TOKENS,
    CountingDatasetBuilder,
    CountingTaskConfig,
)
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.6-35B-A3B:peft:262144"
    renderer_name: str | None = None
    lora_rank: int = 32
    learning_rate: float = 4e-5
    loss_fn: LossFnType = "importance_sampling"
    kl_penalty_coef: float = 0.0
    group_size: int = 16
    groups_per_batch: int = 16
    # Per-turn generation budget.
    max_tokens: int = 1024
    max_trajectory_tokens: int = DEFAULT_MAX_TRAJECTORY_TOKENS
    task: CountingTaskConfig = chz.field(default_factory=CountingTaskConfig)
    seed: int = 0

    eval_every: int = 0
    save_every: int = 20
    max_steps: int | None = None
    base_url: str | None = None
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_config(cli_config: CLIConfig) -> train.Config:
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    task = cli_config.task
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"rl-numerics-{cli_config.model_name.split('/')[-1]}-pages_{task.num_pages_hint}-"
        f"word_{task.word_hint}-{task.num_pages_min}to{task.num_pages_max}p-"
        f"lr{cli_config.learning_rate}-{date_and_time}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/rl_numerics_check/{run_name}"

    dataset_builder = CountingDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        task=task,
        seed=cli_config.seed,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_generation_tokens=cli_config.max_tokens,
    )

    return train.Config(
        model_name=cli_config.model_name,
        recipe_name="recipe_rl_numerics_check",
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        loss_fn=cli_config.loss_fn,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_steps=cli_config.max_steps,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
    )


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    asyncio.run(train.main(config))
