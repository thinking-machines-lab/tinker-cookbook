from __future__ import annotations

import asyncio
from datetime import datetime

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersRLDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_model_name: str | None = None
    renderer_pool_size: int = 1
    lora_rank: int = 32

    env_config_path: str
    num_tasks: int | None = None

    group_size: int = 8
    groups_per_batch: int = 32
    num_substeps: int = 1
    learning_rate: float = 1e-5
    max_tokens: int = 512
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    max_concurrent: int = 0

    eval_every: int = 0
    save_every: int = 10
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig, env: object | None = None) -> None:
    del env
    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"verifiers_rl_{model_name_short}_gp{cli_config.groups_per_batch}"
        f"_gs{cli_config.group_size}_lr{cli_config.learning_rate}"
        f"_rank{cli_config.lora_rank}_{date_and_time}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/verifiers_rl/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    dataset_builder = VerifiersRLDatasetBuilder(
        env_config_path=cli_config.env_config_path,
        model_name=cli_config.model_name,
        renderer_model_name=cli_config.renderer_model_name or cli_config.model_name,
        renderer_pool_size=cli_config.renderer_pool_size,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        num_tasks=cli_config.num_tasks,
        max_concurrent=cli_config.max_concurrent,
    )
    config = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        recipe_name="recipe_verifiers_rl_v1",
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        lora_rank=cli_config.lora_rank,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        stream_minibatch_config=None,
        max_steps=cli_config.max_steps,
    )
    await train.main(config)


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(config))
