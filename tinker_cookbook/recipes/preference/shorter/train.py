import asyncio
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.preference.shorter.env import (
    ShorterComparisonBuilder,
    ShorterPreferenceModelBuilder,
)
from tinker_cookbook.rl import train
from tinker_cookbook.rl.preference_envs import PairwisePreferenceRLDatasetBuilder


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None

    # Training parameters
    batch_size: int = 32
    group_size: int = 16
    learning_rate: float = 3e-5
    max_tokens: int = 64
    eval_every: int = 5

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None


def cli_main(cli_config: CLIConfig):
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(model_name)

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_tag = model_name.replace("/", "-")
    run_name = f"shorter-{model_tag}-{cli_config.batch_size}batch-{cli_config.group_size}group-{cli_config.learning_rate}lr-{date_and_time}"

    log_path = cli_config.log_path or f"/tmp/tinker-examples/shorter/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    comparison_builder = ShorterComparisonBuilder()
    dataset_builder = PairwisePreferenceRLDatasetBuilder(
        comparison_builder=comparison_builder,
        batch_size=cli_config.batch_size,
        policy_renderer_name=renderer_name,
        policy_model_name=model_name,
        group_size=cli_config.group_size,
        preference_model_builder=ShorterPreferenceModelBuilder(),
    )

    config = train.Config(
        model_name=model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        compute_post_kl=True,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        max_steps=cli_config.max_steps,
    )

    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    asyncio.run(train.main(config))


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
