import asyncio
import logging
from datetime import datetime
import chz
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.math_rl import (
    arithmetic_env,
    math_env,
)
from tinker_cookbook.recipes.fpgb.train import Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for FPGB math RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3.5-9B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: str = "arithmetic"  # Options: arithmetic, math, polaris, deepmath, gsm8k
    seed: int = 0  # Random seed for data shuffling

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    # FPGB functional step size. The residual-regression target is
    #     delta_logp ~= fpgb_beta * advantage.
    fpgb_beta: float = 0.1
    max_tokens: int = 5
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0

    # Remove trajectory groups whose rollouts all received the same reward.
    # Such groups have zero centered advantage and therefore no FPGB signal.
    remove_constant_reward_groups: bool = False

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 20

    # Checkpointing
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # FPGB V0 uses fully synchronous on-policy training.
    max_steps: int | None = None


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
) -> RLDatasetBuilder:
    if env == "arithmetic":
        return arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            n_batches=100,
            include_fewshot=True,
            group_size=group_size,
        )
    elif env in ["math", "polaris", "deepmath", "gsm8k"]:
        return math_env.get_math_dataset_builder(
            dataset_name=env,
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown environment: {env}")


async def cli_main(cli_config: CLIConfig):
    """Convert CLI settings to the FPGB trainer config and run training."""

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    model_name = cli_config.model_name.replace("/", "-")
    run_name = (
        f"fpgb-{cli_config.env}-{model_name}-"
        f"{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-"
        f"beta{cli_config.fpgb_beta}-"
        f"{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-"
        f"seed{cli_config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/fpgb_math_rl/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
        ),
        model_name=cli_config.model_name,
        recipe_name="recipe_fpgb_math_rl",
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        num_substeps=cli_config.num_substeps,
        fpgb_beta=cli_config.fpgb_beta,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=None,
        stream_minibatch_config=None,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
