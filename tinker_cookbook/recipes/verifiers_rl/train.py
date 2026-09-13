from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import cast

import chz
import verifiers.v1 as vf

from tinker_cookbook import cli_utils
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.recipes.verifiers_rl.tinker_generate import TinkerGenerateServer
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
    VerifiersEnvGroupBuilder,
    VerifiersRLDatasetBuilder,
    convert_episodes_to_trajectory_group,
    get_vf_env,
    load_vf_env,
    set_vf_env,
)
from tinker_cookbook.rl import rollouts, train
from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
from tinker_cookbook.rl.rollout_strategy import RolloutStrategy
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # model configuration
    model_name: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32
    renderer_name: str | None = "qwen3_5_disable_thinking"

    # environment configuration
    vf_env_id: str = "reverse-text"
    vf_env_args: str | None = None  # JSON vf.EnvConfig data merged over the taskset id
    dataset_n: int = -1
    dataset_seed: int | None = None

    # training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 32
    num_substeps: int = 1
    learning_rate: float = 1e-5
    max_tokens: int = 512
    temperature: float = 1.0
    # JSON chat-template kwargs passed through verifiers' train client to the
    # `renderers` chat template that tokenizes
    # each turn, e.g. '{"enable_thinking": false}' for Qwen thinking control.
    chat_template_kwargs: str | None = None
    kl_penalty_coef: float = 0.0
    max_concurrent_rollouts: int = 32

    # logging configuration
    eval_every: int = 0
    save_every: int = 10
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig, env: vf.Env | None):
    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"verifiers_rl_{model_name_short}_gp{cli_config.groups_per_batch}_gs{cli_config.group_size}"
        f"_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/verifiers_rl/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    env_args = json.loads(cli_config.vf_env_args) if cli_config.vf_env_args else {}
    sampling_extra = (
        {"chat_template_kwargs": json.loads(cli_config.chat_template_kwargs)}
        if cli_config.chat_template_kwargs
        else {}
    )
    vf_env = env or get_vf_env()
    if vf_env is None:
        vf_env = load_vf_env(cli_config.vf_env_id, env_args)
    set_vf_env(vf_env)

    generate_server = TinkerGenerateServer()
    # Bounds concurrent episodes across the whole batch (each live episode is a
    # harness program plus a sampling stream).
    rollout_semaphore = asyncio.Semaphore(cli_config.max_concurrent_rollouts)

    async def verifiers_do_group_rollout(
        builder: EnvGroupBuilder,
        policy: TokenCompleter,
        strategy: RolloutStrategy | None = None,
        termination: TerminationRewardPolicy | None = None,
    ) -> TrajectoryGroup:
        # `strategy` and `termination` are accepted for signature compatibility
        # but unused: the verifiers environment runs and scores each episode
        # itself.
        del strategy, termination
        generate_server.set_sampling_client(cast(TinkerTokenCompleter, policy).sampling_client)
        ctx = vf.ModelContext(
            model=cli_config.model_name,
            client=generate_server.train_client_config(cli_config.model_name),
            sampling=vf.SamplingConfig.model_validate(
                {
                    "max_tokens": cli_config.max_tokens,
                    "temperature": cli_config.temperature,
                    **sampling_extra,
                }
            ),
        )
        vf_builder = cast(VerifiersEnvGroupBuilder, builder)
        slots = vf_builder.vf_env.slots(vf_builder.task, cli_config.group_size)
        episodes = await asyncio.gather(
            *(vf_builder.vf_env.run_slot(slot, ctx, rollout_semaphore) for slot in slots)
        )
        return convert_episodes_to_trajectory_group(episodes)

    # Override do_group_rollout in the rollouts module, where the rollout
    # pipeline resolves it. (Rebinding the name re-exported on rl.train has
    # no effect on calls made inside tinker_cookbook.rl.rollouts.)
    rollouts.do_group_rollout = verifiers_do_group_rollout

    dataset_builder = VerifiersRLDatasetBuilder(
        vf_env_id=cli_config.vf_env_id,
        vf_env_args=env_args,
        groups_per_batch=cli_config.groups_per_batch,
        dataset_n=cli_config.dataset_n,
        dataset_seed=cli_config.dataset_seed,
    )

    config = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        recipe_name="recipe_verifiers_rl",
        renderer_name=cli_config.renderer_name,
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

    async with generate_server, vf_env.serving():
        await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config, None))
