from __future__ import annotations

import asyncio
import logging
from typing import Any, List
from datetime import datetime

import chz
import json
import tinker
import verifiers as vf
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.completers import TokensWithLogprobs, TokenCompleter, TinkerTokenCompleter
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, Transition, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
    VerifiersEnvGroupBuilder,
    VerifiersRLDatasetBuilder,
)

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # model configuration
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32

    # environment configuration
    vf_env_id: str = "reverse-text"
    vf_env_args: str | None = None  # JSON string
    dataset_n: int = -1
    dataset_seed: int | None = None

    # training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 32
    num_substeps: int = 1
    learning_rate: float = 1e-5
    max_tokens: int = 512
    kl_penalty_coef: float = 0.0

    # logging configuration
    eval_every: int = 0
    save_every: int = 10
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig, env: Any | None):
    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"verifiers_rl_{model_name_short}_gp{cli_config.groups_per_batch}_gs{cli_config.group_size}"
        f"_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/verifiers_rl/{run_name}"

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # load verifiers environment (must be installed; `prime env install user/env-id`)
    env_args = json.loads(cli_config.vf_env_args) if cli_config.vf_env_args else {}
    vf_env = vf.load_environment(cli_config.vf_env_id, **env_args)

    # global objects shared across rollout groups
    shared_renderer: renderers.Renderer | None = None
    local_tokenizer: Tokenizer | None = None

    async def custom_do_group_rollout(
        builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> TrajectoryGroup:
        assert isinstance(builder, VerifiersEnvGroupBuilder)
        assert isinstance(policy, TinkerTokenCompleter)
        nonlocal shared_renderer, local_tokenizer

        # initialize tokenizer and renderer lazily
        if local_tokenizer is None:
            local_tokenizer = get_tokenizer(cli_config.model_name)
        if shared_renderer is None:
            renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)
            shared_renderer = renderers.get_renderer(renderer_name, local_tokenizer)
        sampling_client = policy.sampling_client

        async def run_one_rollout() -> tuple[Trajectory, float, dict[str, float | int]]:
            recorded: List[
                tuple[list[renderers.Message], tinker.ModelInput, list[int], list[float]]
            ] = []

            def hook(messages, model_input, tokens, logprobs):
                recorded.append((list(messages), model_input, list(tokens), list(logprobs)))

            # create per-rollout client for hook
            assert shared_renderer is not None and local_tokenizer is not None
            local_client = TinkerAsyncOpenAIClient(
                sampling_client, shared_renderer, local_tokenizer
            )
            local_client.set_generation_hook(hook)

            completion, state = await builder.vf_env.rollout(
                client=local_client,
                model="tinker",
                prompt=builder.prompt,
                answer=builder.answer,
                task=builder.task,
                info=builder.info,
                sampling_args={},
            )

            rs = await builder.vf_env.rubric.score_rollout(
                prompt=builder.prompt,
                completion=completion,
                answer=builder.answer,
                state=state,
                task=builder.task,
                info=builder.info,
            )

            transitions: List[Transition] = []
            for _msgs, model_input, tokens, logprobs in recorded:
                transitions.append(
                    Transition(
                        ob=model_input,
                        ac=TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs),
                        reward=0.0,
                        episode_done=False,
                        metrics={},
                    )
                )
            if transitions:
                transitions[-1] = Transition(
                    ob=transitions[-1].ob,
                    ac=transitions[-1].ac,
                    reward=0.0,
                    episode_done=True,
                    metrics=transitions[-1].metrics,
                )
            traj = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())
            return traj, float(rs.reward), dict(rs.metrics)

        results = await asyncio.gather(*[run_one_rollout() for _ in range(cli_config.group_size)])
        trajectories_G = [t for (t, _r, _m) in results]
        final_rewards_G = [r for (_t, r, _m) in results]
        metrics_G = [m for (_t, _r, m) in results]
        return TrajectoryGroup(trajectories_G, final_rewards_G, metrics_G)

    # override do_group_rollout function inside rl.train
    train.do_group_rollout = custom_do_group_rollout

    cfg = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=VerifiersRLDatasetBuilder(
            vf_env=vf_env,
            groups_per_batch=cli_config.groups_per_batch,
            dataset_n=cli_config.dataset_n,
            dataset_seed=cli_config.dataset_seed,
        ),
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        lora_rank=cli_config.lora_rank,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        stream_minibatch_config=None,
    )

    await train.main(cfg)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config, None))
