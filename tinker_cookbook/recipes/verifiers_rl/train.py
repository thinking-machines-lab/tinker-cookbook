from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence, List
from datetime import datetime

import chz
import json
import tinker
import verifiers as vf
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder, Trajectory, Transition, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer
from .oai_from_tinker import TinkerOpenAIClient
from tinker_cookbook.completers import TokensWithLogprobs

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    learning_rate: float = 5e-6
    max_tokens: int = 128
    batch_size: int = 4
    eval_every: int = 0
    save_every: int = 10
    base_url: str | None = None
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    dataset_n: int = -1
    dataset_seed: int | None = None
    vf_env_id: str = ""
    vf_env_args: str | None = None  # JSON string


async def cli_main(cli_config: CLIConfig, env: Any | None):
    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"verifiers_rl_{model_name_short}_bs{cli_config.batch_size}_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/verifiers_rl/{run_name}"

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # load verifiers environment (must be installed; `prime env install user/env-id`)
    env_args = json.loads(cli_config.vf_env_args) if cli_config.vf_env_args else {}
    vf_env = vf.load_environment(cli_config.vf_env_id, **env_args)

    class VerifiersDataset(RLDataset):
        def __init__(self, rows: list[dict], vf_env: vf.Environment):
            self._rows = rows
            self._vf_env = vf_env

        def __len__(self) -> int:
            return (len(self._rows) + cli_config.batch_size - 1) // cli_config.batch_size

        def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
            start = index * cli_config.batch_size
            end = min(len(self._rows), start + cli_config.batch_size)
            builders: list[EnvGroupBuilder] = []
            for j in range(start, end):
                row = self._rows[j]
                builders.append(VerifiersBuilder(
                    vf_env=vf_env,
                    prompt=row["prompt"],
                    answer=row.get("answer", ""),
                    task=row.get("task", "default"),
                    info=row.get("info", {}),
                ))
            return builders

    class VerifiersDatasetBuilder(RLDatasetBuilder):
        async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
            ds = vf_env.get_dataset(n=cli_config.dataset_n, seed=cli_config.dataset_seed)
            rows = [
                {
                    "prompt": ds["prompt"][i],
                    **({"answer": ds["answer"][i]} if "answer" in ds.column_names else {}),
                    **({"task": ds["task"][i]} if "task" in ds.column_names else {}),
                    **({"info": ds["info"][i]} if "info" in ds.column_names else {}),
                }
                for i in range(len(ds))
            ]
            return VerifiersDataset(rows, vf_env), None

    class VerifiersBuilder(EnvGroupBuilder):
        def __init__(self, vf_env: Any, prompt: Any, answer: str, task: str, info: dict):
            self.vf_env = vf_env
            self.prompt = prompt
            self.answer = answer
            self.task = task
            self.info = info

        async def make_envs(self):
            return []  # unused when using custom_do_group_rollout

        def logging_tags(self):
            return [self.task] if self.task else []

    async def custom_do_group_rollout(cfg: train.Config, sampling_client: tinker.SamplingClient, builder: EnvGroupBuilder, tokenizer: Tokenizer) -> TrajectoryGroup:
        assert isinstance(builder, VerifiersBuilder)
        renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        client = TinkerOpenAIClient(sampling_client, renderer, tokenizer)

        # capture steps
        recorded: List[tuple[list[renderers.Message], tinker.ModelInput, list[int], list[float]]] = []

        def hook(messages, model_input, tokens, logprobs):
            recorded.append((list(messages), model_input, list(tokens), list(logprobs)))

        client.set_generation_hook(hook)

        # run rollout via verifiers env
        completion, state = await builder.vf_env.rollout(
            client=client,
            model="tinker",
            prompt=builder.prompt,
            answer=builder.answer,
            task=builder.task,
            info=builder.info,
            sampling_args={},
        )

        # score
        rs = await builder.vf_env.rubric.score_rollout(
            prompt=builder.prompt,
            completion=completion,
            answer=builder.answer,
            state=state,
            task=builder.task,
            info=builder.info,
        )

        # build trajectory
        transitions: List[Transition] = []
        for _msgs, model_input, tokens, logprobs in recorded:
            transitions.append(Transition(
                ob=model_input,
                ac=TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs),
                reward=0.0,
                episode_done=False,
                metrics={},
            ))
        if transitions:
            transitions[-1] = Transition(
                ob=transitions[-1].ob,
                ac=transitions[-1].ac,
                reward=0.0,
                episode_done=True,
                metrics=transitions[-1].metrics,
            )
        traj = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())


        return TrajectoryGroup([traj], [float(rs.reward)], [dict(rs.metrics)])

    # override the default do_group_rollout with our custom one
    train.do_group_rollout = custom_do_group_rollout
    
    cfg = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=VerifiersDatasetBuilder(),
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        lora_rank=cli_config.lora_rank,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        stream_minibatch_config=None,
    )

    await train.main(cfg)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config, None))