from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Sequence

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

import gem


def apply_general_prompt(init_obs: str) -> str:
    return (
        f"Question: {init_obs}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


def apply_no_template(init_obs: str) -> str:
    return init_obs


PROMPT_FACTORY = {"no": apply_no_template, "general": apply_general_prompt}


class GemTinkerEnv(Env):
    def __init__(
        self,
        env_gem: gem.Env,
        init_obs: Observation,
        renderer: renderers.Renderer,
        prompt_type: str = "general",
        convo_prefix: list[renderers.Message] | None = None,
    ):
        self.env_gem = env_gem
        self.init_obs = init_obs
        self.renderer = renderer
        self.prompt_type = prompt_type
        self.convo: list[renderers.Message] = list(convo_prefix or [])

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        self.convo.append(
            {"role": "user", "content": PROMPT_FACTORY[self.prompt_type](str(self.init_obs))}
        )
        return self.renderer.build_generation_prompt(self.convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        text = message.get("content", "") if parse_success else ""
        next_obs, reward, terminated, truncated, info = self.env_gem.step(text)
        reward = float(reward)

        metrics: Metrics = {}
        for k, v in (info or {}).items():
            if isinstance(v, (int, float)):
                metrics[k] = v

        done = terminated or truncated
        if done:
            next_ob = tinker.ModelInput.empty()
            next_stop = self.stop_condition
        else:
            self.convo.append({"role": "assistant", "content": text})
            self.convo.append({"role": "user", "content": next_obs})
            next_ob = self.renderer.build_generation_prompt(self.convo)
            next_stop = self.stop_condition

        return StepResult(
            reward=reward,
            episode_done=done,
            next_observation=next_ob,
            next_stop_condition=next_stop,
            metrics=metrics,
        )


@dataclass(frozen=True)
class GemEnvGroupBuilder(EnvGroupBuilder):
    pool: list[gem.Env]
    renderer: renderers.Renderer
    prompt_type: str
    group_size: int
    env_id: str
    convo_prefix: list[renderers.Message] | None = None
    group_index: int = -1  # which env in the pool to use for this

    async def make_envs(self) -> Sequence[Env]:
        assert 0 <= self.group_index < len(self.pool), (
            "group_index should be within the range of the pool size"
        )
        assert hasattr(self.pool[0], "get_state"), (
            "env must support get_state() to run in GemEnvGroupBuilder"
        )

        # duplicate the env for the group size
        env_parent = self.pool[self.group_index]
        init_obs, _ = env_parent.reset()
        return [
            GemTinkerEnv(
                env_parent.spawn(same_state=True),
                init_obs,
                self.renderer,
                self.prompt_type,
                self.convo_prefix,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return self.env_id.split(":")


class GemDataset(RLDataset):
    def __init__(self, builder_config: dict[str, Any], groups_per_batch: int, n_batches: int):
        pool = builder_config["pool"]
        assert len(set(env.seed for env in pool)) == len(pool), (
            "All envs in the pool must have different seeds."
        )

        self.builder_config = builder_config
        self.groups_per_batch = groups_per_batch
        self.n_batches = n_batches

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        return [
            GemEnvGroupBuilder(group_index=i, **self.builder_config)
            for i in range(self.groups_per_batch)
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class GemDatasetBuilder(RLDatasetBuilder):
    env_id: str
    model_name_for_tokenizer: str
    renderer_name: str
    prompt_type: str = "general"
    group_size: int
    groups_per_batch: int
    n_batches: int = 100
    env_kwargs_json: str | None = None
    convo_prefix: list[renderers.Message] | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        env_kwargs = json.loads(self.env_kwargs_json) if self.env_kwargs_json else {}
        env_parent = gem.make(self.env_id, seed=int(time.time_ns()), **env_kwargs)
        seed_parent = env_parent.seed  # type: ignore
        pool = [env_parent.spawn(seed=i + seed_parent + 1) for i in range(self.groups_per_batch)]
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        builder_config = {
            "pool": pool,
            "renderer": renderer,
            "prompt_type": self.prompt_type,
            "group_size": self.group_size,
            "env_id": self.env_id,
            "convo_prefix": self.convo_prefix,
        }
        return GemDataset(builder_config, self.groups_per_batch, self.n_batches), None
