from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.distillation.datasets import PromptOnlyEnv
from tinker_cookbook.distillation.train_on_policy import Config, _maybe_export_rollout_summaries
from tinker_cookbook.renderers.base import Message, ParseTermination
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.stores.training_store import TrainingRunStore


class _FakeRenderer:
    def get_stop_sequences(self):
        return []

    def parse_response(self, action: list[int]) -> tuple[Message, ParseTermination]:
        return {"role": "assistant", "content": "a distilled answer"}, ParseTermination.STOP_SEQUENCE


class _FakeBuilder(EnvGroupBuilder):
    async def make_envs(self):  # type: ignore[override]
        return []

    def logging_tags(self) -> list[str]:
        return ["distillation-test"]


def test_prompt_only_env_step_includes_trace_payload():
    env = PromptOnlyEnv(
        prompt="Explain why 2 + 2 = 4.",
        renderer=cast(renderers.Renderer, _FakeRenderer()),
        convo_prefix=[{"role": "system", "content": "Be concise."}],
    )

    result = asyncio.run(env.step([1, 2, 3]))

    assert result.trace is not None
    prompt = cast(dict[str, object], result.trace["prompt"])
    completion = cast(dict[str, object], result.trace["policy_response"])
    assert prompt["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain why 2 + 2 = 4."},
    ]
    assert completion["messages"] == [{"role": "assistant", "content": "a distilled answer"}]


def test_maybe_export_rollout_summaries_writes_store(tmp_path: Path):
    transition = Transition(
        ob=tinker.ModelInput.from_ints([1, 2]),
        ac=TokensWithLogprobs(tokens=[3, 4], maybe_logprobs=[-0.1, -0.2]),
        reward=0.0,
        episode_done=True,
        trace={"policy_response": {"messages": [{"role": "assistant", "content": "answer"}]}},
    )
    trajectory_group = TrajectoryGroup(
        trajectories_G=[Trajectory(transitions=[transition], final_ob=tinker.ModelInput.empty())],
        final_rewards_G=[0.0],
        metrics_G=[{}],
    )
    store = TrainingRunStore(LocalStorage(tmp_path))

    _maybe_export_rollout_summaries(
        config=cast(Config, SimpleNamespace(rollout_json_export=True)),
        iteration=5,
        env_group_builders_P=[_FakeBuilder()],
        trajectory_groups_P=[trajectory_group],
        store=store,
    )

    stored_records = store.read_rollouts(5)
    assert len(stored_records) == 1
    assert stored_records[0]["tags"] == ["distillation-test"]
    assert stored_records[0]["steps"][0]["trace"]["policy_response"]["messages"][0]["content"] == (
        "answer"
    )
    assert stored_records[0]["iteration"] == 5
