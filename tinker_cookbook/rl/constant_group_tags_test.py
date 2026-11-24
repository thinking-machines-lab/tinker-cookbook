"""Per-tag metrics stay attached to their own groups when constant-reward groups are dropped.

Runs the real synchronous loop over the fake clients from ``async_training_test``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, cast

import pytest
import tinker

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.async_training_test import (
    _FakeTokenizer,
    _FakeTrainingClient,
    _Harness,
    _ListDataset,
    _SleepEnv,
    _SleepEnvGroupBuilder,
    _UnusedDatasetBuilder,
)
from tinker_cookbook.rl.train import Config
from tinker_cookbook.rl.types import Env, EnvGroupBuilder
from tinker_cookbook.utils import ml_log

GROUP_SIZE = 2


class _TaggedEnvGroupBuilder(_SleepEnvGroupBuilder):
    """A group with one tag and one reward per rollout."""

    def __init__(self, tag: str, problem_id: int, harness: _Harness, rewards: tuple[float, ...]):
        super().__init__(problem_id, 0.0, len(rewards), harness)
        self.tag = tag
        self.rewards = rewards

    async def make_envs(self) -> Sequence[Env]:
        return [
            _SleepEnv(self.problem_id, 0.0, reward=reward, harness=self.harness, fail=False)
            for reward in self.rewards
        ]

    def logging_tags(self) -> list[str]:
        return [self.tag]


# One constant-reward group and two kept groups whose mean rewards differ.
DROPPED = ("dropped", (1.0, 1.0))
KEPT_A = ("kept_a", (0.0, 1.0))
KEPT_B = ("kept_b", (0.0, 0.5))


def _run_sync(tmp_path, groups: list[tuple[str, tuple[float, ...]]]) -> dict[str, Any]:
    """One batch of ``(tag, rewards)`` groups; returns the step's metrics row."""
    harness = _Harness(filtered_problems=set())
    builders: list[EnvGroupBuilder] = []
    for problem_id, (tag, rewards) in enumerate(groups):
        builders.append(_TaggedEnvGroupBuilder(tag, problem_id, harness, rewards))
        if len(set(rewards)) == 1:
            harness.filtered_problems.add(problem_id)

    training_client = _FakeTrainingClient(harness)
    config = Config(
        learning_rate=1e-5,
        dataset_builder=_UnusedDatasetBuilder(),
        model_name="fake-model",
        recipe_name="constant_group_tags_test",
        max_tokens=4,
        log_path=str(tmp_path),
        num_groups_to_log=0,
        rollout_json_export=False,
        remove_constant_reward_groups=True,
    )
    ml_logger = ml_log.setup_logging(log_dir=str(tmp_path), do_configure_logging_module=False)
    try:
        checkpoint_mgr = checkpoint_utils.CheckpointManager(
            training_client=cast(tinker.TrainingClient, training_client),
            service_client=cast(tinker.ServiceClient, None),
            log_path=str(tmp_path),
            save_every=0,
            store=ml_logger.store,
        )
        asyncio.run(
            asyncio.wait_for(
                train.do_sync_training(
                    start_batch=0,
                    end_batch=1,
                    num_batches=1,
                    config=config,
                    training_client=cast(tinker.TrainingClient, training_client),
                    kl_reference_client=None,
                    evaluators=[],
                    dataset=_ListDataset([builders]),
                    ml_logger=ml_logger,
                    tokenizer=cast(Any, _FakeTokenizer()),
                    checkpoint_mgr=checkpoint_mgr,
                ),
                timeout=30.0,
            )
        )
    finally:
        ml_logger.close()
    with open(f"{tmp_path}/metrics.jsonl") as f:
        rows = [json.loads(line) for line in f]
    [row] = [r for r in rows if "env/all/total_episodes" in r]
    return row


def _tags_with_metrics(row: dict[str, Any]) -> set[str]:
    return {key.split("/")[1] for key in row if key.startswith("env/")} - {"all"}


@pytest.mark.parametrize(
    "groups",
    [[DROPPED, KEPT_A, KEPT_B], [KEPT_A, DROPPED, KEPT_B], [KEPT_A, KEPT_B, DROPPED]],
    ids=["dropped_first", "dropped_in_the_middle", "dropped_last"],
)
def test_tags_follow_the_groups_that_were_kept(tmp_path, groups):
    # Wherever the dropped group sits, each kept tag reports its own group's reward.
    # Before the builders were filtered along with the trajectory groups, every tag
    # after the dropped group reported the next group's numbers.
    row = _run_sync(tmp_path, groups)
    assert _tags_with_metrics(row) == {"kept_a", "kept_b"}
    assert row["env/all/total_episodes"] == 2 * GROUP_SIZE
    assert row["env/kept_a/total_episodes"] == GROUP_SIZE
    assert row["env/kept_b/total_episodes"] == GROUP_SIZE
    assert row["env/kept_a/reward/total"] == pytest.approx(0.5)
    assert row["env/kept_b/reward/total"] == pytest.approx(0.25)


def test_tags_are_unchanged_when_nothing_is_dropped(tmp_path):
    row = _run_sync(tmp_path, [("a", (0.0, 1.0)), ("b", (0.0, 0.5)), ("c", (0.5, 1.0))])
    assert _tags_with_metrics(row) == {"a", "b", "c"}
    assert row["env/all/total_episodes"] == 3 * GROUP_SIZE
    assert row["env/c/reward/total"] == pytest.approx(0.75)


def test_all_constant_batch_keeps_one_group_and_its_builder(tmp_path):
    # remove_constant_reward_groups falls back to the first group when every group is
    # constant. One group cannot split by tag, so only the env/all metrics are logged.
    row = _run_sync(tmp_path, [("a", (1.0, 1.0)), ("b", (0.0, 0.0))])
    assert _tags_with_metrics(row) == set()
    assert row["env/all/total_episodes"] == GROUP_SIZE
    assert row["env/all/reward/total"] == pytest.approx(1.0)
