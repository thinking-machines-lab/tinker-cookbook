"""Per-tag metrics stay attached to their own groups when constant-reward groups are dropped.

Runs the real synchronous loop over the fake clients from ``async_training_test``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import tinker

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.async_training_test import (
    _FakeTokenizer,
    _FakeTrainingClient,
    _Harness,
    _ListDataset,
    _SleepEnvGroupBuilder,
    _UnusedDatasetBuilder,
)
from tinker_cookbook.rl.train import Config
from tinker_cookbook.rl.types import EnvGroupBuilder
from tinker_cookbook.utils import ml_log

GROUP_SIZE = 2


class _TaggedEnvGroupBuilder(_SleepEnvGroupBuilder):
    def __init__(self, tag: str, problem_id: int, harness: _Harness, constant_reward: bool):
        super().__init__(problem_id, 0.0, GROUP_SIZE, harness, constant_reward=constant_reward)
        self.tag = tag

    def logging_tags(self) -> list[str]:
        return [self.tag]


def _run_sync(tmp_path, groups: list[tuple[str, bool]]) -> dict[str, Any]:
    """One batch of ``(tag, constant_reward)`` groups; returns the step's metrics row."""
    harness = _Harness(filtered_problems=set())
    builders: list[EnvGroupBuilder] = []
    for problem_id, (tag, constant) in enumerate(groups):
        builders.append(_TaggedEnvGroupBuilder(tag, problem_id, harness, constant))
        if constant:
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


def test_tags_follow_the_groups_that_were_kept(tmp_path):
    # The dropped group comes first, so every later tag would shift by one if the
    # builders were not filtered along with the trajectory groups.
    row = _run_sync(tmp_path, [("dropped", True), ("kept_a", False), ("kept_b", False)])
    assert _tags_with_metrics(row) == {"kept_a", "kept_b"}
    assert row["env/all/total_episodes"] == 2 * GROUP_SIZE
    assert row["env/kept_a/total_episodes"] == GROUP_SIZE
    assert row["env/kept_b/total_episodes"] == GROUP_SIZE


def test_tags_are_unchanged_when_nothing_is_dropped(tmp_path):
    row = _run_sync(tmp_path, [("a", False), ("b", False), ("c", False)])
    assert _tags_with_metrics(row) == {"a", "b", "c"}
    assert row["env/all/total_episodes"] == 3 * GROUP_SIZE
