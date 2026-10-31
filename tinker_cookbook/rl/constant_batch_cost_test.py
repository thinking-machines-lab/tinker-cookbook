"""Batches whose groups are all dropped still get their cost recorded, in the real loops.

Runs the real streaming step function and the real outer loops over the fake
clients from ``async_training_test`` (one prompt token and one sampled token
per trajectory, no cache hits), so every group of size 2 costs 2 sampled and
2 uncached prefill tokens.
"""

from __future__ import annotations

import asyncio
import json
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
    _SleepEnvGroupBuilder,
    _UnusedDatasetBuilder,
)
from tinker_cookbook.rl.train import AsyncConfig, Config, StreamMinibatchConfig
from tinker_cookbook.rl.types import EnvGroupBuilder
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.cost_estimation import CostTracker, TokenCostRates, TokenUsage

RATES = TokenCostRates(
    prefill_usd_per_million=1.0,
    cached_prefill_usd_per_million=0.2,
    sample_usd_per_million=2.0,
    train_usd_per_million=3.0,
)
GROUP_SIZE = 2
# Per group: GROUP_SIZE sampled tokens and GROUP_SIZE uncached prompt tokens.
GROUP_USD = GROUP_SIZE * (2.0 + 1.0) / 1e6


def _run(
    tmp_path,
    batches: list[list[bool]],
    *,
    loop: str,
    groups_per_batch: int,
    num_minibatches: int = 2,
) -> tuple[_Harness, CostTracker, list[dict[str, Any]]]:
    """``batches[i][j]`` says whether group j of batch i has a constant reward."""
    harness = _Harness(filtered_problems=set())
    training_client = _FakeTrainingClient(harness)
    problem_id = 0
    dataset_batches: list[list[EnvGroupBuilder]] = []
    for batch in batches:
        builders: list[EnvGroupBuilder] = []
        for constant in batch:
            builders.append(
                _SleepEnvGroupBuilder(
                    problem_id, 0.0, GROUP_SIZE, harness, constant_reward=constant
                )
            )
            if constant:
                harness.filtered_problems.add(problem_id)
            problem_id += 1
        dataset_batches.append(builders)

    config = Config(
        learning_rate=1e-5,
        dataset_builder=_UnusedDatasetBuilder(),
        model_name="fake-model",
        recipe_name="constant_batch_cost_test",
        max_tokens=4,
        log_path=str(tmp_path),
        num_groups_to_log=0,
        rollout_json_export=False,
        remove_constant_reward_groups=True,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=groups_per_batch, num_minibatches=num_minibatches
        ),
        async_config=(
            AsyncConfig(max_steps_off_policy=0, groups_per_batch=groups_per_batch)
            if loop == "async"
            else None
        ),
    )
    tracker = CostTracker(RATES)
    ml_logger = ml_log.setup_logging(log_dir=str(tmp_path), do_configure_logging_module=False)
    try:
        checkpoint_mgr = checkpoint_utils.CheckpointManager(
            training_client=cast(tinker.TrainingClient, training_client),
            service_client=cast(tinker.ServiceClient, None),
            log_path=str(tmp_path),
            save_every=0,
            store=ml_logger.store,
        )
        run = (
            train.do_async_training
            if loop == "async"
            else train.do_sync_training_with_stream_minibatch
        )
        asyncio.run(
            asyncio.wait_for(
                run(
                    start_batch=0,
                    end_batch=len(batches),
                    num_batches=len(batches),
                    config=config,
                    training_client=cast(tinker.TrainingClient, training_client),
                    kl_reference_client=None,
                    evaluators=[],
                    dataset=_ListDataset(dataset_batches),
                    ml_logger=ml_logger,
                    tokenizer=cast(Any, _FakeTokenizer()),
                    checkpoint_mgr=checkpoint_mgr,
                    cost_tracker=tracker,
                ),
                timeout=30.0,
            )
        )
    finally:
        ml_logger.close()
    with open(f"{tmp_path}/metrics.jsonl") as f:
        rows = [json.loads(line) for line in f]
    return harness, tracker, rows


@pytest.fixture(autouse=True)
def _default_log_interval(monkeypatch):
    monkeypatch.setattr(train, "DISCARDED_WORK_LOG_INTERVAL_SECONDS", 60.0)


def _step_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if "tokens/trained" in r]


def test_streaming_all_constant_batch_is_recorded_without_an_update(tmp_path):
    harness, tracker, rows = _run(
        tmp_path, [[True, True, True, True]], loop="sync", groups_per_batch=4
    )

    assert harness.optim_steps == 0
    assert harness.trained_records == []
    # The step consumed no groups, so its own counts are zero; the four groups'
    # sampling is in the totals, as discarded.
    (row,) = _step_rows(rows)
    assert row["tokens/sampled"] == 0
    assert row["tokens/trained"] == 0
    assert row["cost/step_usd"] == 0
    assert row["tokens/cumulative/sampled"] == 4 * GROUP_SIZE
    assert row["cost/cumulative_usd"] == pytest.approx(4 * GROUP_USD)
    assert tracker.discarded == TokenUsage(sampled=4 * GROUP_SIZE, prefill_uncached=4 * GROUP_SIZE)
    assert tracker.cumulative_metrics()["cost/discarded_usd"] == pytest.approx(4 * GROUP_USD)


def test_streaming_mixed_then_all_constant_batch(tmp_path):
    harness, tracker, rows = _run(
        tmp_path,
        [[False, True, True, False], [True, True, True, True]],
        loop="sync",
        groups_per_batch=4,
    )

    assert harness.optim_steps == 1
    assert sorted({p for p, _, _ in harness.trained_records}) == [0, 3]
    first, second = _step_rows(rows)
    # The first step counts only the two groups it trained; the dropped ones are
    # in the totals but not in any step.
    assert first["tokens/sampled"] == 2 * GROUP_SIZE
    assert first["tokens/prefill_uncached"] == 2 * GROUP_SIZE
    assert first["tokens/trained"] == 2 * GROUP_SIZE  # one datum per trajectory, 1 input token each
    assert first["tokens/cumulative/sampled"] == 4 * GROUP_SIZE
    assert second["tokens/sampled"] == 0
    assert second["tokens/trained"] == 0
    assert second["tokens/cumulative/sampled"] == 8 * GROUP_SIZE
    assert tracker.discarded.sampled == 6 * GROUP_SIZE
    assert tracker.cumulative.trained == first["tokens/trained"]


def test_streaming_every_batch_constant_keeps_the_totals(tmp_path):
    harness, tracker, rows = _run(
        tmp_path, [[True, True], [True, True], [True, True]], loop="sync", groups_per_batch=2
    )

    assert harness.optim_steps == 0
    step_rows = _step_rows(rows)
    assert [r["tokens/sampled"] for r in step_rows] == [0, 0, 0]
    assert [r["tokens/cumulative/sampled"] for r in step_rows] == [
        2 * GROUP_SIZE,
        4 * GROUP_SIZE,
        6 * GROUP_SIZE,
    ]
    assert step_rows[-1]["cost/cumulative_usd"] == pytest.approx(6 * GROUP_USD)
    assert tracker.discarded.sampled == 6 * GROUP_SIZE


def test_async_all_constant_groups_are_logged_while_no_step_runs(tmp_path, monkeypatch):
    # Interval 0: every drop may log the totals.
    monkeypatch.setattr(train, "DISCARDED_WORK_LOG_INTERVAL_SECONDS", 0.0)
    harness, tracker, rows = _run(
        tmp_path, [[True, True], [True, True]], loop="async", groups_per_batch=2
    )

    assert harness.optim_steps == 0
    assert _step_rows(rows) == []
    # The totals were published as they grew, from the dropped groups alone, as
    # step-less summary rows (a step row would wait for the step to advance).
    summary_rows = [r for r in rows if "tokens/discarded/sampled" in r]
    progress = [r["tokens/discarded/sampled"] for r in summary_rows]
    assert progress and progress[-1] == 4 * GROUP_SIZE
    assert progress == sorted(progress)
    assert all("step" not in r for r in summary_rows)
    assert summary_rows[-1]["tokens/cumulative/sampled"] == 4 * GROUP_SIZE
    assert summary_rows[-1]["cost/discarded_usd"] == pytest.approx(4 * GROUP_USD)
    assert all("tokens/sampled" not in r for r in rows)
    assert tracker.discarded.sampled == 4 * GROUP_SIZE


def test_async_interleaved_discards_do_not_leak_into_steps(tmp_path):
    harness, tracker, rows = _run(
        tmp_path,
        [[False, True], [True, False], [False, False]],
        loop="async",
        groups_per_batch=2,
        num_minibatches=1,
    )

    # Three steps' worth of kept groups (4) were trained; two groups were dropped.
    assert sorted({p for p, _, _ in harness.trained_records}) == [0, 3, 4, 5]
    step_rows = _step_rows(rows)
    assert sum(r["tokens/sampled"] for r in step_rows) == 4 * GROUP_SIZE
    assert sum(r["tokens/trained"] for r in step_rows) == 4 * GROUP_SIZE
    assert step_rows[-1]["tokens/cumulative/sampled"] == 6 * GROUP_SIZE
    assert tracker.discarded.sampled == 2 * GROUP_SIZE
    assert tracker.cumulative.sampled == 6 * GROUP_SIZE


def test_discarded_work_log_respects_the_interval():
    logged: list[dict[str, Any]] = []
    now = [0.0]
    logger = cast(Any, type("L", (), {"log_summary": lambda self, m: logged.append(m)})())
    tracker = CostTracker(RATES)
    log = train.DiscardedWorkLog(tracker, logger, interval_seconds=10.0, clock=lambda: now[0])

    tracker.record_discarded_sampling(TokenUsage(sampled=5))
    log.discarded(step=1)  # too soon after construction
    now[0] = 10.0
    log.discarded(step=1)
    assert logged == [tracker.cumulative_metrics()]
    assert logged[0]["tokens/discarded/sampled"] == 5
    now[0] = 15.0
    log.step_logged()  # a step just logged: restart the interval
    now[0] = 20.0
    log.discarded(step=2)
    assert len(logged) == 1
    now[0] = 25.0
    log.discarded(step=2)
    assert len(logged) == 2
    train.DiscardedWorkLog(None, logger).discarded(step=3)  # no tracker: nothing to log
    assert len(logged) == 2
