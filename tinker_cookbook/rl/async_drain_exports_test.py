"""Sampler-only exports during a finite, non-streaming async drain."""

import asyncio
import inspect
from concurrent.futures import Future
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture

from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.utils.cost_estimation import TokenUsage


@dataclass
class _Sampler:
    step: int


@pytest.mark.parametrize(
    "eval_every,save_every,post_kl,groups,hold_workers,has_evaluators,expected_exports",
    [
        (0, 0, False, 6, False, True, []),
        (2, 0, False, 6, False, False, []),
        (2, 0, False, 5, False, False, []),
        (0, 0, False, 6, True, True, [1]),
        (2, 0, False, 6, False, True, [2, 4, 6]),
        (4, 0, False, 6, False, True, [4, 6]),
        (0, 2, False, 6, False, True, [2, 4, 6]),
        (0, 0, True, 6, False, True, [1, 2, 3, 4, 5, 6]),
        (2, 2, False, 6, False, True, [2, 4, 6]),
        (2, 0, False, 5, False, True, [2, 4, 5]),
        (2, 0, False, 0, False, True, []),
    ],
)
def test_completed_workers_only_export_for_consumers(
    tmp_path,
    monkeypatch,
    eval_every,
    save_every,
    post_kl,
    groups,
    hold_workers,
    has_evaluators,
    expected_exports,
):
    exports: list[int] = []
    periodic: list[int] = []
    evaluated: list[int] = []
    post_kl_steps: list[int] = []
    workers_finished = asyncio.Event()
    first_export = asyncio.Event()
    training_started = asyncio.Event()
    rollouts_started = 0
    completed_step = 0
    original_decrement = train._AsyncCounter.decrement_and_get

    async def decrement(counter):
        remaining = await original_decrement(counter)
        if remaining == 0:
            workers_finished.set()
        return remaining

    class Dataset(RLDataset):
        def get_batch(self, index):
            return [MagicMock()] if index < groups else []

        def __len__(self):
            return 6

    async def train_step(**kwargs):
        nonlocal completed_step
        if completed_step == 0:
            assert not workers_finished.is_set()
        training_started.set()
        if not hold_workers or completed_step > 0:
            await workers_finished.wait()
        completed_step += 1
        await asyncio.sleep(0)
        return []

    async def export(**kwargs):
        assert workers_finished.is_set() is not hold_workers
        first_export.set()
        exports.append(completed_step)
        return _Sampler(completed_step)

    def create_sampler(path, **kwargs):
        step = int(path)
        if step:
            exports.append(step)
        future: Future[_Sampler] = Future()
        future.set_result(_Sampler(step))
        return AwaitableConcurrentFuture(future)

    async def save_periodic(*, step, loop_state):
        periodic.append(step)
        return {"sampler_path": str(step)}

    async def evaluate(evaluators, sampling_client, config, step, **kwargs):
        assert sampling_client.step == step
        evaluated.append(step)
        return {"test/reward": float(step)}

    async def compute_post_kl(data, sampling_client):
        post_kl_steps.append(sampling_client.step)
        return {}, TokenUsage()

    async def rollout(*args, **kwargs):
        nonlocal rollouts_started
        rollouts_started += 1
        if rollouts_started > 1:
            await (first_export if hold_workers else training_started).wait()
        return MagicMock()

    monkeypatch.setattr(train._AsyncCounter, "decrement_and_get", decrement)
    monkeypatch.setattr(
        train.checkpoint_utils,
        "save_checkpoint_async",
        AsyncMock(return_value={"sampler_path": "0", "state_path": "0"}),
    )
    monkeypatch.setattr(train, "do_group_rollout_and_filter_constant_reward", rollout)
    monkeypatch.setattr(train, "prepare_minibatch", AsyncMock(return_value=([], {}, TokenUsage())))
    monkeypatch.setattr(train, "train_step", train_step)
    monkeypatch.setattr(train, "compute_kl_sample_train", lambda *args: {})
    monkeypatch.setattr(train, "compute_post_kl", compute_post_kl)
    monkeypatch.setattr(train, "compute_sampling_client_metrics", lambda *args: {})
    monkeypatch.setattr(train, "_maybe_export_rollout_summary_jsonl", lambda **kwargs: None)
    monkeypatch.setattr(train, "run_evaluations_parallel", evaluate)
    training_client = MagicMock()
    training_client.create_sampling_client = create_sampler
    training_client.save_weights_and_get_sampling_client = export
    checkpoint_mgr = MagicMock()
    checkpoint_mgr.should_save_periodic.side_effect = lambda step: (
        save_every > 0 and step % save_every == 0
    )
    checkpoint_mgr.save_periodic_async = save_periodic
    checkpoint_mgr.maybe_save_rolling_async = AsyncMock()
    config = train.Config(
        learning_rate=1e-4,
        dataset_builder=MagicMock(spec=RLDatasetBuilder),
        model_name="test-model",
        recipe_name="test",
        max_tokens=16,
        log_path=str(tmp_path),
        eval_every=eval_every,
        compute_post_kl=post_kl,
        async_config=train.AsyncConfig(max_steps_off_policy=8, groups_per_batch=1),
    )
    asyncio.run(
        train.do_async_training(
            0,
            6,
            6,
            config,
            training_client,
            None,
            [MagicMock()] if has_evaluators else [],
            Dataset(),
            MagicMock(),
            MagicMock(),
            checkpoint_mgr=checkpoint_mgr,
        )
    )
    assert completed_step == groups
    assert exports == expected_exports
    assert periodic == (
        [step for step in range(1, groups + 1) if step % save_every == 0] if save_every else []
    )
    assert post_kl_steps == (list(range(1, groups + 1)) if post_kl else [])
    assert checkpoint_mgr.maybe_save_rolling_async.await_count == groups
    if eval_every and has_evaluators:
        assert evaluated == ([0, *range(eval_every, groups, eval_every), groups] if groups else [0])
    else:
        assert evaluated == []


def test_reuse_state_is_checked_at_export_boundary(monkeypatch):
    old_client = MagicMock()
    checkpoint_mgr = MagicMock()
    checkpoint_mgr.should_save_periodic.return_value = False
    training_client = MagicMock()
    training_client.save_weights_and_get_sampling_client = AsyncMock(return_value=MagicMock())
    workers_finished = asyncio.Event()
    reuse = train.SamplingClientReuse(old_client, workers_finished)
    monkeypatch.setattr(train, "compute_kl_sample_train", lambda *args: {})
    client, _ = asyncio.run(
        train.compute_full_batch_metrics_and_get_sampling_client(
            training_client, checkpoint_mgr, 1, [], [], False, sampling_client_reuse=reuse
        )
    )
    assert client is training_client.save_weights_and_get_sampling_client.return_value
    assert not reuse.reused
    workers_finished.set()
    client, _ = asyncio.run(
        train.compute_full_batch_metrics_and_get_sampling_client(
            training_client, checkpoint_mgr, 2, [], [], False, sampling_client_reuse=reuse
        )
    )
    assert client is old_client
    assert reuse.reused
    assert training_client.save_weights_and_get_sampling_client.await_count == 1


@pytest.mark.parametrize(
    "eval_every,has_evaluators,step,end_step,expected",
    [
        (0, True, 0, None, False),
        (0, True, 5, 5, False),
        (2, False, 2, None, False),
        (2, False, 5, 5, False),
        (2, True, 0, None, True),
        (2, True, 2, None, True),
        (2, True, 3, None, False),
        (2, True, 3, 3, True),
        (2, True, 3, 5, False),
        (2, True, 5, 3, False),
    ],
)
def test_shared_evaluation_predicate(
    monkeypatch, eval_every, has_evaluators, step, end_step, expected
):
    config = MagicMock(eval_every=eval_every)
    evaluators: list[train.SamplingClientEvaluator] = [MagicMock()] if has_evaluators else []
    run_evals = AsyncMock(return_value={"eval/reward": 1.0})
    monkeypatch.setattr(train, "run_evaluations_parallel", run_evals)
    assert train._should_run_evals(evaluators, config, step, end_step=end_step) is expected
    metrics = asyncio.run(
        train._maybe_run_evals(
            evaluators, MagicMock(), config, step, MagicMock(), end_step=end_step
        )
    )
    assert run_evals.await_count == int(expected)
    assert metrics == ({"eval/reward": 1.0} if expected else {})


@pytest.mark.parametrize("helper", [train._should_run_evals, train._maybe_run_evals])
def test_evaluation_endpoint_is_a_required_keyword(helper):
    parameter = inspect.signature(helper).parameters["end_step"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_default_reuse_state_exports(monkeypatch):
    checkpoint_mgr = MagicMock()
    checkpoint_mgr.should_save_periodic.return_value = False
    training_client = MagicMock()
    training_client.save_weights_and_get_sampling_client = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(train, "compute_kl_sample_train", lambda *args: {})
    client, _ = asyncio.run(
        train.compute_full_batch_metrics_and_get_sampling_client(
            training_client, checkpoint_mgr, 1, [], [], False
        )
    )
    assert client is training_client.save_weights_and_get_sampling_client.return_value
    training_client.save_weights_and_get_sampling_client.assert_awaited_once()
