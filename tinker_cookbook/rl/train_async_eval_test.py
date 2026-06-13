"""Tests for the async-eval dispatcher in ``rl/train.py``."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.rl.train import (
    _AsyncEvalDispatcher,
    get_async_eval_dispatcher,
    run_evaluations_parallel,
    set_async_eval_dispatcher,
)


@pytest.mark.asyncio
async def test_schedule_returns_sentinel_immediately():
    """schedule() must return a sentinel dict without awaiting the evaluator."""
    ml_logger = MagicMock()
    dispatcher = _AsyncEvalDispatcher(ml_logger=ml_logger)

    started_event = asyncio.Event()
    release_event = asyncio.Event()

    async def slow_run(evaluator, config, i_batch, sampling_client, label, store=None):
        started_event.set()
        await release_event.wait()
        return {"score": 0.5}

    with patch("tinker_cookbook.rl.train.run_single_evaluation", side_effect=slow_run):
        sentinel = dispatcher.schedule([MagicMock(name="ev")], MagicMock(), MagicMock(), i_batch=7)
        # schedule() must return immediately. The eval coroutine may or may not have
        # actually started yet (it's a freshly-created task).
        assert sentinel.get("async_eval/in_flight") == 1.0
        assert any(k.endswith("/scheduled") for k in sentinel)
        ml_logger.log_metrics.assert_not_called()

        release_event.set()
        await dispatcher.drain(timeout=5.0)

    ml_logger.log_metrics.assert_called_once()
    call = ml_logger.log_metrics.call_args
    assert call.args[0] == {"score": 0.5}
    assert call.kwargs.get("step") == 7


@pytest.mark.asyncio
async def test_drain_awaits_pending_tasks_and_logs_at_captured_step():
    """drain() must wait for in-flight evals and log each at the captured i_batch."""
    ml_logger = MagicMock()
    dispatcher = _AsyncEvalDispatcher(ml_logger=ml_logger)

    captured_steps: list[int] = []

    async def quick_run(evaluator, config, i_batch, sampling_client, label, store=None):
        captured_steps.append(i_batch)
        await asyncio.sleep(0)
        return {"score": float(i_batch)}

    with patch("tinker_cookbook.rl.train.run_single_evaluation", side_effect=quick_run):
        # Schedule three batches at different steps.
        dispatcher.schedule([MagicMock()], MagicMock(), MagicMock(), i_batch=10)
        dispatcher.schedule([MagicMock()], MagicMock(), MagicMock(), i_batch=20)
        dispatcher.schedule([MagicMock()], MagicMock(), MagicMock(), i_batch=30)
        assert len(dispatcher._pending) == 3
        await dispatcher.drain()

    assert sorted(captured_steps) == [10, 20, 30]
    assert ml_logger.log_metrics.call_count == 3
    logged_steps = {c.kwargs["step"] for c in ml_logger.log_metrics.call_args_list}
    assert logged_steps == {10, 20, 30}


@pytest.mark.asyncio
async def test_run_evaluations_parallel_routes_through_dispatcher_when_installed():
    """When a dispatcher is installed via the ContextVar, run_evaluations_parallel
    must consult it instead of awaiting evaluators inline."""
    ml_logger = MagicMock()
    dispatcher = _AsyncEvalDispatcher(ml_logger=ml_logger)
    set_async_eval_dispatcher(dispatcher)

    try:
        # If the dispatcher wasn't consulted, run_single_evaluation would be awaited
        # immediately and ml_logger would NOT be called (since it's only called from
        # inside the dispatcher's done-callback).
        async def slow_run(*args, **kwargs):
            await asyncio.sleep(60)
            return {}

        with patch("tinker_cookbook.rl.train.run_single_evaluation", side_effect=slow_run):
            result = await run_evaluations_parallel(
                [MagicMock()], MagicMock(), MagicMock(), i_batch=3
            )
        # Sentinel keys mean we went through the dispatcher.
        assert "async_eval/in_flight" in result
        assert len(dispatcher._pending) == 1
        for task in list(dispatcher._pending):
            task.cancel()
    finally:
        set_async_eval_dispatcher(None)


@pytest.mark.asyncio
async def test_get_set_dispatcher_round_trip():
    """ContextVar install/clear works."""
    assert get_async_eval_dispatcher() is None
    dispatcher = _AsyncEvalDispatcher(ml_logger=MagicMock())
    set_async_eval_dispatcher(dispatcher)
    try:
        assert get_async_eval_dispatcher() is dispatcher
    finally:
        set_async_eval_dispatcher(None)
    assert get_async_eval_dispatcher() is None


@pytest.mark.asyncio
async def test_eval_failure_is_swallowed_with_warning():
    """A failing evaluator must not kill training: the exception is logged, not raised."""
    ml_logger = MagicMock()
    dispatcher = _AsyncEvalDispatcher(ml_logger=ml_logger)

    async def boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    with patch("tinker_cookbook.rl.train.run_single_evaluation", side_effect=boom):
        dispatcher.schedule([MagicMock()], MagicMock(), MagicMock(), i_batch=1)
        await dispatcher.drain()
    # ml_logger never received a real metric write
    ml_logger.log_metrics.assert_not_called()
