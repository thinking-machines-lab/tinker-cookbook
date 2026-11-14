"""Tests for InspectEvaluator."""

import asyncio
from typing import Any, cast

import pytest

pytest.importorskip("inspect_ai")

import tinker

from tinker_cookbook.eval import inspect_evaluators
from tinker_cookbook.eval.inspect_evaluators import InspectEvaluatorBuilder


def test_overlapping_evaluations_take_turns_in_eval_async(monkeypatch: pytest.MonkeyPatch):
    """inspect_ai raises if eval_async is called while another call is running,
    so evaluations that overlap (as async RL's can) must take turns."""
    running = 0
    calls = 0

    async def eval_async(**kwargs: Any) -> list[Any]:
        nonlocal running, calls
        if running:
            raise RuntimeError("Multiple concurrent calls to eval_async are not allowed.")
        running += 1
        calls += 1
        await asyncio.sleep(0.01)
        running -= 1
        return []

    monkeypatch.setattr(inspect_evaluators, "eval_async", eval_async)
    monkeypatch.setattr(inspect_evaluators, "InspectAPIFromTinkerSampling", lambda **kwargs: None)
    monkeypatch.setattr(inspect_evaluators, "InspectAIModel", lambda **kwargs: None)
    evaluator = InspectEvaluatorBuilder(
        tasks=[], model_name="fake-model", renderer_name="fake-renderer"
    )()

    async def run() -> None:
        client = cast(tinker.SamplingClient, None)
        await asyncio.gather(evaluator(client), evaluator(client))

    asyncio.run(run())
    assert calls == 2
