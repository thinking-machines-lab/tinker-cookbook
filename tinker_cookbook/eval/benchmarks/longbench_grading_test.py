"""Regression tests for LongBench multiple-choice grading."""

import asyncio
from typing import cast

import pytest

from tinker_cookbook.eval.benchmarks import _longbench
from tinker_cookbook.renderers import Renderer


@pytest.mark.parametrize(
    ("response", "expected", "reward"),
    [
        ("Final answer: C. I reject option B.", "C", 1.0),
        ("The answer is (A). Options B and D contradict the passage.", "A", 1.0),
        ("\\boxed{D} because B is ruled out in section 2.", "D", 1.0),
        ("B", "B", 1.0),
        ("The answer is A. Wait, actually the answer is B.", "B", 1.0),
        ("Option A looks wrong and B is weak, so the answer is C.", "C", 1.0),
        ("Final answer: C. I reject option B.", "B", 0.0),
    ],
)
def test_longbench_mcq_uses_shared_extraction(
    monkeypatch: pytest.MonkeyPatch, response: str, expected: str, reward: float
):
    monkeypatch.setattr(_longbench, "decode_response", lambda action, renderer: response)
    env = _longbench.LongBenchEnv(
        user_content="synthetic question",
        expected=expected,
        subtask="synthetic",
        is_mcq=True,
        renderer=cast(Renderer, None),
    )
    result = asyncio.run(env.step([]))
    assert result.reward == reward
    assert result.episode_done
