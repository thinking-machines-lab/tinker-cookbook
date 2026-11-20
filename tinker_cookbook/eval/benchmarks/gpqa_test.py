"""Tests for how the GPQA benchmark orders its answer choices."""

from typing import cast

import pytest
from datasets import Dataset

from tinker_cookbook.eval.benchmarks import gpqa
from tinker_cookbook.eval.benchmarks._types import BenchmarkConfig
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv


class _Renderer:
    """Only what EnvFromMessageEnv needs at construction time."""

    def get_stop_sequences(self) -> list[str]:
        return []


def _text_answer_rows(n: int) -> list[dict[str, str]]:
    # Same layout as Idavidrein/gpqa: the correct answer has its own column.
    return [
        {
            "Question": f"Question {i}?",
            "Correct Answer": f"right {i}",
            "Incorrect Answer 1": f"wrong {i}a",
            "Incorrect Answer 2": f"wrong {i}b",
            "Incorrect Answer 3": f"wrong {i}c",
        }
        for i in range(n)
    ]


def _build(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, str]]
) -> list[gpqa.GPQAMessageEnv]:
    monkeypatch.setattr(
        gpqa, "load_benchmark_dataset", lambda *args, **kwargs: Dataset.from_list(rows)
    )
    envs = gpqa.GPQABenchmarkBuilder().make_envs(cast(Renderer, _Renderer()), BenchmarkConfig())
    message_envs: list[gpqa.GPQAMessageEnv] = []
    for env in envs:
        assert isinstance(env, EnvFromMessageEnv)
        assert isinstance(env.message_env, gpqa.GPQAMessageEnv)
        message_envs.append(env.message_env)
    return message_envs


def test_expected_letter_labels_the_correct_answer(monkeypatch: pytest.MonkeyPatch):
    rows = _text_answer_rows(40)
    envs = _build(monkeypatch, rows)
    assert len(envs) == len(rows)
    for row, env in zip(rows, envs, strict=True):
        assert f"({env.expected}) {row['Correct Answer']}\n" in env.prompt


def test_correct_answer_is_not_always_option_a(monkeypatch: pytest.MonkeyPatch):
    envs = _build(monkeypatch, _text_answer_rows(40))
    assert {env.expected for env in envs} == {"A", "B", "C", "D"}


def test_choice_order_is_the_same_on_every_build(monkeypatch: pytest.MonkeyPatch):
    rows = _text_answer_rows(10)
    first = [(env.expected, env.prompt) for env in _build(monkeypatch, rows)]
    second = [(env.expected, env.prompt) for env in _build(monkeypatch, rows)]
    assert first == second
