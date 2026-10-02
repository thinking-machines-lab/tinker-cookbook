import asyncio
import json
import logging
import random

import chz
import pytest

from tinker_cookbook.recipes.rl_numerics_check.env import (
    DEFAULT_WORDS,
    CountingProblem,
    CountingReward,
    CountingTaskConfig,
    EpisodeState,
    OneToolCallPerTurnEnv,
    PageReader,
    _warn_turns_not_extending,
    build_env,
    build_prompt,
    closeness_reward,
    count_word,
    parse_answer,
    sample_problem,
    split_pages,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.tokenizer_utils import get_tokenizer

_FILLER = (
    "The river ran between two hills. During the war, however, the town was known "
    "for its mills, which were built before the bridge. Several families lived there "
    "until the flood, when most of them moved away through the valley.\n"
)
CORPUS = _FILLER * 400


def _task(**overrides: object) -> CountingTaskConfig:
    return chz.replace(
        CountingTaskConfig(num_pages_min=3, num_pages_max=5, page_chars=500), **overrides
    )


def _read_call(page: int, call_id: str) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(name="read_page", arguments=json.dumps({"page": page})),
        id=call_id,
    )


def _make_env(
    problem: CountingProblem, max_turns: int
) -> tuple[OneToolCallPerTurnEnv, EpisodeState]:
    state = EpisodeState()
    reader = PageReader(problem.pages, state)
    env = OneToolCallPerTurnEnv(
        tools=[reader.read_page],
        initial_messages=[{"role": "user", "content": problem.prompt}],
        max_turns=max_turns,
        reward_fn=CountingReward(problem, state, tolerance=1.0),
    )
    return env, state


def test_count_word_matches_whole_words_case_insensitively() -> None:
    text = "Time, time's and time-based count; times, lifetime and TIMEs do not. TIME."
    assert count_word(text, "time") == 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I counted.\nAnswer: 42", 42),
        ("Answer: 3 ... actually answer: 1,204", 1204),
        ("**Answer:** 17", 17),
        ("The count is 12.", None),
    ],
)
def test_parse_answer(text: str, expected: int | None) -> None:
    assert parse_answer(text) == expected


def test_closeness_reward() -> None:
    assert closeness_reward(40, 40, tolerance=1.0) == 1.0
    assert closeness_reward(30, 40, tolerance=1.0) == pytest.approx(0.75)
    assert closeness_reward(100, 40, tolerance=1.0) == 0.0
    assert closeness_reward(0, 0, tolerance=1.0) == 1.0
    assert closeness_reward(1, 0, tolerance=1.0) == 0.0
    assert closeness_reward(36, 40, tolerance=0.2) == pytest.approx(0.5)
    assert closeness_reward(30, 40, tolerance=0.2) == 0.0


def test_split_pages_is_contiguous_and_does_not_split_words() -> None:
    pages = split_pages(CORPUS, 0, 6, 500)
    joined = " ".join(pages).split()
    assert joined == CORPUS.split()[: len(joined)]
    assert all(390 <= len(p) <= 520 for p in pages)


def test_sampling_is_deterministic_and_counts_only_document_pages() -> None:
    task = _task()
    a = sample_problem(random.Random(0), CORPUS, task)
    b = sample_problem(random.Random(0), CORPUS, task)
    assert a == b
    assert len(a.pages) == task.max_turns
    assert a.true_count == sum(count_word(p, a.word) for p in a.pages[: a.num_pages])


def test_exact_hints_vary_per_problem_and_appear_in_prompt() -> None:
    task = _task()
    problems = [sample_problem(random.Random(i), CORPUS, task) for i in range(40)]
    assert len({p.word for p in problems}) > 1
    assert len({p.num_pages for p in problems}) > 1
    for p in problems:
        assert f'"{p.word}"' in p.prompt
        assert f"has {p.num_pages} pages" in p.prompt


def test_first_last_hint_varies_the_word_and_hides_it() -> None:
    task = _task(word_hint="first_last")
    problems = [sample_problem(random.Random(i), CORPUS, task) for i in range(40)]
    assert len({p.word for p in problems}) > 1
    for p in problems:
        assert f'"{p.word}"' not in p.prompt
        assert f'starts with "{p.word[0]}" and ends with "{p.word[-1]}"' in p.prompt


def test_hidden_hint_uses_the_single_word_without_revealing_it() -> None:
    task = _task(word_hint="hidden", words=("however",))
    problems = [sample_problem(random.Random(i), CORPUS, task) for i in range(10)]
    assert {p.word for p in problems} == {"however"}
    for p in problems:
        assert "however" not in p.prompt
        assert "starts with" not in p.prompt


def test_default_words_have_distinct_first_last_letters() -> None:
    assert len(DEFAULT_WORDS) == 28
    assert len({(w[0], w[-1]) for w in DEFAULT_WORDS}) == len(DEFAULT_WORDS)


def test_range_hint_hides_shared_page_count() -> None:
    task = _task(num_pages_hint="range", num_pages_min=3, num_pages_max=5, secret_num_pages=4)
    problems = [sample_problem(random.Random(i), CORPUS, task) for i in range(10)]
    assert {p.num_pages for p in problems} == {4}
    for p in problems:
        assert "between 3 and 5 pages" in p.prompt
        assert "has 4 pages" not in p.prompt


def test_hidden_page_count_is_fixed_when_unset() -> None:
    task = _task(num_pages_hint="range")
    assert task.hidden_num_pages == _task(num_pages_hint="range").hidden_num_pages
    assert task.num_pages_min <= task.hidden_num_pages <= task.num_pages_max


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        _task(num_pages_min=6, num_pages_max=5)
    with pytest.raises(ValueError):
        _task(secret_num_pages=9)
    with pytest.raises(ValueError):
        _task(words=("Hello",))
    with pytest.raises(ValueError):
        _task(words=("time", "time"))
    with pytest.raises(ValueError, match="share first and last letters"):
        _task(word_hint="first_last", words=("where", "while"))
    with pytest.raises(ValueError, match="single word"):
        _task(word_hint="hidden")
    with pytest.raises(ValueError, match="matching example"):
        _task(words=("light",))
    _task(word_hint="exact", words=("where", "while"))


def test_prompt_mentions_rules() -> None:
    prompt = build_prompt(_task(), "during", 4)
    assert "exactly once per response" in prompt
    assert "Answer: <count>" in prompt
    assert 'for the word "light"' in prompt


def _run_episode(
    problem: CountingProblem, turns: list[Message], max_turns: int
) -> tuple[float, dict[str, float], EpisodeState]:
    async def run() -> tuple[float, dict[str, float], EpisodeState]:
        env, state = _make_env(problem, max_turns)
        await env.initial_observation()
        result = None
        for message in turns:
            result = await env.step(message)
            if result.episode_done:
                break
        assert result is not None and result.episode_done
        return result.reward, result.metrics, state

    return asyncio.run(run())


def _reading_turns(pages: list[int]) -> list[Message]:
    return [
        {"role": "assistant", "content": "", "tool_calls": [_read_call(p, f"call_{p}")]}
        for p in pages
    ]


def test_reading_every_page_and_answering_correctly_gets_full_reward() -> None:
    task = _task()
    problem = sample_problem(random.Random(1), CORPUS, task)
    answer: Message = {"role": "assistant", "content": f"Answer: {problem.true_count}"}
    turns = _reading_turns(list(range(1, problem.num_pages + 1))) + [answer]
    reward, metrics, _ = _run_episode(problem, turns, task.max_turns)
    assert reward == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["read_all_pages"] == 1.0
    assert metrics["read_past_end"] == 0.0


def test_stopping_early_and_missing_answer() -> None:
    task = _task()
    problem = sample_problem(random.Random(2), CORPUS, task)
    turns = _reading_turns([1]) + [{"role": "assistant", "content": "I am not sure."}]
    reward, metrics, _ = _run_episode(problem, turns, task.max_turns)
    assert reward == 0.0
    assert metrics["answer_parsed"] == 0.0
    assert metrics["read_all_pages"] == 0.0


def test_extra_tool_calls_in_a_turn_are_rejected() -> None:
    task = _task()
    problem = sample_problem(random.Random(3), CORPUS, task)

    async def run() -> tuple[list[Message], EpisodeState, dict[str, float]]:
        env, state = _make_env(problem, task.max_turns)
        await env.initial_observation()
        message: Message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [_read_call(1, "a"), _read_call(2, "b"), _read_call(3, "c")],
        }
        result = await env.step(message)
        assert not result.episode_done
        result = await env.step({"role": "assistant", "content": "Answer: 1"})
        assert result.episode_done
        return env.history, state, result.metrics

    history, state, metrics = asyncio.run(run())
    tool_messages = [m for m in history if m["role"] == "tool"]
    assert [m.get("tool_call_id") for m in tool_messages] == ["a", "b", "c"]
    assert "[Page 1]" in str(tool_messages[0]["content"])
    assert "not executed" in str(tool_messages[1]["content"])
    assert state.pages_read == {1}
    assert metrics["rejected_tool_calls"] == 2


def test_pages_past_the_end_return_text_and_are_tracked() -> None:
    task = _task(num_pages_hint="range", secret_num_pages=3)
    problem = sample_problem(random.Random(4), CORPUS, task)
    wrong_answer = sum(count_word(p, problem.word) for p in problem.pages[:4])
    turns = _reading_turns([1, 2, 3, 4, 99]) + [
        {"role": "assistant", "content": f"Answer: {wrong_answer}"}
    ]
    reward, metrics, state = _run_episode(problem, turns, task.max_turns)
    assert metrics["read_past_end"] == 1.0
    assert state.invalid_page_requests == 1
    assert reward == pytest.approx(closeness_reward(wrong_answer, problem.true_count, 1.0))


@pytest.mark.parametrize(
    ("renderer_name", "sampled_prefix", "expected_rate"),
    [
        ("qwen3_5_disable_thinking", "", 1.0),
        ("qwen3_5_preserve_thinking", "I need page 1.\n</think>\n\n", 1.0),
        ("qwen3_5", "I need page 1.\n</think>\n\n", 0.0),
    ],
)
def test_extension_rate_reflects_whether_turns_merge(
    renderer_name: str, sampled_prefix: str, expected_rate: float
) -> None:
    tokenizer = get_tokenizer("Qwen/Qwen3.6-35B-A3B")
    renderer = get_renderer(renderer_name, tokenizer)
    task = _task()
    problem = sample_problem(random.Random(6), CORPUS, task)
    env = build_env(problem, renderer, task, max_trajectory_tokens=None, max_generation_tokens=None)
    tool_turn = (
        f"{sampled_prefix}Reading page 1.\n\n<tool_call>\n<function=read_page>\n"
        "<parameter=page>\n1\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    answer_turn = f"{sampled_prefix}Answer: {problem.true_count}<|im_end|>"

    async def run() -> dict[str, float | int]:
        await env.initial_observation()
        result = await env.step(tokenizer.encode(tool_turn, add_special_tokens=False))
        assert not result.episode_done
        result = await env.step(tokenizer.encode(answer_turn, add_special_tokens=False))
        assert result.episode_done
        return result.metrics

    assert asyncio.run(run())["extension_rate"] == expected_rate


@pytest.mark.parametrize(("tool_turns", "warns"), [(4, False), (5, True)])
def test_warns_only_when_most_turns_of_a_long_episode_fail_to_extend(
    tool_turns: int, warns: bool, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = get_tokenizer("Qwen/Qwen3.6-35B-A3B")
    renderer = get_renderer("qwen3_5", tokenizer)
    task = _task(num_pages_min=5, num_pages_max=5)
    problem = sample_problem(random.Random(7), CORPUS, task)
    env = build_env(problem, renderer, task, max_trajectory_tokens=None, max_generation_tokens=None)
    think = "I need the next page.\n</think>\n\n"

    async def run() -> None:
        await env.initial_observation()
        for page in range(1, tool_turns + 1):
            turn = (
                f"{think}<tool_call>\n<function=read_page>\n<parameter=page>\n{page}\n"
                "</parameter>\n</function>\n</tool_call><|im_end|>"
            )
            await env.step(tokenizer.encode(turn, add_special_tokens=False))
        await env.step(tokenizer.encode(f"{think}Answer: 1<|im_end|>", add_special_tokens=False))

    _warn_turns_not_extending.cache_clear()
    with caplog.at_level(logging.WARNING):
        asyncio.run(run())
    assert ("did not extend" in caplog.text) == warns
