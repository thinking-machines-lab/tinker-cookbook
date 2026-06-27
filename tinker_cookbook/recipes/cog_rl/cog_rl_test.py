"""Offline tests for the Cog recipe (no network, no GPU, no API keys).

Covers the language interpreter, the standalone agent's self-correction loop (mock
client), the sampling proxy's token capture (mock sampler), trainer-side grading, the
GRPO datum construction, and the task families.
"""

from __future__ import annotations

import asyncio
import types as pytypes
from pathlib import Path

import pytest

from tinker_cookbook.recipes.cog_rl.agent_app import extract_program, run_cog
from tinker_cookbook.recipes.cog_rl.agent_app.agent import CogAgent
from tinker_cookbook.recipes.cog_rl.training.grading import shaped_reward
from tinker_cookbook.recipes.cog_rl.training.proxy import SamplingProxy, TurnCapture
from tinker_cookbook.recipes.cog_rl.training.tasks import (
    EVAL_FAMILIES,
    TRAIN_FAMILIES,
    CogTask,
    build_tasks,
)
from tinker_cookbook.recipes.cog_rl.training.train import _build_datums

# A small solve-based task for grading tests: solve(n) = sum 1..n, graded on hidden inputs.
_SUM_TASK = CogTask(
    name="sum",
    family="sum",
    prompt="define solve(n) = sum 1..n",
    tests=(("3", "6"), ("5", "15"), ("10", "55")),
)
_CORRECT_SOLVE = (
    "forge solve(n) {\n  0 -> s\n  1 -> i\n"
    "  sustain i <= n { s + i -> s\n i + 1 -> i }\n  give s\n}"
)

_EXAMPLES = Path(__file__).parent / "agent_app" / "cog_lang" / "examples"


# ---- language ----


def test_interpreter_examples_run():
    expected = {
        "primes": "\n".join(["2", "3", "5", "7", "11", "13", "17", "19", "23", "29"]),
        "wordrev": "delta gamma beta alpha",
    }
    for name, want in expected.items():
        res = run_cog((_EXAMPLES / f"{name}.cog").read_text())
        assert res.ok and res.output == want


def test_error_categories():
    assert run_cog("0 -> s walk").error.startswith("parse:")
    assert run_cog("emit (1 / 0)").error.startswith("runtime:")
    assert run_cog("sustain yes { 1 -> x }", max_steps=100).error.startswith("budget:")


def test_trace_is_separate_from_emit_output():
    res = run_cog('trace "dbg=" + 7\nemit 42')
    assert res.ok
    assert res.output == "42"  # emit is program output
    assert res.trace == "dbg=7"  # trace is a separate debug channel
    # trace survives a later error (partial debug output preserved)
    res2 = run_cog('trace "step1"\nemit (1 / 0)')
    assert res2.error.startswith("runtime:") and res2.trace == "step1"


def test_run_cog_endpoint():
    from fastapi.testclient import TestClient

    from tinker_cookbook.recipes.cog_rl.agent_app import server

    client = TestClient(server.app)
    assert client.get("/playground").status_code == 200
    r = client.post("/api/run_cog", json={"program": 'trace "x"\nemit 1 + 2'})
    d = r.json()
    assert d["ok"] and d["output"] == "3" and d["trace"] == "x" and d["steps"] > 0


def test_stray_halt_is_runtime_error_not_crash():
    # A `halt` with no enclosing loop must return a runtime error, not raise (it crashed
    # an Experiment 3 training run before this was guarded).
    res = run_cog("when yes { halt }")
    assert not res.ok and res.error.startswith("runtime:")
    res2 = run_cog("forge solve(n) { when n > 0 { halt } give n }\nemit solve(5)")
    assert not res2.ok and res2.error.startswith("runtime:")


def test_extract_program():
    assert extract_program("hi\n```cog\nemit 1\n```\nbye") == "emit 1"
    assert extract_program("```\nnope\n```\n```cog\nemit 2\n```") == "emit 2"
    assert extract_program("emit 3") == "emit 3"


# ---- trainer-side grading ----


def test_shaped_reward_grades_on_hidden_inputs():
    r_ok, info = shaped_reward(_CORRECT_SOLVE, _SUM_TASK)
    assert info["correct"] and r_ok == pytest.approx(1.0)
    # parse error -> zero
    r_parse, _ = shaped_reward("forge solve(n) { give", _SUM_TASK)
    assert r_parse == 0.0
    # runs but wrong on all inputs -> partial (parse+run), not correct, below correct
    r_wrong, info_w = shaped_reward("forge solve(n) { give n }", _SUM_TASK)
    assert not info_w["correct"] and r_parse < r_wrong < r_ok


def test_constant_emit_does_not_reward_hack():
    """The Experiment 1 hack — emitting the literal answer — must fail under hidden-input
    grading: a constant can match at most one of several distinct expected outputs."""
    r_const, info = shaped_reward("emit 6", _SUM_TASK)  # correct only for n=3
    assert not info["correct"]
    assert info["frac_correct"] < 1.0
    assert r_const < shaped_reward(_CORRECT_SOLVE, _SUM_TASK)[0]


# ---- task families ----


def test_tasks_solvable_and_families_disjoint():
    assert not (set(TRAIN_FAMILIES) & set(EVAL_FAMILIES))
    train, eval_ = build_tasks(seed=0)
    assert len(train) >= 15 and len(eval_) >= 4
    assert {t.family for t in train} == set(TRAIN_FAMILIES)
    assert {t.family for t in eval_} == set(EVAL_FAMILIES)
    # every task has hidden tests with at least two distinct expected outputs
    for t in train + eval_:
        assert len(t.tests) >= 2 and len({e for _, e in t.tests}) >= 2


# ---- the standalone agent loop (mock OpenAI client) ----


def _fake_client(responses: list[str]):
    seq = iter(responses)

    class Completions:
        async def create(self, **kw):
            content = next(seq)
            msg = pytypes.SimpleNamespace(content=content)
            return pytypes.SimpleNamespace(choices=[pytypes.SimpleNamespace(message=msg)])

    return pytypes.SimpleNamespace(chat=pytypes.SimpleNamespace(completions=Completions()))


def test_agent_self_corrects_on_interpreter_error():
    async def run():
        agent = CogAgent(model="fake", max_turns=3)
        # turn 1 fails to parse, turn 2 runs clean
        agent._client = lambda: _fake_client(["emit 1 +", "```cog\nemit 1 + 2\n```"])
        res = await agent.solve("emit 3")
        assert res.turns == 2 and res.ran_clean
        assert res.program == "emit 1 + 2" and res.output == "3"
        kinds = [e["type"] for e in res.transcript]
        assert kinds == ["start", "assistant", "run", "fix_request", "assistant", "run", "done"]

    asyncio.run(run())


# ---- sampling proxy: render -> sample -> capture (mock sampler) ----


def test_proxy_samples_and_captures_tokens():
    from fastapi.testclient import TestClient

    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer("Qwen/Qwen3.5-4B")
    renderer = renderers.get_renderer("qwen3", tok)
    program = "```cog\nemit 42\n```"

    class FakeSampling:
        async def sample_async(self, prompt, num_samples, sampling_params):
            toks = tok.encode(program, add_special_tokens=False)
            seq = pytypes.SimpleNamespace(tokens=toks, logprobs=[-0.1] * len(toks))
            return pytypes.SimpleNamespace(sequences=[seq])

    proxy = SamplingProxy(renderer, default_max_tokens=64)
    proxy.set_policy(FakeSampling(), temperature=1.0)
    client = TestClient(proxy.app)
    body = {
        "model": "cog",
        "messages": [{"role": "user", "content": "emit 42"}],
        "max_tokens": 64,
    }
    resp = client.post("/v1/rid-1/chat/completions", json=body)
    assert resp.status_code == 200
    assert "emit 42" in resp.json()["choices"][0]["message"]["content"]
    caps = proxy.pop_captures()
    assert list(caps.keys()) == ["rid-1"]
    cap = caps["rid-1"][0]
    assert len(cap.prompt_tokens) > 0
    assert len(cap.sampled_tokens) == len(cap.logprobs) > 0
    assert proxy.pop_captures() == {}  # cleared after pop


# ---- GRPO datum construction ----


def test_build_datums_shapes_and_advantage_placement():
    caps = [
        TurnCapture(
            prompt_tokens=[10, 11, 12],
            sampled_tokens=[20, 21, 22, 23],
            logprobs=[-0.1, -0.2, -0.3, -0.4],
        ),
        TurnCapture(prompt_tokens=[10, 11, 12, 30], sampled_tokens=[40, 41], logprobs=[-0.5, -0.6]),
    ]
    datums = _build_datums(caps, advantage=0.7)
    assert len(datums) == 2
    d0 = datums[0]
    adv = d0.loss_fn_inputs["advantages"].to_numpy().tolist()
    tgt = d0.loss_fn_inputs["target_tokens"].to_numpy().tolist()
    # prompt len 3 -> ob_len 2 zeros, then one advantage per sampled token (4).
    assert adv[:2] == [0.0, 0.0] and all(a == pytest.approx(0.7) for a in adv[2:])
    assert tgt == [0, 0, 20, 21, 22, 23]
    assert d0.model_input.length == len(tgt) == len(adv)
    # empty sampled sequence is skipped.
    assert _build_datums([TurnCapture([1, 2], [], [])], 0.5) == []


# ---- corpus task pipeline (offline: pure parsing + the Cog-print adapter) ----


def test_cog_repr_matches_interpreter_to_str():
    from tinker_cookbook.recipes.cog_rl.agent_app.cog_lang.interp import _to_str
    from tinker_cookbook.recipes.cog_rl.training.cog_format import cog_repr

    for v in [0, 7, -3, True, False, None, "abc", "", [1, 2, 3], [], ["x", "y"], [True, None, 5]]:
        assert cog_repr(v) == _to_str(v)


def test_cog_literal_escapes_and_supported_filter():
    from tinker_cookbook.recipes.cog_rl.training.cog_format import cog_args, cog_literal, supported

    assert cog_literal('a"b\\c') == '"a\\"b\\\\c"'
    assert cog_args((5, "hi", [1, 2], True)) == '5, "hi", [1, 2], yes'
    assert supported([1, ["a", False]]) and not supported(1.5)
    assert not supported({"a": 1}) and not supported((1, 2))


def test_corpus_parse_assert_and_build_one():
    from tinker_cookbook.recipes.cog_rl.training import corpus_tasks

    func, params = corpus_tasks._func_def("def add(a, b):\n    return a + b")
    assert func == "add" and params == ["a", "b"]
    assert corpus_tasks._parse_assert("assert add(2, 3) == 5", "add") == ((2, 3), 5)
    assert corpus_tasks._parse_assert("assert is_even(4)", "is_even") == ((4,), True)
    assert corpus_tasks._parse_assert("assert f(x) == 5", "f") is None  # non-literal arg

    row = {
        "task_id": 1,
        "prompt": "Write a function to add two numbers",
        "code": "def add(a, b):\n    return a + b",
        "test_imports": [],
        "test_list": ["assert add(2, 3) == 5", "assert add(0, 0) == 0", "assert add(10, 2) == 12"],
    }
    task = corpus_tasks._build_one(row)
    assert task is not None and task.family == "corpus" and task.name == "mbpp_1"
    assert task.tests == (("2, 3", "5"), ("0, 0", "0"), ("10, 2", "12"))

    # A float output is filtered out (Cog has no float-print contract).
    float_row = dict(
        row,
        code="def add(a, b):\n    return a / b",
        test_list=["assert add(6, 2) == 3.0", "assert add(9, 3) == 3.0"],
    )
    assert corpus_tasks._build_one(float_row) is None
