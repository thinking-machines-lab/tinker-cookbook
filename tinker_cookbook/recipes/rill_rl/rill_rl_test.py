"""Offline tests for the RILL recipe (no network, no GPU, no API keys).

Covers the language interpreter, the standalone agent's self-correction loop (mock
client), the sampling proxy's token capture (mock sampler), trainer-side grading, the
GRPO datum construction, and the task families.
"""

from __future__ import annotations

import asyncio
import types as pytypes
from pathlib import Path

import pytest

from tinker_cookbook.recipes.rill_rl.agent_app import extract_program, run_rill
from tinker_cookbook.recipes.rill_rl.agent_app.agent import RillAgent
from tinker_cookbook.recipes.rill_rl.training.grading import RillTask, shaped_reward
from tinker_cookbook.recipes.rill_rl.training.proxy import SamplingProxy, TurnCapture
from tinker_cookbook.recipes.rill_rl.training.tasks import (
    EVAL_FAMILIES,
    TRAIN_FAMILIES,
    build_tasks,
)
from tinker_cookbook.recipes.rill_rl.training.train import _build_datums

_EXAMPLES = Path(__file__).parent / "agent_app" / "rill_lang" / "examples"


# ---- language ----


def test_interpreter_examples_run():
    expected = {
        "primes": "\n".join(["2", "3", "5", "7", "11", "13", "17", "19", "23", "29"]),
        "wordrev": "delta gamma beta alpha",
    }
    for name, want in expected.items():
        res = run_rill((_EXAMPLES / f"{name}.rill").read_text())
        assert res.ok and res.output == want


def test_error_categories():
    assert run_rill("0 -> s walk").error.startswith("parse:")
    assert run_rill("emit (1 / 0)").error.startswith("runtime:")
    assert run_rill("sustain yes { 1 -> x }", max_steps=100).error.startswith("budget:")


def test_extract_program():
    assert extract_program("hi\n```rill\nemit 1\n```\nbye") == "emit 1"
    assert extract_program("```\nnope\n```\n```rill\nemit 2\n```") == "emit 2"
    assert extract_program("emit 3") == "emit 3"


# ---- trainer-side grading ----


def test_shaped_reward_ordering():
    task = RillTask("sum", "sum_to_n", "sum 1..10", "55")
    r_ok, info = shaped_reward("0 -> s\nwalk k across range(1, 11) { s + k -> s }\nemit s", task)
    r_off, _ = shaped_reward("0 -> s\nwalk k across range(1, 10) { s + k -> s }\nemit s", task)
    r_empty, _ = shaped_reward("0 -> s\nwalk k across range(1, 11) { s + k -> s }", task)
    r_parse, _ = shaped_reward("0 -> s walk", task)
    assert info["correct"] and r_ok == pytest.approx(1.0)
    assert r_ok > r_off > r_empty > r_parse == 0.0


# ---- task families ----


def test_tasks_solvable_and_families_disjoint():
    assert not (set(TRAIN_FAMILIES) & set(EVAL_FAMILIES))
    train, eval_ = build_tasks(seed=0)
    assert len(train) > 100 and len(eval_) > 30
    assert {t.family for t in train} == set(TRAIN_FAMILIES)
    assert {t.family for t in eval_} == set(EVAL_FAMILIES)
    assert all(t.expect != "" for t in train + eval_)


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
        agent = RillAgent(model="fake", max_turns=3)
        # turn 1 fails to parse, turn 2 runs clean
        agent._client = lambda: _fake_client(["emit 1 +", "```rill\nemit 1 + 2\n```"])
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
    program = "```rill\nemit 42\n```"

    class FakeSampling:
        async def sample_async(self, prompt, num_samples, sampling_params):
            toks = tok.encode(program, add_special_tokens=False)
            seq = pytypes.SimpleNamespace(tokens=toks, logprobs=[-0.1] * len(toks))
            return pytypes.SimpleNamespace(sequences=[seq])

    proxy = SamplingProxy(renderer, default_max_tokens=64)
    proxy.set_policy(FakeSampling(), temperature=1.0)
    client = TestClient(proxy.app)
    body = {
        "model": "rill",
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
