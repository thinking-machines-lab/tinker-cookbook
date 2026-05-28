"""Minimal multi-turn calc-tool environment for the renderer-vs-TITO equivalence demo.

Tool calls are text-encoded inside assistant ``content`` (matching how Llama 3 RL
recipes typically work today, since ``Llama3Renderer`` doesn't render structured
``tool_calls``). The env is deterministic: given a seed, it produces a fixed
four-message conversation, with the assistant turns pre-computed as canonical
text. Both the renderer-driven and TITO-driven rollout paths consume the *same*
message list, so any divergence in the produced training sample is a code-path
difference, not a sampler-difference.

Schema::

    [user]        What is {a} {op} {b}? Use the calc tool then answer.
    [assistant]   <call>{op}({a},{b})</call>
    [user]        <result>{z}</result>
    [assistant]   {z}

This is intentionally tiny — the goal is the equivalence check, not the model
learning anything interesting.
"""

from __future__ import annotations

import operator
import random
from dataclasses import dataclass
from typing import Literal

from tinker_cookbook.renderers.base import Message

Op = Literal["+", "*", "-"]

_OPS: dict[Op, callable] = {
    "+": operator.add,
    "*": operator.mul,
    "-": operator.sub,
}

_OP_NAMES: dict[Op, str] = {"+": "add", "*": "mul", "-": "sub"}


@dataclass(frozen=True)
class Problem:
    a: int
    b: int
    op: Op

    @property
    def answer(self) -> int:
        return _OPS[self.op](self.a, self.b)


def sample_problem(rng: random.Random) -> Problem:
    """Deterministic problem given an rng seeded by the caller."""
    a = rng.randint(1, 50)
    b = rng.randint(1, 50)
    op = rng.choice(["+", "*", "-"])
    return Problem(a=a, b=b, op=op)


def make_problems(n: int, seed: int = 0) -> list[Problem]:
    rng = random.Random(seed)
    return [sample_problem(rng) for _ in range(n)]


def build_canonical_messages(problem: Problem) -> list[Message]:
    """The full four-message rollout, with both assistant turns canonical.

    Both rollout drivers consume this exact list, so any divergence in the
    resulting ``(tokens, weights)`` is a code-path artifact, not a sampler one.
    """
    op_name = _OP_NAMES[problem.op]
    z = problem.answer
    return [
        # Explicit system message: keeps both paths from disagreeing about the
        # default-system header that HF's Llama 3.x chat template auto-injects
        # when none is provided. Cookbook's Llama3Renderer doesn't auto-inject,
        # so equivalence holds only when the caller supplies a system message.
        {
            "role": "system",
            "content": "You are a helpful calculator assistant.",
        },
        {
            "role": "user",
            "content": (
                f"What is {problem.a} {problem.op} {problem.b}? "
                f"Use the calc tool then answer."
            ),
        },
        {
            "role": "assistant",
            "content": f"<call>{op_name}({problem.a},{problem.b})</call>",
        },
        {
            "role": "user",
            "content": f"<result>{z}</result>",
        },
        {
            "role": "assistant",
            "content": str(z),
        },
    ]
