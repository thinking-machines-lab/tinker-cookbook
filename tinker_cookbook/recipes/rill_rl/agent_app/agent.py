"""The RILL coding agent.

A plain chat agent: given a task, it asks an OpenAI chat model to write a RILL program,
runs the program through the reference interpreter, and — if the program fails to run —
shows the model the interpreter error and asks it to fix it, up to ``max_turns``.

The agent uses the interpreter as a *tool* to self-correct on errors. It does not know
whether the output is the "right answer" (it has no expected output), and it computes no
reward. That keeps it a realistic production agent: training/eval happens by pointing it
at different model backends, not by reaching into this loop.

Configuration is standard OpenAI: ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` (or a
per-call ``base_url`` / ``api_key`` override so the app can be pointed at any
OpenAI-compatible endpoint).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .program import extract_program
from .prompts import RILL_SYSTEM_PROMPT
from .rill_lang import run_rill


@dataclass
class AgentResult:
    prompt: str
    program: str  # the final program the agent settled on
    output: str  # interpreter stdout of the final program
    error: str | None  # interpreter error category of the final program, if any
    ran_clean: bool  # whether the final program ran without an interpreter error
    turns: int
    transcript: list[dict] = field(default_factory=list)
    api_error: str | None = None


def _error_feedback(error: str) -> str:
    category = error.split(":", 1)[0]
    hint = {
        "parse": (
            "Re-check RILL syntax: assignment is `value -> name`, equality is a single "
            "`=`, blocks use braces, comments start with `~`."
        ),
        "runtime": "Check indexing, types, and division/modulo by zero.",
        "budget": (
            "Your program exceeded the step budget (likely an infinite loop); make sure "
            "the `sustain` condition eventually becomes false."
        ),
    }.get(category, "")
    return (
        f"Running your program failed with `{error}`. {hint}\n"
        "Return the full corrected RILL program in a ```rill code block."
    ).strip()


class RillAgent:
    def __init__(
        self,
        *,
        model: str = "gpt-5.5",
        max_turns: int = 3,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
    ):
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        # Standard OpenAI config: explicit override, else environment.
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)

    def _create_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        return kwargs

    async def iter_solve(self, prompt: str) -> AsyncIterator[dict]:
        """Run the agent on a task, yielding events for streaming UIs."""
        client = self._client()
        messages: list[dict] = [
            {"role": "system", "content": RILL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        yield {"type": "start", "prompt": prompt}

        program, output, error, ran_clean = "", "", None, False
        turn = 0
        for turn in range(1, self.max_turns + 1):
            try:
                resp = await client.chat.completions.create(**self._create_kwargs(messages))
                content = resp.choices[0].message.content or ""
            except Exception as e:
                yield {"type": "api_error", "turn": turn, "message": repr(e)}
                return

            messages.append({"role": "assistant", "content": content})
            program = extract_program(content)
            res = run_rill(program)
            output, error, ran_clean = res.output, res.error, res.ok

            yield {"type": "assistant", "turn": turn, "content": content, "program": program}
            yield {
                "type": "run",
                "turn": turn,
                "ran_clean": ran_clean,
                "output": output,
                "error": error,
            }

            if ran_clean or turn == self.max_turns:
                break
            feedback = _error_feedback(error or "runtime:unknown")
            messages.append({"role": "user", "content": feedback})
            yield {"type": "fix_request", "turn": turn, "content": feedback}

        yield {
            "type": "done",
            "ran_clean": ran_clean,
            "program": program,
            "output": output,
            "turns": turn,
        }

    async def solve(self, prompt: str) -> AgentResult:
        """Run the agent to completion and return the final program and its output."""
        transcript: list[dict] = []
        program, output, error, ran_clean, turns, api_error = "", "", None, False, 0, None
        async for ev in self.iter_solve(prompt):
            transcript.append(ev)
            if ev["type"] == "run":
                output, error, ran_clean, turns = (
                    ev["output"],
                    ev["error"],
                    ev["ran_clean"],
                    ev["turn"],
                )
            elif ev["type"] == "assistant":
                program = ev["program"]
            elif ev["type"] == "api_error":
                api_error, turns = ev["message"], ev["turn"]
        return AgentResult(
            prompt=prompt,
            program=program,
            output=output,
            error=error,
            ran_clean=ran_clean,
            turns=turns,
            transcript=transcript,
            api_error=api_error,
        )
