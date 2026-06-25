"""Trainer-side grading. The production app never sees the expected output; the trainer
holds it and scores each rollout's final program.

The reward is shaped (parses -> runs clean -> emits -> correct, with line-level partial
credit) so the policy gets gradient even early, when exact-match is nearly all-zero on a
brand-new DSL. Uses the vendored interpreter as a library (the language spec); it does not
import the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tinker_cookbook.recipes.rill_rl.agent_app.rill_lang import run_rill

W_PARSE, W_RUN, W_NONEMPTY, W_CORRECT = 0.15, 0.15, 0.10, 0.60


@dataclass(frozen=True)
class RillTask:
    """A verifiable RILL task. ``expect`` is the exact stdout of a reference solution."""

    name: str
    family: str
    prompt: str
    expect: str
    max_steps: int = 50_000
    metadata: dict = field(default_factory=dict)


def _line_overlap(got: str, want: str) -> float:
    want_lines = want.split("\n")
    got_lines = got.split("\n")
    hits = sum(1 for i, w in enumerate(want_lines) if i < len(got_lines) and got_lines[i] == w)
    return hits / max(len(want_lines), 1)


def shaped_reward(program: str, task: RillTask) -> tuple[float, dict]:
    """Run the final program and score it in ``[0, 1]`` against the task's expected output."""
    res = run_rill(program, max_steps=task.max_steps)
    info: dict = {
        "error": res.error,
        "steps": res.steps,
        "output": res.output,
        "correct": False,
        "overlap": 0.0,
    }
    if res.error and res.error.startswith("parse:"):
        info["reward"] = 0.0
        return 0.0, info

    r = W_PARSE
    if res.ok:
        r += W_RUN
        if res.output:
            r += W_NONEMPTY
        if res.output == task.expect:
            r += W_CORRECT
            info["correct"] = True
            info["overlap"] = 1.0
        else:
            overlap = _line_overlap(res.output, task.expect)
            r += W_CORRECT * overlap
            info["overlap"] = overlap
    info["reward"] = round(r, 4)
    return r, info
