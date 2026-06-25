"""Trainer-side grading: run the candidate's ``solve`` on hidden inputs.

The production app never sees the answer key. The trainer holds hidden test inputs for
each task, calls the candidate's ``forge solve(...)`` on each, and scores the outputs.
Grading on inputs the model never saw is what stops the Experiment 1 reward hack: a
program that just emits a constant cannot match multiple distinct expected outputs.

The reward is shaped (parses -> runs clean -> emits -> fraction of hidden inputs correct)
so the policy gets gradient even before it is fully correct. ``RillTask`` lives in
``tasks.py``; this module only scores.
"""

from __future__ import annotations

import re

from tinker_cookbook.recipes.rill_rl.agent_app.rill_lang import run_rill
from tinker_cookbook.recipes.rill_rl.training.tasks import RillTask

W_PARSE, W_RUN, W_NONEMPTY, W_CORRECT = 0.15, 0.15, 0.10, 0.60

# A top-level emit statement (column 0). Indented emits inside `solve` are kept; we strip
# only the model's own top-level test prints so the grader can call solve on its inputs.
_TOP_EMIT = re.compile(r"^emit\b")


def _strip_top_level_emits(program: str) -> str:
    return "\n".join(line for line in program.splitlines() if not _TOP_EMIT.match(line))


def shaped_reward(program: str, task: RillTask) -> tuple[float, dict]:
    """Score ``program`` in [0, 1] by calling its ``solve`` on the task's hidden inputs."""
    body = _strip_top_level_emits(program)
    n = len(task.tests)
    ran_ok = 0
    correct = 0
    any_output = False
    first_output = ""
    first_error: str | None = None

    for i, (args, expected) in enumerate(task.tests):
        res = run_rill(f"{body}\nemit solve({args})", max_steps=task.max_steps)
        if i == 0:
            first_output, first_error = res.output, res.error
        # A parse error means the candidate program itself is malformed -> zero.
        if res.error and res.error.startswith("parse:"):
            return 0.0, {
                "error": res.error,
                "correct": False,
                "frac_correct": 0.0,
                "frac_ran": 0.0,
                "output": res.output,
            }
        if res.ok:
            ran_ok += 1
        if res.output:
            any_output = True
        if res.ok and res.output == expected:
            correct += 1

    frac_ran = ran_ok / n
    frac_correct = correct / n
    reward = (
        W_PARSE + W_RUN * frac_ran + (W_NONEMPTY if any_output else 0.0) + W_CORRECT * frac_correct
    )
    info = {
        "error": first_error,
        "correct": correct == n,
        "frac_correct": round(frac_correct, 4),
        "frac_ran": round(frac_ran, 4),
        "output": first_output,
        "reward": round(reward, 4),
    }
    return reward, info
