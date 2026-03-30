"""Terminal-Bench 2.0 benchmark -- terminal command generation.

Dataset: ``ia03/terminal-bench`` on HuggingFace (89 tasks).
Metric: Fraction of tasks where the LLM judge rates the solution as correct.
Pattern: Single-turn generate + LLM judge grading.

Requires ``config.judge_sampling_client`` for the judge model.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)

_CANDIDATE_PROMPT_TEMPLATE = """\
You are an expert Linux/Unix systems administrator. Solve the following \
terminal task by writing the exact command(s) or script needed.

## Task
{task_description}

{environment_info}

Write a solution script that accomplishes the task. Provide your commands in \
a ```bash code block. Be precise and use standard Unix utilities."""

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator of terminal/shell solutions. Compare a candidate \
solution to a reference solution for the following task.

## Task
{task_description}

## Reference Solution
{reference_solution}

## Candidate Solution
{candidate_solution}

Evaluate whether the candidate solution would correctly accomplish the task. \
Consider:
- Does it achieve the same end result as the reference?
- Are the commands syntactically correct?
- Would it work in a standard Linux environment?
- Minor stylistic differences are acceptable if the result is the same.

Provide a brief explanation, then give your verdict in this exact format: \
"Verdict: [[CORRECT]]" or "Verdict: [[INCORRECT]]"."""


def _extract_script(text: str) -> str:
    """Extract a bash script from a model response."""
    match = re.search(r"```(?:bash|sh|shell)\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_judge_verdict(judge_response: str) -> bool | None:
    """Extract a CORRECT/INCORRECT verdict from the judge response."""
    match = re.search(r"\[\[(CORRECT|INCORRECT)\]\]", judge_response, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "CORRECT"
    match = re.search(r"Verdict:\s*(CORRECT|INCORRECT)", judge_response, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "CORRECT"
    return None


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class TerminalBenchEnv(Env):
    """Single-turn env for one Terminal-Bench task, graded by an LLM judge."""

    def __init__(
        self,
        prompt: str,
        task_description: str,
        reference_solution: str,
        judge_completer: TinkerMessageCompleter,
        renderer: Renderer,
    ):
        self.prompt = prompt
        self.task_description = task_description
        self.reference_solution = reference_solution
        self.judge_completer = judge_completer
        self.renderer = renderer

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        answer = self.renderer.tokenizer.decode(action)
        script = _extract_script(answer)

        if not self.reference_solution:
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=[],
                metrics={"correct": 0.0},
                logs={
                    "input": self.task_description[:200],
                    "output": script[:500],
                    "error": "no reference solution",
                },
            )

        # Judge the solution
        judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
            task_description=self.task_description[:3000],
            reference_solution=self.reference_solution[:3000],
            candidate_solution=script[:3000],
        )
        judge_messages: list[Message] = [{"role": "user", "content": judge_prompt}]
        try:
            from tinker_cookbook import renderers
            judge_response = await self.judge_completer(judge_messages)
            judge_text = renderers.get_text_content(judge_response)
            correct = _extract_judge_verdict(judge_text)
        except Exception as e:
            logger.warning(f"Terminal-Bench judge failed: {e}")
            correct = None
            judge_text = ""

        if correct is None:
            correct = False

        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct)},
            logs={
                "input": self.task_description[:200],
                "output": script[:500],
                "judge_output": judge_text[:300],
                "reference": self.reference_solution[:200],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class TerminalBenchBenchmarkBuilder(BenchmarkBuilder):
    """Terminal-Bench 2.0: terminal task solving (89 tasks).

    TODO: Implement full multi-turn sandbox execution with Harbor framework.
    Current implementation generates a solution script and uses an LLM judge
    to compare against the reference. The full implementation requires a
    terminal sandbox where the model executes commands interactively.
    See https://github.com/laude-institute/terminal-bench for the reference.
    """

    name = "terminal_bench"
    multi_turn = True
    recommended_timeout = 1800

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_dataset("ia03/terminal-bench", split="test"))
        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        j_client = config.judge_sampling_client
        j_renderer = config.judge_renderer or renderer
        if j_client is None:
            logger.warning(
                "terminal_bench: no judge_sampling_client configured, "
                "will fail at grading time. Set config.judge_sampling_client."
            )

        judge_completer = TinkerMessageCompleter(
            sampling_client=j_client,  # type: ignore[arg-type]
            renderer=j_renderer,
            max_tokens=4096,
            temperature=0.0,
        )

        envs = []
        for row in ds:
            task_description = row.get("task", row.get("description", row.get("prompt", "")))
            reference_solution = row.get("solution", row.get("reference", row.get("answer", "")))
            environment_info = row.get("environment", row.get("setup", ""))
            if not task_description:
                continue

            env_section = f"## Environment\n{environment_info}" if environment_info else ""
            prompt = _CANDIDATE_PROMPT_TEMPLATE.format(
                task_description=task_description[:3000],
                environment_info=env_section,
            )
            envs.append(TerminalBenchEnv(
                prompt, task_description, str(reference_solution), judge_completer, renderer
            ))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(TerminalBenchBenchmarkBuilder())
