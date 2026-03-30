"""SWE-bench Verified benchmark -- software engineering patch generation.

Dataset: ``princeton-nlp/SWE-bench_Verified`` on HuggingFace (500 problems).
Metric: Fraction of problems where an LLM judge rates patch quality > 0.5.
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
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig, BenchmarkResult
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)

_CANDIDATE_PROMPT_TEMPLATE = """\
You are a software engineer working on a Python repository. A bug report or \
feature request has been filed. Your task is to generate a unified diff patch \
that resolves the issue.

## Repository: {repo}

## Problem Statement
{problem_statement}

## Hints
{hints}

Please generate a unified diff patch (```diff ... ```) that fixes the issue. \
Only output the patch, no explanation needed."""

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert software engineering reviewer. Evaluate the quality of a \
proposed patch for the following issue.

## Repository: {repo}

## Problem Statement
{problem_statement}

## Proposed Patch
{patch}

Rate the patch quality on a scale of 0.0 to 1.0 considering:
- Does the patch address the core issue described?
- Is the patch syntactically valid as a unified diff?
- Does it modify plausible files/locations?
- Would it likely pass the test suite?

Provide a brief explanation, then give your score in this exact format: \
"Score: [[X.X]]", for example: "Score: [[0.7]]"."""


def _extract_patch(text: str) -> str:
    """Extract a unified diff patch from a model response."""
    match = re.search(r"```diff\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_judge_score(judge_response: str) -> float | None:
    """Extract a numeric score from the judge response like ``[[0.7]]``."""
    match = re.search(r"\[\[([0-9]+(?:\.[0-9]+)?)\]\]", judge_response)
    if match:
        return float(match.group(1))
    match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", judge_response, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class SWEBenchEnv(Env):
    """Single-turn env for one SWE-bench problem, graded by an LLM judge."""

    def __init__(
        self,
        prompt: str,
        repo: str,
        problem_statement: str,
        instance_id: str,
        judge_completer: TinkerMessageCompleter,
        renderer: Renderer,
    ):
        self.prompt = prompt
        self.repo = repo
        self.problem_statement = problem_statement
        self.instance_id = instance_id
        self.judge_completer = judge_completer
        self.renderer = renderer

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        answer = self.renderer.tokenizer.decode(action)
        patch = _extract_patch(answer)

        # Judge the patch
        judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
            repo=self.repo,
            problem_statement=self.problem_statement[:4000],
            patch=patch[:4000],
        )
        judge_messages: list[Message] = [{"role": "user", "content": judge_prompt}]
        try:
            from tinker_cookbook import renderers
            judge_response = await self.judge_completer(judge_messages)
            judge_text = renderers.get_text_content(judge_response)
            score = _extract_judge_score(judge_text)
        except Exception as e:
            logger.warning(f"SWE-bench judge failed: {e}")
            score = None
            judge_text = ""

        correct = score is not None and score > 0.5
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct), "score": float(score) if score is not None else 0.0},
            logs={
                "input": self.problem_statement[:200],
                "output": patch[:500],
                "judge_output": judge_text[:300],
                "score": score if score is not None else -1,
                "repo": self.repo,
                "instance_id": self.instance_id,
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class SWEBenchBenchmarkBuilder(BenchmarkBuilder):
    """SWE-bench Verified: software engineering patch generation (LLM judge)."""

    name = "swe_bench"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_dataset("princeton-nlp/SWE-bench_Verified", split="test"))
        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        j_client = config.judge_sampling_client
        j_renderer = config.judge_renderer or renderer
        if j_client is None:
            logger.warning(
                "swe_bench: no judge_sampling_client configured, "
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
            repo = row.get("repo", "unknown")
            problem_statement = row.get("problem_statement", "")
            hints = row.get("hints_text", "")
            instance_id = row.get("instance_id", "")
            if not problem_statement:
                continue

            prompt = _CANDIDATE_PROMPT_TEMPLATE.format(
                repo=repo,
                problem_statement=problem_statement[:4000],
                hints=hints[:2000],
            )
            envs.append(SWEBenchEnv(prompt, repo, problem_statement, instance_id, judge_completer, renderer))
        return envs

    def aggregate(self, rewards: list[float], metrics_list: list[dict]) -> BenchmarkResult:
        """Aggregate with accuracy and average score."""
        num_correct = sum(1 for r in rewards if r > 0)
        accuracy = num_correct / len(rewards) if rewards else 0.0

        scores = [m.get("score", 0.0) for m in metrics_list]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkResult(
            name=self.name,
            score=accuracy,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics={
                "swe_bench/accuracy": accuracy,
                "swe_bench/avg_score": avg_score,
            },
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(SWEBenchBenchmarkBuilder())
