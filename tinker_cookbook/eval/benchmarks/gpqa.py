"""GPQA-Diamond benchmark -- graduate-level science QA (multiple choice).

Dataset: ``Idavidrein/gpqa`` (gpqa_diamond config) on HuggingFace.
Metric: Multiple-choice accuracy (A/B/C/D).
Pattern: Single-turn generate + programmatic grading.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult


def _extract_mcq_answer(text: str) -> str:
    """Extract a multiple-choice letter (A-D) from a model response."""
    idx = text.find("\\boxed{")
    if idx != -1:
        start = idx + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            boxed = text[start : i - 1].strip().upper()
            if re.fullmatch(r"[ABCD]", boxed):
                return boxed

    answer_match = re.search(
        r"(?:answer is|answer:)\s*\(?([ABCD])\)?", text, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).upper()

    letters = re.findall(r"\b([ABCD])\b", text[-300:])
    if letters:
        return letters[-1]
    return ""


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class GPQAEnv(Env):
    """Single-turn env for one GPQA-Diamond question."""

    def __init__(self, prompt: str, expected: str, renderer: Renderer):
        self.prompt = prompt
        self.expected = expected
        self.renderer = renderer

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = self.renderer.tokenizer.decode(action)
        extracted = _extract_mcq_answer(response)
        correct = extracted == self.expected
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct)},
            logs={
                "input": self.prompt[:200],
                "expected": self.expected,
                "extracted": extracted,
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class GPQABenchmarkBuilder(BenchmarkBuilder):
    """GPQA-Diamond: graduate-level science multiple-choice (198 questions)."""

    name = "gpqa"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))
        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        envs = []
        for row in ds:
            question = row["Question"]
            correct_answer = row.get("Answer", row.get("Correct Answer", ""))

            choice_cols = [
                col for col in row.keys()
                if col.startswith("Choice") or col in ("choice_a", "choice_b", "choice_c", "choice_d")
            ]

            if choice_cols:
                choices = [row[c] for c in sorted(choice_cols) if row.get(c)]
            else:
                choices = [row.get("Correct Answer", "")]
                for i in range(1, 4):
                    inc = row.get(f"Incorrect Answer {i}", "")
                    if inc:
                        choices.append(inc)

            if not choices:
                continue

            if correct_answer in ("A", "B", "C", "D"):
                expected = correct_answer
            else:
                expected = "A"
                for i, c in enumerate(choices):
                    if c.strip() == str(correct_answer).strip():
                        expected = chr(65 + i)
                        break

            choice_text = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
            prompt = (
                f"{question}\n\n{choice_text}\n\n"
                "Think step by step, then give your final answer as a single letter (A, B, C, or D)."
            )
            envs.append(GPQAEnv(prompt, expected, renderer))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(GPQABenchmarkBuilder())
