"""MMLU-Pro benchmark -- multi-task language understanding (professional).

Dataset: ``TIGER-Lab/MMLU-Pro`` on HuggingFace.
Metric: Multiple-choice accuracy (A-J, up to 10 options).
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


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_mcq_answer(text: str, valid_letters: str = "ABCD") -> str:
    """Extract a multiple-choice letter from a model response.

    Tries boxed format, "answer is (X)" pattern, then last standalone letter.
    """
    pattern = f"[{valid_letters}]"

    # Try \boxed{X}
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
            if re.fullmatch(pattern, boxed):
                return boxed

    answer_match = re.search(
        rf"(?:answer is|answer:)\s*\(?([{valid_letters}])\)?",
        text,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).upper()

    letters = re.findall(rf"\b({pattern})\b", text[-300:])
    if letters:
        return letters[-1]
    return ""


# ---------------------------------------------------------------------------
# Env implementation
# ---------------------------------------------------------------------------


class MMLUProEnv(Env):
    """Single-turn env for one MMLU-Pro question."""

    def __init__(
        self,
        prompt: str,
        expected: str,
        valid_letters: str,
        renderer: Renderer,
    ):
        self.prompt = prompt
        self.expected = expected
        self.valid_letters = valid_letters
        self.renderer = renderer

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = self.renderer.tokenizer.decode(action)
        extracted = extract_mcq_answer(response, self.valid_letters)
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


class MMLUProBenchmarkBuilder(BenchmarkBuilder):
    """MMLU-Pro: professional-level multiple-choice questions (up to 10 options)."""

    name = "mmlu_pro"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        try:
            ds = cast(Dataset, load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
        except Exception:
            ds = cast(Dataset, load_dataset("cais/mmlu", "all", split="test"))

        if config.max_examples is not None:
            ds = ds.shuffle(seed=42).select(range(min(config.max_examples, len(ds))))

        valid_letters = "ABCDEFGHIJ"
        envs = []
        for row in ds:
            question = row.get("question", row.get("input", ""))
            choices = row.get("options", row.get("choices", []))
            answer_idx = row.get("answer_index", row.get("answer", None))

            if choices:
                choice_text = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
                prompt = f"{question}\n\n{choice_text}\n\nAnswer with just the letter (A, B, C, D, etc.)."
            else:
                prompt = f"{question}\n\nAnswer with just the letter."

            if isinstance(answer_idx, int):
                expected = chr(65 + answer_idx)
            elif isinstance(answer_idx, str) and len(answer_idx) == 1:
                expected = answer_idx.upper()
            else:
                expected = str(answer_idx).strip().upper()

            envs.append(MMLUProEnv(prompt, expected, valid_letters, renderer))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(MMLUProBenchmarkBuilder())
