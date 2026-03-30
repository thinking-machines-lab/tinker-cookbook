"""GSM8K benchmark — grade-school math word problems.

Dataset: ``openai/gsm8k`` (main split, test).
Metric: Accuracy — fraction of problems where the extracted numeric answer matches the ground truth.
Pattern: Single-turn generate + programmatic grading.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import tinker
from datasets import load_dataset

from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult


# ---------------------------------------------------------------------------
# Answer extraction (shared with math benchmarks)
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    r"""Extract content from ``\boxed{...}`` handling nested braces."""
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else None


def extract_number(text: str) -> str:
    """Extract a number from text, stripping LaTeX formatting."""
    cleaned = re.sub(r"\\text\{[^}]*\}", "", text)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "").replace("$", "")
    cleaned = cleaned.replace(",", "").replace(" ", "")
    match = re.search(r"[-]?\d+\.?\d*", cleaned)
    return match.group(0) if match else cleaned.strip()


def extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer from a model response."""
    boxed = extract_boxed(text)
    if boxed:
        return extract_number(boxed)

    hash_match = re.search(r"####\s*(.+)", text)
    if hash_match:
        return extract_number(hash_match.group(1))

    answer_match = re.search(
        r"(?:answer is|answer:)\s*\$?([0-9,.-]+)", text, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip().replace(",", "")

    numbers = re.findall(r"[-]?\d+[,\d]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def check_gsm8k(response: str, expected: str) -> bool:
    """Check if extracted answer matches expected."""
    extracted = extract_gsm8k_answer(response)
    try:
        return abs(float(extracted) - float(expected)) < 1e-5
    except (ValueError, TypeError):
        return extracted.strip() == expected.strip()


# ---------------------------------------------------------------------------
# Env implementation
# ---------------------------------------------------------------------------


class GSM8KEnv(Env):
    """Single-turn env for one GSM8K problem."""

    def __init__(self, question: str, expected: str, renderer: Renderer, example_id: str = ""):
        self.question = question
        self.expected = expected
        self.renderer = renderer
        self.example_id = example_id

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.question}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = self.renderer.tokenizer.decode(action)
        correct = check_gsm8k(response, self.expected)
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct)},
            logs={
                "example_id": self.example_id,
                "input": self.question[:200],
                "expected": self.expected,
                "extracted": extract_gsm8k_answer(response),
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class GSM8KBenchmarkBuilder(BenchmarkBuilder):
    """GSM8K: 1,319 grade-school math word problems."""

    name = "gsm8k"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        envs = []
        for i, row in enumerate(ds):
            expected = row["answer"].split("####")[-1].strip()
            # Stable ID: hash of question text (same question = same ID across runs)
            import hashlib
            example_id = f"gsm8k_{hashlib.md5(row['question'].encode()).hexdigest()[:12]}"
            envs.append(GSM8KEnv(row["question"], expected, renderer, example_id=example_id))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(GSM8KBenchmarkBuilder())
