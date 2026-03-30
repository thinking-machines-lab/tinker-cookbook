"""AIME 2025 benchmark -- math competition problems with integer answers (0-999).

Dataset: Tries multiple HuggingFace sources for AIME 2025 problems.
Metric: Accuracy -- fraction of problems where the extracted integer matches ground truth.
Pattern: Single-turn generate + programmatic grading.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)

# Known HuggingFace dataset IDs for AIME 2025 (tried in order)
_AIME_DATASET_IDS = (
    "HuggingFaceH4/aime-2025",
    "yentinglin/aime_2025",
    "Maxwell-Jia/AIME_2025",
    "opencompass/AIME2025",
    "di-zhang-fdu/AIME24-25",
)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _extract_boxed(text: str) -> str | None:
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


def _extract_number(text: str) -> str:
    """Extract a number from text, stripping LaTeX formatting."""
    cleaned = re.sub(r"\\text\{[^}]*\}", "", text)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "").replace("$", "")
    cleaned = cleaned.replace(",", "").replace(" ", "")
    match = re.search(r"[-]?\d+\.?\d*", cleaned)
    return match.group(0) if match else cleaned.strip()


def _extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer as fallback."""
    boxed = _extract_boxed(text)
    if boxed:
        return _extract_number(boxed)
    hash_match = re.search(r"####\s*(.+)", text)
    if hash_match:
        return _extract_number(hash_match.group(1))
    answer_match = re.search(
        r"(?:answer is|answer:)\s*\$?([0-9,.-]+)", text, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip().replace(",", "")
    numbers = re.findall(r"[-]?\d+[,\d]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def _load_aime_dataset() -> Dataset | None:
    for dataset_id in _AIME_DATASET_IDS:
        for split in ("test", "train"):
            try:
                ds = cast(Dataset, load_dataset(dataset_id, split=split))
                logger.info(f"Loaded AIME 2025 from {dataset_id}/{split} ({len(ds)} problems)")
                return ds
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class AIMEEnv(Env):
    """Single-turn env for one AIME problem."""

    def __init__(self, problem: str, expected: int, renderer: Renderer):
        self.problem = problem
        self.expected = expected
        self.renderer = renderer

    async def initial_observation(self):
        prompt = (
            f"{self.problem}\n\n"
            "This is an AIME problem. The answer is an integer from 000 to 999. "
            "Show your work step by step, then put your final answer in \\boxed{}."
        )
        messages: list[Message] = [{"role": "user", "content": prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = self.renderer.tokenizer.decode(action)
        boxed = _extract_boxed(response)
        extracted_str = _extract_number(boxed) if boxed else _extract_gsm8k_answer(response)
        try:
            extracted_val = int(float(extracted_str))
            correct = extracted_val == self.expected
        except (ValueError, TypeError):
            extracted_val = None
            correct = False
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct)},
            logs={
                "input": self.problem[:200],
                "expected": str(self.expected),
                "extracted": str(extracted_val) if extracted_val is not None else extracted_str,
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class AIMEBenchmarkBuilder(BenchmarkBuilder):
    """AIME 2025: math competition problems with integer answers (0-999)."""

    name = "aime"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = _load_aime_dataset()
        if ds is None:
            logger.warning("Could not load AIME 2025 dataset from HuggingFace.")
            return []

        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        envs = []
        for row in ds:
            problem = row.get("problem", row.get("question", row.get("Problem", "")))
            expected_raw = row.get("answer", row.get("Answer", row.get("expected_answer", "")))
            if not problem or expected_raw is None:
                continue

            try:
                expected = int(str(expected_raw).strip())
            except (ValueError, TypeError):
                m = re.search(r"\d+", str(expected_raw))
                if m:
                    expected = int(m.group(0))
                else:
                    continue

            envs.append(AIMEEnv(problem, expected, renderer))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(AIMEBenchmarkBuilder())
