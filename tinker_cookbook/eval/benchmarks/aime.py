"""AIME benchmarks -- math competition problems with integer answers (0-999).

Provides separate benchmarks for each year:
- ``aime_2025``: AIME 2025 (30 problems)
- ``aime_2026``: AIME 2026 (30 problems)
- ``aime``: Alias for ``aime_2025`` (backward compatibility)

Datasets: Tries multiple HuggingFace sources per year.
Metric: Accuracy -- fraction of problems where the extracted integer matches ground truth.
Pattern: Single-turn generate + programmatic grading.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    decode_response,
    extract_boxed,
    extract_gsm8k_answer,
    extract_number,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)

# Pinned dataset source per year (MathArena — verified against published scores)
_AIME_DATASETS: dict[str, str] = {
    "2025": "MathArena/aime_2025",
    "2026": "MathArena/aime_2026",
}


def _load_aime_dataset(year: str) -> Dataset | None:
    """Load AIME dataset for a specific year from MathArena."""
    dataset_id = _AIME_DATASETS.get(year)
    if dataset_id is None:
        logger.warning(f"No known dataset for AIME {year}")
        return None
    for split in ("test", "train"):
        try:
            ds = cast(Dataset, load_benchmark_dataset(dataset_id, split=split))
            logger.info(f"Loaded AIME {year} from {dataset_id}/{split} ({len(ds)} problems)")
            return ds
        except Exception as e:
            logger.debug(f"Failed to load {dataset_id}/{split}: {e}")
            continue
    return None


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class AIMEEnv(Env):
    """Single-turn env for one AIME problem."""

    def __init__(self, problem: str, expected: int, renderer: Renderer, example_id: str = ""):
        self.problem = problem
        self.expected = expected
        self.renderer = renderer
        self.example_id = example_id

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
        response = decode_response(action, self.renderer)
        boxed = extract_boxed(response)
        extracted_str = extract_number(boxed) if boxed else extract_gsm8k_answer(response)
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
                "example_id": self.example_id,
                "input": self.problem[:200],
                "expected": str(self.expected),
                "extracted": str(extracted_val) if extracted_val is not None else extracted_str,
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builders
# ---------------------------------------------------------------------------


class _AIMEBenchmarkBuilder(BenchmarkBuilder):
    """Base builder for AIME benchmarks — parameterized by year."""

    def __init__(self, year: str, benchmark_name: str):
        self._year = year
        self.name = benchmark_name

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = _load_aime_dataset(self._year)
        if ds is None:
            logger.warning(f"Could not load AIME {self._year} dataset from HuggingFace.")
            return []

        ds = limit_dataset(ds, config.max_examples)

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

            example_id = make_example_id(self.name, problem)
            envs.append(AIMEEnv(problem, expected, renderer, example_id=example_id))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(_AIMEBenchmarkBuilder("2025", "aime_2025"))
register(_AIMEBenchmarkBuilder("2026", "aime_2026"))
# Backward-compatible alias: "aime" -> AIME 2025
register(_AIMEBenchmarkBuilder("2025", "aime"))
