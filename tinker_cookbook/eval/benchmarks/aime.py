"""AIME benchmarks -- math competition problems with integer answers (0-999).

Provides separate benchmarks for each year:
- ``aime_2025``: AIME 2025 (30 problems)
- ``aime_2026``: AIME 2026 (30 problems)
- ``aime``: Alias for ``aime_2025`` (backward compatibility)

Datasets: Tries multiple HuggingFace sources per year.
Metric: Accuracy -- fraction of problems where the extracted integer matches ground truth.
Pattern: Single-turn ``MessageEnv`` + programmatic grading.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import cast

from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    build_messages,
    extract_boxed,
    extract_gsm8k_answer,
    extract_number,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env

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
# MessageEnv implementation
# ---------------------------------------------------------------------------


class AIMEMessageEnv(MessageEnv):
    """Single-turn message env for one AIME problem.

    Receives a parsed ``Message`` (thinking already stripped by
    ``EnvFromMessageEnv``), grades the answer, and returns the result.
    """

    def __init__(
        self,
        problem: str,
        expected: int,
        example_id: str = "",
        system_prompt: str | None = None,
    ):
        self.problem = problem
        self.expected = expected
        self.example_id = example_id
        self.system_prompt = system_prompt

    async def initial_observation(self) -> list[Message]:
        prompt = (
            f"{self.problem}\n\n"
            "This is an AIME problem. The answer is an integer from 000 to 999. "
            "Show your work step by step, then put your final answer in \\boxed{}."
        )
        return build_messages(prompt, self.system_prompt)

    async def step(self, message: Message) -> MessageStepResult:
        response = get_text_content(message)
        boxed = extract_boxed(response)
        extracted_str = extract_number(boxed) if boxed else extract_gsm8k_answer(response)
        try:
            extracted_val = int(float(extracted_str))
            correct = extracted_val == self.expected
        except (ValueError, TypeError):
            extracted_val = None
            correct = False
        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
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

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
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
            msg_env = AIMEMessageEnv(
                problem,
                expected,
                example_id=example_id,
                system_prompt=config.system_prompt,
            )
            envs.append(
                EnvFromMessageEnv(
                    renderer=renderer,
                    message_env=msg_env,
                    failed_parse_reward=0.0,
                )
            )
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(_AIMEBenchmarkBuilder("2025", "aime_2025"))
register(_AIMEBenchmarkBuilder("2026", "aime_2026"))
# Backward-compatible alias: "aime" -> AIME 2025
register(_AIMEBenchmarkBuilder("2025", "aime"))
