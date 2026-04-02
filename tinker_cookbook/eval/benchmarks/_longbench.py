"""LongBench v2 benchmark -- long-context comprehension across multiple subtasks.

Dataset: ``THUDM/LongBench-v2`` (or v1 fallback) on HuggingFace.
Metric: Accuracy -- fraction of questions answered correctly.
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
    format_mcq_choices,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
)
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)


def _load_longbench() -> Dataset | None:
    """Load LongBench v2 from THUDM/LongBench-v2."""
    for split in ("test", "train"):
        try:
            ds = cast(Dataset, load_benchmark_dataset("THUDM/LongBench-v2", split=split))
            logger.info(f"Loaded LongBench v2 from THUDM/LongBench-v2/{split} ({len(ds)} examples)")
            return ds
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class LongBenchEnv(Env):
    """Single-turn env for one LongBench question."""

    def __init__(
        self,
        user_content: str,
        expected: str,
        subtask: str,
        is_mcq: bool,
        renderer: Renderer,
        example_id: str = "",
    ):
        self.user_content = user_content
        self.expected = expected
        self.subtask = subtask
        self.is_mcq = is_mcq
        self.renderer = renderer
        self.example_id = example_id

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.user_content}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = decode_response(action, self.renderer)
        if self.is_mcq:
            letters = re.findall(r"\b([A-D])\b", response[-300:])
            extracted = letters[-1] if letters else ""
            correct = extracted == self.expected
        else:
            extracted = response.strip()[:200]
            pattern = r"\b" + re.escape(str(self.expected).strip()) + r"\b"
            correct = bool(re.search(pattern, response, re.IGNORECASE))
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct), "subtask": self.subtask},  # type: ignore[arg-type]
            logs={
                "example_id": self.example_id,
                "input": self.user_content[:200],
                "expected": self.expected,
                "extracted": extracted,
                "subtask": self.subtask,
                "output": response[:200],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class LongBenchBenchmarkBuilder(BenchmarkBuilder):
    """LongBench: long-context comprehension across multiple subtasks."""

    name = "longbench"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = _load_longbench()
        if ds is None:
            logger.warning("Could not load LongBench dataset.")
            return []

        ds = limit_dataset(ds, config.max_examples, shuffle_seed=42)

        envs = []
        for row in ds:
            row = dict(row)
            context = row.get("context", "")
            subtask = row.get("domain", row.get("dataset", "unknown"))

            question = row.get("question", row.get("input", ""))
            choices = []
            for letter in ("A", "B", "C", "D"):
                choice = row.get(f"choice_{letter}", "")
                if choice:
                    choices.append(choice)

            expected = str(row.get("answer", "")).strip().upper()

            if choices:
                user_content = (
                    f"Read the following text carefully, then answer the question.\n\n"
                    f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                    f"Question: {question}\n\n{format_mcq_choices(choices)}\n\n"
                    "Answer with just the letter (A, B, C, or D)."
                )
                is_mcq = expected in ("A", "B", "C", "D")
            else:
                user_content = (
                    f"Read the following text carefully, then answer the question.\n\n"
                    f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                    f"Question: {question}\n\nGive a concise answer."
                )
                is_mcq = False

            if not question or not context:
                continue

            example_id = make_example_id("longbench", question)
            envs.append(
                LongBenchEnv(
                    user_content, expected, subtask, is_mcq, renderer, example_id=example_id
                )
            )
        return envs

    def aggregate(self, rewards: list[float], metrics_list: list[dict]) -> BenchmarkResult:
        """Aggregate with per-subtask breakdown."""
        num_correct = sum(1 for r in rewards if r > 0)
        accuracy = num_correct / len(rewards) if rewards else 0.0

        subtask_results: dict[str, list[bool]] = {}
        for r, m in zip(rewards, metrics_list):
            st = m.get("subtask", "unknown")
            if isinstance(st, str):
                subtask_results.setdefault(st, []).append(r > 0)

        metrics: dict[str, float] = {"longbench/accuracy": accuracy}
        for st, st_res in sorted(subtask_results.items()):
            metrics[f"longbench/{st}/accuracy"] = sum(st_res) / len(st_res) if st_res else 0.0

        return BenchmarkResult(
            name=self.name,
            score=accuracy,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics=metrics,
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(LongBenchBenchmarkBuilder())
