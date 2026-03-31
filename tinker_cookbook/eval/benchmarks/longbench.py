"""LongBench v2 benchmark -- long-context comprehension across multiple subtasks.

Dataset: ``THUDM/LongBench-v2`` (or v1 fallback) on HuggingFace.
Metric: Accuracy -- fraction of questions answered correctly.
Pattern: Single-turn generate + programmatic grading.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import load_benchmark_dataset
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig, BenchmarkResult
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)


def _load_longbench() -> tuple[Dataset | None, str]:
    """Try loading LongBench v2, falling back to v1. Returns (dataset, version)."""
    for dataset_id, split in (
        ("THUDM/LongBench-v2", "test"),
        ("THUDM/LongBench-v2", "train"),
        ("THUDM/LongBench", "test"),
    ):
        try:
            ds = cast(Dataset, load_benchmark_dataset(dataset_id, split=split))
            version = "v2" if "v2" in dataset_id else "v1"
            logger.info(f"Loaded LongBench from {dataset_id}/{split} ({len(ds)} examples)")
            return ds, version
        except Exception:
            try:
                ds = cast(Dataset, load_benchmark_dataset(dataset_id, name="default", split=split))
                version = "v2" if "v2" in dataset_id else "v1"
                logger.info(f"Loaded LongBench from {dataset_id}/default/{split} ({len(ds)} examples)")
                return ds, version
            except Exception:
                continue
    return None, "v2"


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
        response = self.renderer.tokenizer.decode(action)
        if self.is_mcq:
            letters = re.findall(r"\b([A-D])\b", response[-300:])
            extracted = letters[-1] if letters else ""
            correct = extracted == self.expected
        else:
            extracted = response.strip()[:200]
            pattern = r'\b' + re.escape(str(self.expected).strip()) + r'\b'
            correct = bool(re.search(pattern, response, re.IGNORECASE))
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct), "subtask": self.subtask},
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
        ds, dataset_version = _load_longbench()
        if ds is None:
            logger.warning("Could not load LongBench dataset.")
            return []

        if config.max_examples is not None:
            ds = ds.shuffle(seed=42).select(range(min(config.max_examples, len(ds))))

        envs = []
        for row in ds:
            context = row.get("context", "")
            subtask = row.get("domain", row.get("dataset", "unknown"))

            if dataset_version == "v2":
                question = row.get("question", row.get("input", ""))
                choices = []
                for letter in ("A", "B", "C", "D"):
                    choice = row.get(f"choice_{letter}", "")
                    if choice:
                        choices.append(choice)

                expected = str(row.get("answer", "")).strip().upper()

                if choices:
                    choice_text = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
                    user_content = (
                        f"Read the following text carefully, then answer the question.\n\n"
                        f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                        f"Question: {question}\n\n{choice_text}\n\n"
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
            else:
                question = row.get("input", "")
                expected_answers = row.get("answers", row.get("all_classes", []))
                expected = expected_answers[0] if expected_answers else ""
                user_content = (
                    f"Read the following text carefully, then answer the question.\n\n"
                    f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                    f"Question: {question}\n\nGive a concise answer."
                )
                is_mcq = False

            if not question or not context:
                continue

            example_id = f"longbench_{hashlib.md5(question.encode()).hexdigest()[:12]}"
            envs.append(LongBenchEnv(user_content, expected, subtask, is_mcq, renderer, example_id=example_id))
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
