"""MMLU-Redux benchmark -- cleaner subset of MMLU with verified annotations.

Dataset: ``edinburgh-dawg/mmlu-redux`` on HuggingFace.
Metric: Multiple-choice accuracy (A/B/C/D).
Pattern: Single-turn ``MessageEnv`` + programmatic grading.

MMLU-Redux re-annotates 3,000 MMLU examples (30 subjects x 100) and flags
questions with errors. We evaluate only on samples with ``error_type == "ok"``.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import cast

from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    build_messages,
    extract_mcq_answer,
    format_mcq_choices,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
)
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env

logger = logging.getLogger(__name__)

# All 30 MMLU-Redux subjects
_SUBJECTS = [
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "formal_logic",
    "global_facts",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "high_school_us_history",
    "human_aging",
    "logical_fallacies",
    "machine_learning",
    "miscellaneous",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "public_relations",
    "virology",
]


def _load_mmlu_redux(max_examples: int | None) -> list[dict]:
    """Load all MMLU-Redux subjects, filtering to error_type=='ok'."""
    all_rows: list[dict] = []
    for subject in _SUBJECTS:
        try:
            ds = cast(Dataset, load_benchmark_dataset("edinburgh-dawg/mmlu-redux", name=subject))
            for row in ds:
                row_dict = dict(row)
                row_dict["_subject"] = subject
                if row_dict.get("error_type", "ok") == "ok":
                    all_rows.append(row_dict)
        except Exception as e:
            logger.warning(f"Could not load MMLU-Redux/{subject}: {e}")
            continue

    if max_examples and len(all_rows) > max_examples:
        rng = random.Random(42)
        rng.shuffle(all_rows)
        all_rows = all_rows[:max_examples]

    return all_rows


# ---------------------------------------------------------------------------
# MessageEnv implementation
# ---------------------------------------------------------------------------


class MMLUReduxMessageEnv(MessageEnv):
    """Single-turn message env for one MMLU-Redux question.

    Receives a parsed ``Message`` (thinking already stripped by
    ``EnvFromMessageEnv``), grades the answer, and returns the result.
    """

    def __init__(
        self,
        prompt: str,
        expected: str,
        subject: str,
        example_id: str = "",
        system_prompt: str | None = None,
    ):
        self.prompt = prompt
        self.expected = expected
        self.subject = subject
        self.example_id = example_id
        self.system_prompt = system_prompt

    async def initial_observation(self) -> list[Message]:
        return build_messages(self.prompt, self.system_prompt)

    async def step(self, message: Message) -> MessageStepResult:
        response = get_text_content(message)
        extracted = extract_mcq_answer(response)
        correct = extracted == self.expected
        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float(correct)},
            logs={
                "example_id": self.example_id,
                "input": self.prompt[:200],
                "expected": self.expected,
                "extracted": extracted,
                "subject": self.subject,
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class MMLUReduxBenchmarkBuilder(BenchmarkBuilder):
    """MMLU-Redux: verified subset of MMLU (30 subjects, 4-choice MCQ)."""

    name = "mmlu_redux"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        rows = _load_mmlu_redux(config.max_examples)
        if not rows:
            logger.warning("Could not load MMLU-Redux dataset.")
            return []

        envs: list[Env] = []
        for row in rows:
            question = row.get("question", "")
            choices = row.get("choices", [])
            answer_idx = row.get("answer", 0)

            if not question or not choices:
                continue

            expected = chr(65 + int(answer_idx))
            prompt = f"{question}\n\n{format_mcq_choices(choices)}\n\nAnswer with just the letter (A, B, C, or D)."
            subject = row.get("_subject", "unknown")
            example_id = make_example_id("mmlu_redux", question)
            msg_env = MMLUReduxMessageEnv(
                prompt,
                expected,
                subject,
                example_id=example_id,
                system_prompt=config.system_prompt,
            )
            envs.append(
                EnvFromMessageEnv(
                    renderer=renderer,
                    message_env=msg_env,
                    failed_parse_reward=0.0,
                    context_overflow_reward=0.0,
                )
            )
        return envs

    def aggregate(self, rewards: list[float], metrics_list: list[dict]) -> BenchmarkResult:
        """Aggregate with per-subject breakdown."""
        num_correct = sum(1 for r in rewards if r > 0)
        accuracy = num_correct / len(rewards) if rewards else 0.0

        subject_results: dict[str, list[bool]] = {}
        for r, m in zip(rewards, metrics_list):
            subj = m.get("subject", "unknown")
            if isinstance(subj, str):
                subject_results.setdefault(subj, []).append(r > 0)

        metrics: dict[str, float] = {"mmlu_redux/accuracy": accuracy}
        for subj, subj_res in sorted(subject_results.items()):
            metrics[f"mmlu_redux/{subj}/accuracy"] = (
                sum(subj_res) / len(subj_res) if subj_res else 0.0
            )

        return BenchmarkResult(
            name=self.name,
            score=accuracy,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics=metrics,
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(MMLUReduxBenchmarkBuilder())
