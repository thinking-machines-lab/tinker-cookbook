"""C-Eval benchmark -- Chinese examination multiple-choice QA.

Dataset: ``ceval/ceval-exam`` on HuggingFace (52 subjects, 4-option MCQA).
Metric: Multiple-choice accuracy (A/B/C/D).
Pattern: Single-turn ``MessageEnv`` + programmatic grading.

Official reference: https://cevalbenchmark.com/
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

from datasets import Dataset, concatenate_datasets

from tinker_cookbook.eval.benchmarks._common import (
    build_messages,
    extract_mcq_answer,
    format_mcq_choices,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    Metrics,
)
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MessageEnv implementation
# ---------------------------------------------------------------------------


class CEvalMessageEnv(MessageEnv):
    """Single-turn message env for one C-Eval question."""

    def __init__(
        self,
        prompt: str,
        expected: str,
        subject: str = "",
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
            metrics={"correct": float(correct), "subject": self.subject},
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

# C-Eval has 52 subjects; we load all of them and concatenate.
_CEVAL_SUBJECTS = [
    "computer_network", "operating_system", "computer_architecture",
    "college_programming", "college_physics", "college_chemistry",
    "advanced_mathematics", "probability_and_statistics", "discrete_mathematics",
    "electrical_engineer", "metrology_engineer", "high_school_mathematics",
    "high_school_physics", "high_school_chemistry", "high_school_biology",
    "middle_school_mathematics", "middle_school_biology", "middle_school_physics",
    "middle_school_chemistry", "veterinary_medicine", "college_economics",
    "business_administration", "marxism", "mao_zedong_thought",
    "education_science", "teacher_qualification", "high_school_politics",
    "high_school_geography", "middle_school_politics", "middle_school_geography",
    "modern_chinese_history", "ideological_and_moral_cultivation",
    "logic", "law", "chinese_language_and_literature", "art_studies",
    "professional_tour_guide", "legal_professional", "high_school_chinese",
    "high_school_history", "middle_school_history", "civil_servant",
    "sports_science", "plant_protection", "basic_medicine", "clinical_medicine",
    "urban_and_rural_planner", "accountant", "fire_engineer",
    "environmental_impact_assessment_engineer", "tax_accountant",
    "physician",
]


class CEvalBenchmarkBuilder(BenchmarkBuilder):
    """C-Eval: comprehensive Chinese exam benchmark (52 subjects, ~12K questions)."""

    name = "ceval"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        # Load and concatenate all subjects' val splits
        datasets: list[Dataset] = []
        for subject in _CEVAL_SUBJECTS:
            try:
                ds = cast(
                    Dataset,
                    load_benchmark_dataset("ceval/ceval-exam", name=subject, split="val"),
                )
                # Add subject column for per-subject breakdown
                ds = ds.map(lambda x: {"_subject": subject})
                datasets.append(ds)
            except Exception as e:
                logger.debug(f"Could not load C-Eval subject {subject}: {e}")
                continue

        if not datasets:
            logger.warning("Could not load any C-Eval subjects")
            return []

        combined = concatenate_datasets(datasets)
        logger.info(f"Loaded C-Eval: {len(combined)} questions across {len(datasets)} subjects")
        combined = limit_dataset(combined, config.max_examples, shuffle_seed=42)

        envs: list[Env] = []
        for row in combined:
            row = dict(row)
            question = row.get("question", "")
            answer = row.get("answer", "")
            if not question or not answer:
                continue

            choices = [row.get("A", ""), row.get("B", ""), row.get("C", ""), row.get("D", "")]
            choices = [c for c in choices if c]
            if len(choices) < 2:
                continue

            subject = row.get("_subject", "")
            prompt = (
                f"{question}\n\n{format_mcq_choices(choices)}\n\n"
                "Think step by step, then give your final answer as a single letter (A, B, C, or D)."
            )
            example_id = make_example_id("ceval", question)
            msg_env = CEvalMessageEnv(
                prompt,
                answer.strip().upper(),
                subject=subject,
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

    def aggregate(
        self,
        rewards: list[float],
        metrics_list: list[Metrics],
    ) -> BenchmarkResult:
        """Aggregate with per-subject breakdown."""

        num_correct = sum(1 for r in rewards if r > 0)
        accuracy = num_correct / len(rewards) if rewards else 0.0

        # Per-subject accuracy
        subject_scores: dict[str, list[float]] = {}
        for r, m in zip(rewards, metrics_list):
            subj = m.get("subject", "unknown")
            if isinstance(subj, str):
                subject_scores.setdefault(subj, []).append(r)

        metrics: Metrics = {"ceval/accuracy": accuracy}
        for subj, scores in sorted(subject_scores.items()):
            if scores:
                metrics[f"ceval/{subj}/accuracy"] = sum(1 for s in scores if s > 0) / len(scores)

        return BenchmarkResult(
            name=self.name,
            score=accuracy,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics=metrics,
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(CEvalBenchmarkBuilder())
