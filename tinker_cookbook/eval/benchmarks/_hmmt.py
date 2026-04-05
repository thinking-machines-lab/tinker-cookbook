"""HMMT benchmarks -- Harvard-MIT Math Tournament competition problems.

Provides separate benchmarks for each competition:
- ``hmmt_feb_2025``: HMMT February 2025 (30 problems)
- ``hmmt_nov_2025``: HMMT November 2025 (30 problems)

Datasets: ``MathArena/hmmt_feb_2025`` and ``MathArena/hmmt_nov_2025`` on HuggingFace.
Metric: Accuracy -- fraction of problems where extracted answer matches ground truth.
Pattern: Single-turn ``MessageEnv`` + programmatic grading.

Note: HMMT answers can be arbitrary LaTeX expressions (fractions, radicals, etc.),
not just integers like AIME. Grading uses normalized string comparison, which may
miss equivalent expressions (e.g., ``1/2`` vs ``\\frac{1}{2}``). For precise
evaluation, consider using a custom ``grade_fn`` with symbolic math comparison.
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


# Pinned dataset source per competition
_HMMT_DATASETS: dict[str, str] = {
    "feb_2025": "MathArena/hmmt_feb_2025",
    "nov_2025": "MathArena/hmmt_nov_2025",
}


def _normalize_latex(s: str) -> str:
    """Normalize a LaTeX answer string for comparison.

    Strips whitespace, removes ``$``, ``\\text{}``, ``\\mathrm{}``, and normalizes
    common LaTeX patterns.
    """
    s = s.strip()
    # Remove dollar signs
    s = s.replace("$", "")
    # Remove \text{...} and \mathrm{...} wrappers
    s = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", s)
    # Remove \left and \right
    s = s.replace("\\left", "").replace("\\right", "")
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Remove trailing period
    s = s.rstrip(".")
    return s


def _check_math_equal(extracted: str, expected: str) -> bool:
    """Check if two math expressions are equal using sympy.

    Tries multiple strategies:
    1. Normalized string comparison
    2. Numeric float comparison
    3. Sympy symbolic comparison via parse_latex

    Following MathArena's approach (eth-sri/matharena).
    """
    # Strategy 1: normalized string match
    ext_norm = _normalize_latex(extracted)
    exp_norm = _normalize_latex(expected)
    if ext_norm == exp_norm:
        return True

    if not ext_norm or not exp_norm:
        return False

    # Strategy 2: numeric float comparison
    try:
        if abs(float(ext_norm) - float(exp_norm)) < 1e-6:
            return True
    except ValueError:
        pass

    # Strategy 3: sympy symbolic comparison
    # Requires antlr4-python3-runtime for parse_latex; graceful fallback if missing.
    try:
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex

        parsed_ext = parse_latex(ext_norm)
        parsed_exp = parse_latex(exp_norm)
        if parsed_ext is None or parsed_exp is None:
            return False

        # Try .equals() first (exact symbolic)
        if parsed_ext.equals(parsed_exp):
            return True

        # Try simplify(a - b) == 0
        diff = simplify(parsed_ext - parsed_exp)
        if diff == 0:
            return True

        # Try numeric evaluation
        num_diff = abs(complex(N(diff)))
        if num_diff < 1e-8:
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# MessageEnv implementation
# ---------------------------------------------------------------------------


class HMMTMessageEnv(MessageEnv):
    """Single-turn message env for one HMMT problem.

    Grading uses normalized string comparison of the LaTeX answer.
    """

    def __init__(
        self,
        problem: str,
        expected: str,
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
            "This is an HMMT competition problem. "
            "Show your work step by step, then put your final answer in \\boxed{}."
        )
        return build_messages(prompt, self.system_prompt)

    async def step(self, message: Message) -> MessageStepResult:
        response = get_text_content(message)
        boxed = extract_boxed(response)
        extracted = boxed if boxed else ""

        correct = _check_math_equal(extracted, self.expected) if extracted else False

        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float(correct)},
            logs={
                "example_id": self.example_id,
                "input": self.problem[:200],
                "expected": self.expected,
                "extracted": extracted,
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builders
# ---------------------------------------------------------------------------


class _HMMTBenchmarkBuilder(BenchmarkBuilder):
    """Base builder for HMMT benchmarks — parameterized by competition."""

    experimental = True

    def __init__(self, competition: str, benchmark_name: str):
        self._competition = competition
        self.name = benchmark_name

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        dataset_id = _HMMT_DATASETS.get(self._competition)
        if dataset_id is None:
            logger.warning(f"No known dataset for HMMT {self._competition}")
            return []

        ds: Dataset | None = None
        for split in ("test", "train"):
            try:
                ds = cast(Dataset, load_benchmark_dataset(dataset_id, split=split))
                logger.info(
                    f"Loaded HMMT {self._competition} from {dataset_id}/{split} "
                    f"({len(ds)} problems)"
                )
                break
            except Exception as e:
                logger.debug(f"Failed to load {dataset_id}/{split}: {e}")
                continue

        if ds is None:
            logger.warning(f"Could not load HMMT {self._competition} dataset")
            return []

        ds = limit_dataset(ds, config.max_examples)

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            problem = row.get("problem", row.get("question", ""))
            expected = str(row.get("answer", "")).strip()
            if not problem or not expected:
                continue

            example_id = make_example_id(self.name, problem)
            msg_env = HMMTMessageEnv(
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
                    context_overflow_reward=0.0,
                )
            )
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(_HMMTBenchmarkBuilder("feb_2025", "hmmt_feb_2025"))
register(_HMMTBenchmarkBuilder("nov_2025", "hmmt_nov_2025"))
