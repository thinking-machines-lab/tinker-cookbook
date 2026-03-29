"""Base types for the Nemotron-Cascade-2 evaluation system."""

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Result of running a single benchmark evaluation.

    Attributes:
        benchmark: Name of the benchmark (e.g. "gsm8k", "ifeval").
        score: Primary metric normalized to 0-1 range.
        num_examples: Total number of examples evaluated (excluding errors).
        num_correct: Number of examples graded as correct.
        metrics: Additional metrics specific to the benchmark
            (e.g. per-subtask breakdowns, loose vs strict accuracy).
        examples: Optional list of per-example dicts for analysis. Each dict
            should include at minimum {"input": ..., "output": ..., "correct": bool}.
    """

    benchmark: str
    score: float  # primary metric (0-1)
    num_examples: int
    num_correct: int
    metrics: dict = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)
