"""Typed, offline comparison of two :class:`EvalStore` runs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tinker_cookbook.stores.storage import storage_join

if TYPE_CHECKING:
    from tinker_cookbook.eval.benchmarks._types import StoredTrajectory
    from tinker_cookbook.stores.eval_store import EvalStore


ExampleClassification = Literal[
    "regression",
    "improvement",
    "unchanged_correct",
    "unchanged_incorrect",
    "persistent_error",
    "new_error",
    "resolved_error",
    "baseline_only",
    "candidate_only",
]
"""Classification assigned to an example while comparing two evaluation runs."""


@dataclass
class ExampleComparison:
    """Comparison of one matched or unmatched evaluation example."""

    benchmark: str
    example_id: str
    classification: ExampleClassification
    baseline_reward: float | None
    candidate_reward: float | None
    baseline_error: str | None
    candidate_error: str | None
    baseline_logs: dict[str, Any] | None
    candidate_logs: dict[str, Any] | None
    baseline_trajectory_index: int | None
    candidate_trajectory_index: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)


@dataclass
class BenchmarkComparison:
    """Score and per-example changes for one benchmark shared by two runs."""

    name: str
    baseline_score: float
    candidate_score: float
    score_delta: float
    baseline_num_examples: int
    candidate_num_examples: int
    baseline_trajectories_available: bool
    candidate_trajectories_available: bool
    num_regressions: int
    num_improvements: int
    num_unchanged: int
    num_unchanged_correct: int
    num_unchanged_incorrect: int
    num_persistent_errors: int
    num_new_errors: int
    num_resolved_errors: int
    num_baseline_only: int
    num_candidate_only: int
    examples: list[ExampleComparison] = field(default_factory=list)

    @property
    def num_unmatched(self) -> int:
        return self.num_baseline_only + self.num_candidate_only

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        data = asdict(self)
        data["num_unmatched"] = self.num_unmatched
        return data


@dataclass
class RunComparison:
    """Offline comparison of two stored evaluation runs."""

    baseline_run_id: str
    candidate_run_id: str
    baseline_model_name: str
    candidate_model_name: str
    baseline_checkpoint_path: str | None
    candidate_checkpoint_path: str | None
    baseline_checkpoint_name: str | None
    candidate_checkpoint_name: str | None
    shared_benchmarks: list[str]
    baseline_only_benchmarks: list[str]
    candidate_only_benchmarks: list[str]
    benchmarks: dict[str, BenchmarkComparison]
    num_regressions: int
    num_improvements: int
    num_unchanged: int
    num_persistent_errors: int
    num_new_errors: int
    num_resolved_errors: int
    num_baseline_only: int
    num_candidate_only: int

    @property
    def num_unmatched(self) -> int:
        return self.num_baseline_only + self.num_candidate_only

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        data = asdict(self)
        data["num_unmatched"] = self.num_unmatched
        for name, benchmark in self.benchmarks.items():
            data["benchmarks"][name]["num_unmatched"] = benchmark.num_unmatched
        return data

    def to_markdown(self, max_examples_per_section: int = 20) -> str:
        """Render a concise Markdown regression report."""
        if max_examples_per_section < 0:
            raise ValueError("max_examples_per_section must be non-negative")

        lines = [
            "# Evaluation Comparison",
            "",
            _run_line(
                "Baseline",
                self.baseline_run_id,
                self.baseline_model_name,
                self.baseline_checkpoint_name or self.baseline_checkpoint_path,
            ),
            _run_line(
                "Candidate",
                self.candidate_run_id,
                self.candidate_model_name,
                self.candidate_checkpoint_name or self.candidate_checkpoint_path,
            ),
            "",
            (
                f"Regressions: **{self.num_regressions}** · "
                f"Improvements: **{self.num_improvements}** · "
                f"Unchanged: **{self.num_unchanged}** · "
                f"Errors: **{self.num_new_errors} new, "
                f"{self.num_resolved_errors} resolved, "
                f"{self.num_persistent_errors} persistent** · "
                f"Unmatched: **{self.num_unmatched}**"
            ),
            "",
            "## Benchmark summary",
            "",
        ]
        if self.shared_benchmarks:
            lines.extend(
                [
                    (
                        "| Benchmark | Baseline | Candidate | Delta | "
                        "Regressions | Improvements | Errors | Unmatched |"
                    ),
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for name in self.shared_benchmarks:
                benchmark = self.benchmarks[name]
                num_errors = (
                    benchmark.num_new_errors
                    + benchmark.num_resolved_errors
                    + benchmark.num_persistent_errors
                )
                lines.append(
                    f"| {name.replace('|', '/')} | {benchmark.baseline_score:.4f} "
                    f"| {benchmark.candidate_score:.4f} | {benchmark.score_delta:+.4f} "
                    f"| {benchmark.num_regressions} | {benchmark.num_improvements} "
                    f"| {num_errors} | {benchmark.num_unmatched} |"
                )
        else:
            lines.append("_No shared benchmarks._")

        missing = []
        for name in self.shared_benchmarks:
            benchmark = self.benchmarks[name]
            sides = []
            if not benchmark.baseline_trajectories_available:
                sides.append("baseline")
            if not benchmark.candidate_trajectories_available:
                sides.append("candidate")
            if sides:
                missing.append(f"`{_inline(name)}` ({' and '.join(sides)})")
        if missing:
            lines.extend(["", "**Trajectory data unavailable:** " + "; ".join(missing)])
        for label, names in (
            ("Baseline-only benchmarks", self.baseline_only_benchmarks),
            ("Candidate-only benchmarks", self.candidate_only_benchmarks),
        ):
            if names:
                lines.extend(["", f"{label}: " + ", ".join(f"`{_inline(n)}`" for n in names)])

        sections: list[tuple[str, tuple[ExampleClassification, ...]]] = [
            ("Regressions", ("regression",)),
            ("Improvements", ("improvement",)),
            ("Persistent failures", ("unchanged_incorrect",)),
            ("Errors", ("new_error", "resolved_error", "persistent_error")),
            ("Unmatched examples", ("baseline_only", "candidate_only")),
        ]
        all_examples = [
            example for name in self.shared_benchmarks for example in self.benchmarks[name].examples
        ]
        for title, classifications in sections:
            examples = [
                example for example in all_examples if example.classification in classifications
            ]
            if not examples:
                continue
            shown = examples[:max_examples_per_section]
            lines.extend(["", f"## {title}", ""])
            if len(shown) < len(examples):
                lines.extend([f"_Showing {len(shown)} of {len(examples)} examples._", ""])
            for example in shown:
                lines.extend(_example_lines(example))
        return "\n".join(lines).rstrip() + "\n"


def _inline(value: str) -> str:
    return value.replace("`", "'").replace("\n", " ")


def _run_line(label: str, run_id: str, model: str, checkpoint: str | None) -> str:
    return (
        f"**{label}:** `{_inline(run_id)}` "
        f"(model: `{_inline(model)}`, checkpoint: `{_inline(checkpoint or 'base')}`)"
    )


def _example_lines(example: ExampleComparison) -> list[str]:
    lines = [
        f"- **{example.benchmark}** / `{_inline(example.example_id or '<missing example_id>')}`",
        f"  - Status: `{example.classification}`",
    ]
    for label, value in (
        ("Baseline reward", example.baseline_reward),
        ("Candidate reward", example.candidate_reward),
        ("Baseline error", example.baseline_error),
        ("Candidate error", example.candidate_error),
    ):
        if value is not None:
            lines.append(f"  - {label}: {value}")
    for label, logs in (
        ("Baseline logs", example.baseline_logs),
        ("Candidate logs", example.candidate_logs),
    ):
        if logs:
            value = json.dumps(logs, ensure_ascii=False, sort_keys=True, default=str)
            lines.append(f"  - {label}: `{_inline(value[:300])}`")
    return lines


def _classify(
    baseline: StoredTrajectory,
    candidate: StoredTrajectory,
) -> ExampleClassification:
    if baseline.error is not None and candidate.error is not None:
        return "persistent_error"
    if candidate.error is not None:
        return "new_error"
    if baseline.error is not None:
        return "resolved_error"
    if baseline.reward > 0 and candidate.reward <= 0:
        return "regression"
    if baseline.reward <= 0 and candidate.reward > 0:
        return "improvement"
    return "unchanged_correct" if baseline.reward > 0 else "unchanged_incorrect"


def _comparison(
    benchmark: str,
    baseline: StoredTrajectory | None,
    candidate: StoredTrajectory | None,
) -> ExampleComparison:
    trajectory = baseline or candidate
    if trajectory is None:
        raise ValueError("An example comparison requires at least one trajectory")
    if baseline is None:
        classification: ExampleClassification = "candidate_only"
    elif candidate is None:
        classification = "baseline_only"
    else:
        classification = _classify(baseline, candidate)
    return ExampleComparison(
        benchmark=benchmark,
        example_id=trajectory.example_id,
        classification=classification,
        baseline_reward=baseline.reward if baseline is not None else None,
        candidate_reward=candidate.reward if candidate is not None else None,
        baseline_error=baseline.error if baseline is not None else None,
        candidate_error=candidate.error if candidate is not None else None,
        baseline_logs=baseline.logs if baseline is not None else None,
        candidate_logs=candidate.logs if candidate is not None else None,
        baseline_trajectory_index=baseline.idx if baseline is not None else None,
        candidate_trajectory_index=candidate.idx if candidate is not None else None,
    )


def _index(
    trajectories: list[StoredTrajectory],
    *,
    run_id: str,
    benchmark: str,
) -> tuple[dict[str, StoredTrajectory], list[StoredTrajectory]]:
    by_id: dict[str, StoredTrajectory] = {}
    missing_id: list[StoredTrajectory] = []
    for trajectory in trajectories:
        if not trajectory.example_id:
            missing_id.append(trajectory)
        elif trajectory.example_id in by_id:
            raise ValueError(
                f"Duplicate example_id {trajectory.example_id!r} in benchmark "
                f"{benchmark!r} for run {run_id!r}"
            )
        else:
            by_id[trajectory.example_id] = trajectory
    return by_id, sorted(missing_id, key=lambda trajectory: trajectory.idx)


def _compare_examples(
    baseline: list[StoredTrajectory],
    candidate: list[StoredTrajectory],
    *,
    baseline_run_id: str,
    candidate_run_id: str,
    benchmark: str,
) -> list[ExampleComparison]:
    baseline_by_id, baseline_missing_id = _index(
        baseline, run_id=baseline_run_id, benchmark=benchmark
    )
    candidate_by_id, candidate_missing_id = _index(
        candidate, run_id=candidate_run_id, benchmark=benchmark
    )
    examples = [
        _comparison(benchmark, baseline_by_id.get(example_id), candidate_by_id.get(example_id))
        for example_id in sorted(baseline_by_id.keys() | candidate_by_id.keys())
    ]
    examples.extend(_comparison(benchmark, trajectory, None) for trajectory in baseline_missing_id)
    examples.extend(_comparison(benchmark, None, trajectory) for trajectory in candidate_missing_id)
    return examples


def compare_eval_runs(
    store: EvalStore,
    baseline_run_id: str,
    candidate_run_id: str,
) -> RunComparison:
    """Build an offline comparison from two runs in ``store``."""
    baseline_metadata = store.read_run(baseline_run_id)
    candidate_metadata = store.read_run(candidate_run_id)
    baseline_benchmarks = set(store.list_benchmarks(baseline_run_id))
    candidate_benchmarks = set(store.list_benchmarks(candidate_run_id))
    shared = sorted(baseline_benchmarks & candidate_benchmarks)
    benchmarks: dict[str, BenchmarkComparison] = {}

    for name in shared:
        baseline_result = store.read_result(baseline_run_id, name)
        candidate_result = store.read_result(candidate_run_id, name)
        if baseline_result is None or candidate_result is None:
            run_id = baseline_run_id if baseline_result is None else candidate_run_id
            raise ValueError(f"Benchmark result {name!r} for run {run_id!r} could not be read")
        baseline_path = storage_join(
            store.prefix, "runs", baseline_run_id, name, "trajectories.jsonl"
        )
        candidate_path = storage_join(
            store.prefix, "runs", candidate_run_id, name, "trajectories.jsonl"
        )
        examples = _compare_examples(
            store.read_trajectories(baseline_run_id, name),
            store.read_trajectories(candidate_run_id, name),
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            benchmark=name,
        )
        counts = Counter(example.classification for example in examples)
        benchmarks[name] = BenchmarkComparison(
            name=name,
            baseline_score=baseline_result.score,
            candidate_score=candidate_result.score,
            score_delta=candidate_result.score - baseline_result.score,
            baseline_num_examples=baseline_result.num_examples,
            candidate_num_examples=candidate_result.num_examples,
            baseline_trajectories_available=store.storage.exists(baseline_path),
            candidate_trajectories_available=store.storage.exists(candidate_path),
            num_regressions=counts["regression"],
            num_improvements=counts["improvement"],
            num_unchanged=counts["unchanged_correct"] + counts["unchanged_incorrect"],
            num_unchanged_correct=counts["unchanged_correct"],
            num_unchanged_incorrect=counts["unchanged_incorrect"],
            num_persistent_errors=counts["persistent_error"],
            num_new_errors=counts["new_error"],
            num_resolved_errors=counts["resolved_error"],
            num_baseline_only=counts["baseline_only"],
            num_candidate_only=counts["candidate_only"],
            examples=examples,
        )

    return RunComparison(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        baseline_model_name=baseline_metadata.model_name,
        candidate_model_name=candidate_metadata.model_name,
        baseline_checkpoint_path=baseline_metadata.checkpoint_path,
        candidate_checkpoint_path=candidate_metadata.checkpoint_path,
        baseline_checkpoint_name=baseline_metadata.checkpoint_name,
        candidate_checkpoint_name=candidate_metadata.checkpoint_name,
        shared_benchmarks=shared,
        baseline_only_benchmarks=sorted(baseline_benchmarks - candidate_benchmarks),
        candidate_only_benchmarks=sorted(candidate_benchmarks - baseline_benchmarks),
        benchmarks=benchmarks,
        num_regressions=sum(value.num_regressions for value in benchmarks.values()),
        num_improvements=sum(value.num_improvements for value in benchmarks.values()),
        num_unchanged=sum(value.num_unchanged for value in benchmarks.values()),
        num_persistent_errors=sum(value.num_persistent_errors for value in benchmarks.values()),
        num_new_errors=sum(value.num_new_errors for value in benchmarks.values()),
        num_resolved_errors=sum(value.num_resolved_errors for value in benchmarks.values()),
        num_baseline_only=sum(value.num_baseline_only for value in benchmarks.values()),
        num_candidate_only=sum(value.num_candidate_only for value in benchmarks.values()),
    )
