"""Offline tests for EvalStore run comparison."""

import json
from pathlib import Path

import pytest

from tinker_cookbook.eval.benchmarks._types import BenchmarkResult, StoredTrajectory
from tinker_cookbook.stores.eval_store import EvalStore
from tinker_cookbook.stores.storage import LocalStorage


def _store(
    tmp_path: Path,
    baseline_benchmarks: tuple[str, ...] = ("bench",),
    candidate_benchmarks: tuple[str, ...] = ("bench",),
) -> EvalStore:
    store = EvalStore(LocalStorage(tmp_path), "eval")
    store.create_run(
        "baseline-model",
        list(baseline_benchmarks),
        checkpoint_path="tinker://baseline",
        checkpoint_name="step-100",
        run_id="baseline",
    )
    store.create_run(
        "candidate-model",
        list(candidate_benchmarks),
        checkpoint_path="tinker://candidate",
        checkpoint_name="step-200",
        run_id="candidate",
    )
    return store


def _trajectory(
    example_id: str,
    reward: float,
    idx: int,
    *,
    benchmark: str = "bench",
    error: str | None = None,
    logs: dict | None = None,
) -> StoredTrajectory:
    return StoredTrajectory(
        idx=idx,
        benchmark=benchmark,
        example_id=example_id,
        reward=reward,
        error=error,
        logs=logs or {},
    )


def _write(
    store: EvalStore,
    run_id: str,
    trajectories: list[StoredTrajectory],
    *,
    benchmark: str = "bench",
    score: float = 0.0,
    num_examples: int | None = None,
) -> None:
    store.write_result(
        run_id,
        BenchmarkResult(
            name=benchmark,
            score=score,
            num_examples=len(trajectories) if num_examples is None else num_examples,
            num_correct=sum(trajectory.reward > 0 for trajectory in trajectories),
            num_errors=sum(trajectory.error is not None for trajectory in trajectories),
        ),
    )
    for trajectory in trajectories:
        store.write_trajectory(run_id, benchmark, trajectory)


def test_scores_metadata_benchmark_sets_and_reordered_matching(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        baseline_benchmarks=("bench", "baseline_only"),
        candidate_benchmarks=("bench", "candidate_only"),
    )
    _write(
        store,
        "baseline",
        [_trajectory("a", 0.25, 10), _trajectory("b", 0.0, 20)],
        score=0.5,
        num_examples=8,
    )
    _write(
        store,
        "candidate",
        [_trajectory("b", 0.4, 200), _trajectory("a", 0.0, 100)],
        score=0.625,
        num_examples=12,
    )
    _write(store, "baseline", [], benchmark="baseline_only", score=0.5)
    _write(store, "candidate", [], benchmark="candidate_only", score=0.9)

    comparison = store.compare_runs("baseline", "candidate")
    benchmark = comparison.benchmarks["bench"]
    examples = {example.example_id: example for example in benchmark.examples}

    assert comparison.baseline_model_name == "baseline-model"
    assert comparison.candidate_checkpoint_name == "step-200"
    assert comparison.shared_benchmarks == ["bench"]
    assert comparison.baseline_only_benchmarks == ["baseline_only"]
    assert comparison.candidate_only_benchmarks == ["candidate_only"]
    assert benchmark.score_delta == pytest.approx(0.125)
    assert (benchmark.baseline_num_examples, benchmark.candidate_num_examples) == (8, 12)
    assert examples["a"].classification == "regression"
    assert examples["a"].baseline_trajectory_index == 10
    assert examples["a"].candidate_trajectory_index == 100
    assert examples["b"].classification == "improvement"


def test_all_matched_classifications_and_debug_data(tmp_path: Path) -> None:
    store = _store(tmp_path)
    baseline = [
        _trajectory("regression", 0.1, 0, logs={"expected": "yes"}),
        _trajectory("improvement", 0.0, 1),
        _trajectory("unchanged_correct", 0.2, 2),
        _trajectory("unchanged_incorrect", 0.0, 3),
        _trajectory("persistent_error", 0.0, 4, error="old timeout"),
        _trajectory("new_error", 0.3, 5),
        _trajectory("resolved_error", 0.0, 6, error="old crash"),
    ]
    candidate = [
        _trajectory("regression", 0.0, 10, logs={"output": "no"}),
        _trajectory("improvement", 0.2, 11),
        _trajectory("unchanged_correct", 0.4, 12),
        _trajectory("unchanged_incorrect", -0.5, 13),
        _trajectory("persistent_error", 0.0, 14, error="new timeout"),
        _trajectory("new_error", 0.0, 15, error="new crash"),
        _trajectory("resolved_error", 0.6, 16),
    ]
    _write(store, "baseline", baseline)
    _write(store, "candidate", candidate)

    comparison = store.compare_runs("baseline", "candidate")
    benchmark = comparison.benchmarks["bench"]
    examples = {example.example_id: example for example in benchmark.examples}

    assert {example_id: example.classification for example_id, example in examples.items()} == {
        name: name
        for name in (
            "regression",
            "improvement",
            "unchanged_correct",
            "unchanged_incorrect",
            "persistent_error",
            "new_error",
            "resolved_error",
        )
    }
    assert (benchmark.num_regressions, benchmark.num_improvements, benchmark.num_unchanged) == (
        1,
        1,
        2,
    )
    assert (
        benchmark.num_persistent_errors,
        benchmark.num_new_errors,
        benchmark.num_resolved_errors,
    ) == (1, 1, 1)
    assert examples["regression"].baseline_logs == {"expected": "yes"}
    assert examples["regression"].candidate_logs == {"output": "no"}
    assert examples["persistent_error"].baseline_error == "old timeout"
    assert examples["persistent_error"].candidate_error == "new timeout"


def test_unmatched_and_empty_ids_are_never_matched_by_index(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write(
        store,
        "baseline",
        [_trajectory("baseline-only", 1.0, 1), _trajectory("", 0.0, 7)],
    )
    _write(
        store,
        "candidate",
        [_trajectory("candidate-only", 1.0, 2), _trajectory("", 0.0, 7)],
    )

    comparison = store.compare_runs("baseline", "candidate")
    benchmark = comparison.benchmarks["bench"]

    assert (benchmark.num_baseline_only, benchmark.num_candidate_only) == (2, 2)
    assert comparison.num_unmatched == 4
    assert sum(example.example_id == "" for example in benchmark.examples) == 2


def test_no_shared_benchmarks_and_missing_runs(tmp_path: Path) -> None:
    store = _store(tmp_path, ("old",), ("new",))
    _write(store, "baseline", [], benchmark="old")
    _write(store, "candidate", [], benchmark="new")

    comparison = store.compare_runs("baseline", "candidate")
    assert comparison.shared_benchmarks == []
    assert comparison.baseline_only_benchmarks == ["old"]
    assert comparison.candidate_only_benchmarks == ["new"]
    assert "_No shared benchmarks._" in comparison.to_markdown()

    with pytest.raises(FileNotFoundError, match="missing"):
        store.compare_runs("missing", "candidate")
    with pytest.raises(FileNotFoundError, match="missing"):
        store.compare_runs("baseline", "missing")


def test_missing_empty_and_corrupted_trajectory_files(tmp_path: Path) -> None:
    store = _store(tmp_path, ("missing", "empty"), ("missing", "empty"))
    available = _trajectory("available", 1.0, 0, benchmark="missing")
    _write(store, "baseline", [available], benchmark="missing", score=1.0)
    _write(store, "candidate", [], benchmark="missing", num_examples=1)
    _write(store, "baseline", [], benchmark="empty", num_examples=1)
    _write(store, "candidate", [], benchmark="empty", num_examples=1)
    store.storage.write("eval/runs/baseline/empty/trajectories.jsonl", b"")
    store.storage.write("eval/runs/candidate/empty/trajectories.jsonl", b"")
    store.storage.append(
        "eval/runs/baseline/missing/trajectories.jsonl",
        b"{not-json}\n" + json.dumps({"benchmark": "missing"}).encode() + b"\n",
    )

    comparison = store.compare_runs("baseline", "candidate")
    missing = comparison.benchmarks["missing"]
    empty = comparison.benchmarks["empty"]

    assert missing.baseline_trajectories_available
    assert not missing.candidate_trajectories_available
    assert missing.num_baseline_only == 1
    assert empty.baseline_trajectories_available and empty.candidate_trajectories_available
    assert empty.examples == []
    assert "`missing` (candidate)" in comparison.to_markdown()


def test_duplicate_non_empty_id_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    duplicate = [_trajectory("duplicate", 1.0, 0), _trajectory("duplicate", 0.0, 1)]
    _write(store, "baseline", duplicate)
    _write(store, "candidate", [_trajectory("duplicate", 1.0, 0)])

    with pytest.raises(ValueError, match=r"Duplicate example_id 'duplicate'.*run 'baseline'"):
        store.compare_runs("baseline", "candidate")


def test_identical_run_ids(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write(store, "baseline", [_trajectory("same", 0.01, 0)], score=1.0)

    comparison = store.compare_runs("baseline", "baseline")

    assert comparison.num_unchanged == 1
    assert comparison.num_regressions == comparison.num_unmatched == 0


def test_markdown_bound_and_serialization(tmp_path: Path) -> None:
    store = _store(tmp_path)
    baseline = [
        _trajectory("regression-a", 1.0, 0, logs={"expected": "A"}),
        _trajectory("regression-b", 1.0, 1, logs={"expected": "B"}),
    ]
    candidate = [
        _trajectory("regression-a", 0.0, 10, logs={"output": "wrong A"}),
        _trajectory("regression-b", 0.0, 11, logs={"output": "wrong B"}),
    ]
    _write(store, "baseline", baseline, score=1.0)
    _write(store, "candidate", candidate, score=0.0)

    comparison = store.compare_runs("baseline", "candidate")
    serialized = comparison.to_dict()
    json.dumps(serialized)
    markdown = comparison.to_markdown(max_examples_per_section=1)

    assert serialized["benchmarks"]["bench"]["score_delta"] == -1.0
    assert serialized["benchmarks"]["bench"]["num_unmatched"] == 0
    assert "| bench | 1.0000 | 0.0000 | -1.0000 |" in markdown
    assert "_Showing 1 of 2 examples._" in markdown
    assert "`regression-a`" in markdown and "`regression-b`" not in markdown
    assert '"output": "wrong A"' in markdown
    with pytest.raises(ValueError, match="non-negative"):
        comparison.to_markdown(max_examples_per_section=-1)
