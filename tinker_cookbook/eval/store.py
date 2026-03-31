"""Persistent storage for evaluation runs — supports cross-checkpoint comparison.

Storage layout::

    eval_store/                         # Root directory
      runs.jsonl                        # Append-only index of all runs
      runs/
        {run_id}/                       # One directory per eval invocation
          metadata.json                 # Model, checkpoint, config, timestamp, scores
          {benchmark}/
            result.json                 # BenchmarkResult
            trajectories.jsonl          # Per-example StoredTrajectory with stable example_id

Key design decisions:
- File-based (no database dependency), easy to inspect and version-control
- Append-only runs.jsonl index for listing without directory scanning
- Stable example_id per trajectory (deterministic from dataset, not positional)
- Cross-run comparison by matching example_ids across runs

Usage::

    store = EvalStore("~/experiments/evals")

    # Create a run, evaluate, finalize
    run_id = store.create_run(
        model_name="nvidia/...",
        checkpoint_name="sft_step500",
        benchmarks=["gsm8k", "mmlu_pro"],
    )
    await run_benchmarks(
        ["gsm8k", "mmlu_pro"], sampling_client, renderer,
        BenchmarkConfig(save_dir=store.run_dir(run_id)),
    )
    store.finalize_run(run_id)

    # Query results — the store already has everything
    result = store.get_result(run_id, "gsm8k")
    wrong = store.get_trajectories(run_id, "gsm8k", incorrect_only=True)
    summary = store.get_summary(run_id)

    # Compare checkpoints
    comp = store.compare_runs("run_001", "run_002", "gsm8k")
    store.print_dashboard()
"""

from __future__ import annotations

import fcntl
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from tinker_cookbook.eval.benchmarks._types import BenchmarkResult, StoredTrajectory

logger = logging.getLogger(__name__)


@dataclass
class RunMetadata:
    """Metadata for a single evaluation run."""

    run_id: str
    model_name: str
    checkpoint_path: str | None
    checkpoint_name: str | None
    benchmarks: list[str]
    config: dict
    timestamp: str
    scores: dict[str, float] = field(default_factory=dict)
    """Populated after run completes: benchmark_name -> score."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunMetadata:
        return cls(**d)


@dataclass
class RunComparison:
    """Result of comparing two runs on a specific benchmark."""

    benchmark: str
    run_a_id: str
    run_b_id: str
    score_a: float
    score_b: float
    score_delta: float
    regressions: dict[str, tuple[StoredTrajectory, StoredTrajectory]]
    """Examples where run_a was correct but run_b was wrong (by example_id)."""
    improvements: dict[str, tuple[StoredTrajectory, StoredTrajectory]]
    """Examples where run_a was wrong but run_b was correct (by example_id)."""
    num_shared: int
    """Number of examples with matching example_ids in both runs."""


class EvalStore:
    """Persistent, file-based storage for evaluation runs.

    Supports creating runs, saving results/trajectories, listing past runs,
    and comparing runs side-by-side.

    Args:
        root_dir: Root directory for the eval store. Created if it doesn't exist.
    """

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "runs.jsonl"
        self._runs_dir = self.root / "runs"
        self._runs_dir.mkdir(exist_ok=True)

    def create_run(
        self,
        model_name: str,
        benchmarks: list[str],
        checkpoint_path: str | None = None,
        checkpoint_name: str | None = None,
        config: dict | None = None,
        run_id: str | None = None,
    ) -> str:
        """Create a new evaluation run and return its run_id.

        The run_id is used as the directory name under ``runs/`` and as the
        key for all subsequent operations. If not provided, a timestamp-based
        ID is generated.

        Args:
            model_name: Model being evaluated.
            benchmarks: List of benchmark names to run.
            checkpoint_path: Tinker checkpoint path (for reproduction).
            checkpoint_name: Human-readable name (e.g. ``"sft_step500"``).
            config: BenchmarkConfig as a dict (for reproduction).
            run_id: Optional custom run ID. Default: timestamp-based.

        Returns:
            The run_id string.
        """
        if run_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_label = checkpoint_name or "base"
            short_id = uuid.uuid4().hex[:6]
            run_id = f"{ckpt_label}_{ts}_{short_id}"

        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metadata = RunMetadata(
            run_id=run_id,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            benchmarks=benchmarks,
            config=config or {},
            timestamp=datetime.now().isoformat(),
        )

        # Save metadata
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Append to index (locked to prevent interleaving from concurrent writers)
        with open(self._index_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps({"run_id": run_id, "timestamp": metadata.timestamp,
                                "model": model_name, "checkpoint": checkpoint_name}) + "\n")

        logger.info(f"Created eval run: {run_id} at {run_dir}")
        return run_id

    def run_dir(self, run_id: str) -> str:
        """Return the directory path for a run (use as save_dir in BenchmarkConfig)."""
        return str(self._runs_dir / run_id)

    def finalize_run(self, run_id: str) -> RunMetadata:
        """Finalize a run by collecting scores from all benchmark results.

        Call this after all benchmarks have completed. Updates metadata.json
        with the final scores.

        Args:
            run_id: The run to finalize.

        Returns:
            Updated RunMetadata with scores populated.
        """
        run_path = self._runs_dir / run_id
        metadata_path = run_path / "metadata.json"

        with open(metadata_path) as f:
            metadata = RunMetadata.from_dict(json.load(f))

        # Collect scores from result.json files
        for benchmark in metadata.benchmarks:
            result_path = run_path / benchmark / "result.json"
            if result_path.exists():
                try:
                    with open(result_path) as f:
                        result = json.load(f)
                    metadata.scores[benchmark] = result.get("score", 0.0)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    logger.warning(
                        f"Skipping benchmark {benchmark} in run {run_id}: "
                        f"failed to load result.json: {e}"
                    )

        # Update metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Finalized run {run_id}: {metadata.scores}")
        return metadata

    def list_runs(self) -> list[RunMetadata]:
        """List all evaluation runs, most recent first."""
        runs = []
        if not self._index_path.exists():
            return runs

        # Scan run directories for full metadata
        for run_dir in sorted(self._runs_dir.iterdir(), reverse=True):
            meta_path = run_dir / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        runs.append(RunMetadata.from_dict(json.load(f)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Skipping run {run_dir.name}: failed to load metadata.json: {e}")
        return runs

    def load_run(self, run_id: str) -> RunMetadata:
        """Load metadata for a specific run.

        Args:
            run_id: The run to load.

        Returns:
            RunMetadata for the given run.

        Raises:
            FileNotFoundError: If the run does not exist.
        """
        meta_path = self._runs_dir / run_id / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Run {run_id} not found at {meta_path}")
        with open(meta_path) as f:
            return RunMetadata.from_dict(json.load(f))

    def get_result(self, run_id: str, benchmark: str) -> BenchmarkResult | None:
        """Get the aggregated result for a benchmark in a run.

        Args:
            run_id: The run to query.
            benchmark: Benchmark name (e.g. ``"gsm8k"``).

        Returns:
            BenchmarkResult or None if not found.
        """
        path = self._runs_dir / run_id / benchmark / "result.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                d = json.load(f)
            if "pass_at_k" in d:
                d["pass_at_k"] = {int(k): v for k, v in d["pass_at_k"].items()}
            return BenchmarkResult(**d)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to read result for {run_id}/{benchmark}: {e}")
            return None

    def get_summary(self, run_id: str) -> dict[str, dict]:
        """Get the combined summary across all benchmarks in a run.

        Args:
            run_id: The run to query.

        Returns:
            Dict mapping benchmark name to score/count info.
        """
        path = self._runs_dir / run_id / "summary.json"
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to read summary for {run_id}: {e}")
            return {}

    def get_trajectories(
        self,
        run_id: str,
        benchmark: str,
        correct_only: bool = False,
        incorrect_only: bool = False,
        errors_only: bool = False,
    ) -> list[StoredTrajectory]:
        """Get trajectories for a specific run and benchmark.

        Args:
            run_id: The run to query.
            benchmark: Benchmark name (e.g. ``"gsm8k"``).
            correct_only: If True, return only trajectories with reward > 0.
            incorrect_only: If True, return only trajectories with reward == 0
                and no error.
            errors_only: If True, return only trajectories with an error.

        Returns:
            List of StoredTrajectory objects matching the filters.
        """
        path = self._runs_dir / run_id / benchmark / "trajectories.jsonl"
        if not path.exists():
            return []

        trajectories = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    t = StoredTrajectory.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    logger.warning(
                        f"Skipping corrupted trajectory line {line_num} in "
                        f"{run_id}/{benchmark}/trajectories.jsonl: {e}"
                    )
                    continue
                if correct_only and t.reward <= 0:
                    continue
                if incorrect_only and (t.reward > 0 or t.error is not None):
                    continue
                if errors_only and t.error is None:
                    continue
                trajectories.append(t)
        return trajectories

    def compare_runs(
        self,
        run_a_id: str,
        run_b_id: str,
        benchmark: str,
    ) -> RunComparison:
        """Compare two runs on a specific benchmark.

        Matches trajectories by example_id (from ``StoredTrajectory.example_id``
        or falling back to ``idx``). Identifies regressions (A correct, B wrong)
        and improvements (A wrong, B correct).

        Args:
            run_a_id: First run ID (typically the earlier checkpoint).
            run_b_id: Second run ID (typically the later checkpoint).
            benchmark: Benchmark name to compare.

        Returns:
            RunComparison with regressions, improvements, and score delta.
        """
        trajs_a = self.get_trajectories(run_a_id, benchmark)
        trajs_b = self.get_trajectories(run_b_id, benchmark)

        # Build index by example_id
        def _get_id(t: StoredTrajectory) -> str:
            return t.example_id if t.example_id else str(t.idx)

        index_a = {_get_id(t): t for t in trajs_a}
        index_b = {_get_id(t): t for t in trajs_b}

        shared_ids = set(index_a.keys()) & set(index_b.keys())

        regressions = {}
        improvements = {}
        for ex_id in shared_ids:
            a, b = index_a[ex_id], index_b[ex_id]
            if a.reward > 0 and b.reward <= 0:
                regressions[ex_id] = (a, b)
            elif a.reward <= 0 and b.reward > 0:
                improvements[ex_id] = (a, b)

        # Load scores
        meta_a = self.load_run(run_a_id)
        meta_b = self.load_run(run_b_id)
        score_a = meta_a.scores.get(benchmark, 0.0)
        score_b = meta_b.scores.get(benchmark, 0.0)

        return RunComparison(
            benchmark=benchmark,
            run_a_id=run_a_id,
            run_b_id=run_b_id,
            score_a=score_a,
            score_b=score_b,
            score_delta=score_b - score_a,
            regressions=regressions,
            improvements=improvements,
            num_shared=len(shared_ids),
        )

    def print_comparison(self, comparison: RunComparison) -> None:
        """Pretty-print a run comparison to stdout.

        Args:
            comparison: The RunComparison to display (from :meth:`compare_runs`).
        """
        print(f"=== {comparison.benchmark}: {comparison.run_a_id} vs {comparison.run_b_id} ===")
        print(f"  Score: {comparison.score_a:.3f} -> {comparison.score_b:.3f} "
              f"(delta={comparison.score_delta:+.3f})")
        print(f"  Shared examples: {comparison.num_shared}")
        print(f"  Regressions: {len(comparison.regressions)} "
              f"(correct in A, wrong in B)")
        print(f"  Improvements: {len(comparison.improvements)} "
              f"(wrong in A, correct in B)")

        if comparison.regressions:
            print(f"\n  Top regressions:")
            for ex_id, (a, b) in list(comparison.regressions.items())[:5]:
                print(f"    {ex_id}: {a.logs.get('expected', '?')} "
                      f"(A: {a.logs.get('extracted', '?')}, B: {b.logs.get('extracted', '?')})")

    def print_dashboard(self) -> None:
        """Print a summary dashboard of all runs."""
        runs = self.list_runs()
        if not runs:
            print("No evaluation runs found.")
            return

        # Collect all benchmark names
        all_benchmarks = sorted({b for r in runs for b in r.scores})

        # Header
        header = f"{'Run ID':30s} {'Checkpoint':20s}"
        for b in all_benchmarks:
            header += f" {b:>10s}"
        print(header)
        print("-" * len(header))

        for r in runs:
            row = f"{r.run_id:30s} {(r.checkpoint_name or 'base'):20s}"
            for b in all_benchmarks:
                score = r.scores.get(b)
                row += f" {score:10.3f}" if score is not None else f" {'—':>10s}"
            print(row)
