"""EvalStore — manages evaluation runs across checkpoints.

Refactored from ``eval/store.py`` to use the Storage protocol.
Supports local and cloud backends.

Storage layout::

    {prefix}/
      runs.jsonl                        # Append-only index
      runs/
        {run_id}/
          metadata.json                 # RunMetadata
          {benchmark}/
            result.json                 # BenchmarkResult
            trajectories.jsonl          # StoredTrajectory per line
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tinker_cookbook.stores.storage import Storage, storage_join

if TYPE_CHECKING:
    from tinker_cookbook.eval.benchmarks._types import BenchmarkResult, StoredTrajectory


def _get_eval_types() -> tuple[type, type]:
    """Lazy import to break circular: stores.eval_store → eval._types → eval.__init__ → eval.store → stores.eval_store"""
    import tinker_cookbook.eval.benchmarks._types as t

    return t.BenchmarkResult, t.StoredTrajectory


logger = logging.getLogger(__name__)


@dataclass
class RunMetadata:
    """Metadata for a single evaluation run."""

    run_id: str
    model_name: str
    checkpoint_path: str | None
    checkpoint_name: str | None
    benchmarks: list[str]
    timestamp: str
    config: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunMetadata:
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class RunComparison:
    """Result of comparing two runs on a specific benchmark.

    Stores example IDs rather than full trajectory objects to avoid
    holding large amounts of data in memory for big benchmarks.
    Use ``EvalStore.read_single_trajectory`` to load specific examples.
    """

    benchmark: str
    run_a_id: str
    run_b_id: str
    score_a: float
    score_b: float
    score_delta: float
    regressions: list[str]
    """Example IDs where run_a was correct but run_b was wrong."""
    improvements: list[str]
    """Example IDs where run_a was wrong but run_b was correct."""
    num_shared: int


class EvalStore:
    """Manages evaluation runs across checkpoints.

    All file I/O goes through the ``Storage`` protocol, making this
    backend-agnostic (local disk, S3, GCS).

    Pickle-serializable when freshly constructed.
    """

    def __init__(self, storage_or_path: Storage | str | Path, prefix: str = "") -> None:
        if isinstance(storage_or_path, (str, Path)):
            # Backward compat: EvalStore("/path/to/eval_store")
            from tinker_cookbook.stores.storage import LocalStorage

            self._storage: Storage = LocalStorage(Path(storage_or_path).expanduser())
            self._prefix = prefix
        else:
            self._storage = storage_or_path
            self._prefix = prefix

    def _path(self, *parts: str) -> str:
        return storage_join(self._prefix, *parts)

    def _read_json(self, *parts: str) -> dict[str, Any] | None:
        try:
            data = self._storage.read(self._path(*parts))
            return json.loads(data)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", self._path(*parts), e)
            return None

    def _write_json(self, data: dict[str, Any], *parts: str) -> None:
        self._storage.write(self._path(*parts), json.dumps(data, indent=2).encode("utf-8"))

    def _append_jsonl(self, record: dict[str, Any], *parts: str) -> None:
        self._storage.append(self._path(*parts), (json.dumps(record) + "\n").encode("utf-8"))

    def _read_jsonl(self, *parts: str) -> list[dict[str, Any]]:
        try:
            data = self._storage.read(self._path(*parts))
        except FileNotFoundError:
            return []
        except OSError as e:
            logger.warning("Failed to read %s: %s", self._path(*parts), e)
            return []
        records = []
        for line in data.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line in %s", self._path(*parts))
        return records

    # ── Run management ────────────────────────────────────────────────

    def create_run(
        self,
        model_name: str,
        benchmarks: list[str],
        checkpoint_path: str | None = None,
        checkpoint_name: str | None = None,
        config: dict | None = None,
        run_id: str | None = None,
    ) -> str:
        """Create a new evaluation run and return its run_id."""
        if run_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_label = checkpoint_name or "base"
            short_id = uuid.uuid4().hex[:6]
            run_id = f"{ckpt_label}_{ts}_{short_id}"

        metadata = RunMetadata(
            run_id=run_id,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            benchmarks=benchmarks,
            config=config or {},
            timestamp=datetime.now().isoformat(),
        )

        self._write_json(metadata.to_dict(), "runs", run_id, "metadata.json")
        self._append_jsonl(
            {
                "run_id": run_id,
                "timestamp": metadata.timestamp,
                "model": model_name,
                "checkpoint": checkpoint_name,
            },
            "runs.jsonl",
        )

        logger.info("Created eval run: %s", run_id)
        return run_id

    def run_dir(self, run_id: str) -> str:
        """Return filesystem path for backward compat with BenchmarkConfig.save_dir.

        Only works with LocalStorage (returns a local path string).
        For cloud backends, use ``url()`` on the storage directly.
        """
        if not hasattr(self._storage, "root"):
            raise RuntimeError(
                "run_dir() only works with LocalStorage. "
                f"Use storage.url() instead: {self._storage.url(self._path('runs', run_id))}"
            )
        root = getattr(self._storage, "root")  # noqa: B009
        return str(root / self._path("runs", run_id))

    def finalize_run(self, run_id: str) -> RunMetadata:
        """Collect scores from benchmark results and update metadata."""
        meta_dict = self._read_json("runs", run_id, "metadata.json")
        if meta_dict is None:
            raise FileNotFoundError(f"Run {run_id} not found")
        metadata = RunMetadata.from_dict(meta_dict)

        for benchmark in metadata.benchmarks:
            result_dict = self._read_json("runs", run_id, benchmark, "result.json")
            if result_dict:
                metadata.scores[benchmark] = result_dict.get("score", 0.0)

        self._write_json(metadata.to_dict(), "runs", run_id, "metadata.json")
        logger.info("Finalized run %s: %s", run_id, metadata.scores)
        return metadata

    # ── Queries ───────────────────────────────────────────────────────

    def list_runs(self) -> list[RunMetadata]:
        """List all evaluation runs, most recent first."""
        runs = []
        for name in sorted(self._storage.list_dir(self._path("runs")), reverse=True):
            meta = self._read_json("runs", name, "metadata.json")
            if meta:
                try:
                    runs.append(RunMetadata.from_dict(meta))
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning("Skipping run %s: %s", name, e)
        return runs

    def load_run(self, run_id: str) -> RunMetadata:
        """Load metadata for a specific run. Raises FileNotFoundError if missing."""
        meta = self._read_json("runs", run_id, "metadata.json")
        if meta is None:
            raise FileNotFoundError(f"Run {run_id} not found")
        return RunMetadata.from_dict(meta)

    def list_benchmarks(self, run_id: str) -> list[str]:
        """List benchmark names that have results for a run."""
        items = self._storage.list_dir(self._path("runs", run_id))
        return sorted(
            name
            for name in items
            if self._storage.exists(self._path("runs", run_id, name, "result.json"))
        )

    def read_result(self, run_id: str, benchmark: str) -> BenchmarkResult | None:
        """Get aggregated result for a benchmark."""
        BenchmarkResult, _ = _get_eval_types()
        d = self._read_json("runs", run_id, benchmark, "result.json")
        if d is None:
            return None
        try:
            if "pass_at_k" in d:
                d["pass_at_k"] = {int(k): v for k, v in d["pass_at_k"].items()}
            return BenchmarkResult(**d)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Failed to parse result %s/%s: %s", run_id, benchmark, e)
            return None

    def read_trajectories(
        self,
        run_id: str,
        benchmark: str,
        *,
        correct_only: bool = False,
        incorrect_only: bool = False,
        errors_only: bool = False,
    ) -> list[StoredTrajectory]:
        """Get trajectories with optional filtering."""
        _, StoredTrajectory = _get_eval_types()
        raw = self._read_jsonl("runs", run_id, benchmark, "trajectories.jsonl")
        trajectories = []
        for d in raw:
            try:
                t = StoredTrajectory.from_dict(d)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping corrupted trajectory in %s/%s: %s", run_id, benchmark, e)
                continue
            if correct_only and t.reward <= 0:
                continue
            if incorrect_only and (t.reward > 0 or t.error is not None):
                continue
            if errors_only and t.error is None:
                continue
            trajectories.append(t)
        return trajectories

    def read_single_trajectory(
        self, run_id: str, benchmark: str, idx: int
    ) -> StoredTrajectory | None:
        """Get a single trajectory by index (O(n) scan — loads all trajectories)."""
        for t in self.read_trajectories(run_id, benchmark):
            if t.idx == idx:
                return t
        return None

    def read_summary(self, run_id: str) -> dict[str, Any] | None:
        """Read the combined summary for a run, or ``None`` if missing."""
        return self._read_json("runs", run_id, "summary.json")

    # ── Writes (for eval runner) ──────────────────────────────────────

    def write_result(self, run_id: str, result: BenchmarkResult) -> None:
        """Save a benchmark result."""
        d = asdict(result)
        if d.get("pass_at_k"):
            d["pass_at_k"] = {str(k): v for k, v in d["pass_at_k"].items()}
        self._write_json(d, "runs", run_id, result.name, "result.json")

    def append_trajectory(self, run_id: str, benchmark: str, traj: StoredTrajectory) -> None:
        """Append one trajectory to the JSONL file."""
        self._append_jsonl(dict(traj.to_dict()), "runs", run_id, benchmark, "trajectories.jsonl")

    def write_summary(self, run_id: str, results: dict[str, BenchmarkResult]) -> None:
        """Save a combined summary."""
        summary = {}
        for name, result in results.items():
            summary[name] = {
                "score": result.score,
                "num_examples": result.num_examples,
                "num_correct": result.num_correct,
                "num_errors": result.num_errors,
            }
        self._write_json(summary, "runs", run_id, "summary.json")

    def delete_run(self, run_id: str) -> None:
        """Delete all data for a run. Idempotent (no error if already gone).

        Removes metadata, summary, and all benchmark result/trajectory files.
        The ``runs.jsonl`` index is append-only and not modified; ``list_runs()``
        checks for ``metadata.json`` existence so deleted runs are excluded.
        """
        for benchmark in self.list_benchmarks(run_id):
            self._storage.remove(self._path("runs", run_id, benchmark, "result.json"))
            self._storage.remove(self._path("runs", run_id, benchmark, "trajectories.jsonl"))
            self._storage.remove_dir(self._path("runs", run_id, benchmark))
        self._storage.remove(self._path("runs", run_id, "metadata.json"))
        self._storage.remove(self._path("runs", run_id, "summary.json"))
        self._storage.remove_dir(self._path("runs", run_id))

    # ── Cross-run comparison ──────────────────────────────────────────

    def compare_runs(self, run_a_id: str, run_b_id: str, benchmark: str) -> RunComparison:
        """Compare two runs on a specific benchmark.

        Returns example IDs for regressions/improvements rather than full
        trajectory objects. Use ``read_single_trajectory`` to inspect details.
        """
        trajs_a = self.read_trajectories(run_a_id, benchmark)
        trajs_b = self.read_trajectories(run_b_id, benchmark)

        def _get_id(t: StoredTrajectory) -> str:
            return t.example_id if t.example_id else str(t.idx)

        index_a = {_get_id(t): t for t in trajs_a}
        index_b = {_get_id(t): t for t in trajs_b}
        shared = set(index_a) & set(index_b)

        regressions: list[str] = []
        improvements: list[str] = []
        for ex_id in sorted(shared):
            a, b = index_a[ex_id], index_b[ex_id]
            if a.reward > 0 and b.reward <= 0:
                regressions.append(ex_id)
            elif a.reward <= 0 and b.reward > 0:
                improvements.append(ex_id)

        meta_a = self.load_run(run_a_id)
        meta_b = self.load_run(run_b_id)

        return RunComparison(
            benchmark=benchmark,
            run_a_id=run_a_id,
            run_b_id=run_b_id,
            score_a=meta_a.scores.get(benchmark, 0.0),
            score_b=meta_b.scores.get(benchmark, 0.0),
            score_delta=meta_b.scores.get(benchmark, 0.0) - meta_a.scores.get(benchmark, 0.0),
            regressions=regressions,
            improvements=improvements,
            num_shared=len(shared),
        )

    def scores_table(self) -> list[dict[str, Any]]:
        """Build a scores table across all runs."""
        table = []
        for run in self.list_runs():
            table.append(
                {
                    "run_id": run.run_id,
                    "model_name": run.model_name,
                    "checkpoint_name": run.checkpoint_name,
                    "timestamp": run.timestamp,
                    "scores": run.scores,
                }
            )
        return table

    def print_dashboard(self) -> None:
        """Print a summary dashboard to stdout."""
        runs = self.list_runs()
        if not runs:
            print("No evaluation runs found.")
            return
        all_benchmarks = sorted({b for r in runs for b in r.scores})
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

    def print_comparison(self, comparison: RunComparison) -> None:
        """Pretty-print a run comparison."""
        print(f"=== {comparison.benchmark}: {comparison.run_a_id} vs {comparison.run_b_id} ===")
        print(
            f"  Score: {comparison.score_a:.3f} -> {comparison.score_b:.3f} (delta={comparison.score_delta:+.3f})"
        )
        print(
            f"  Shared: {comparison.num_shared}, Regressions: {len(comparison.regressions)}, Improvements: {len(comparison.improvements)}"
        )

    # ── Async variants ────────────────────────────────────────────────

    async def alist_runs(self) -> list[RunMetadata]:
        return await asyncio.to_thread(self.list_runs)

    async def aread_trajectories(
        self, run_id: str, benchmark: str, **kw: Any
    ) -> list[StoredTrajectory]:
        return await asyncio.to_thread(self.read_trajectories, run_id, benchmark, **kw)

    async def aread_result(self, run_id: str, benchmark: str) -> BenchmarkResult | None:
        return await asyncio.to_thread(self.read_result, run_id, benchmark)

    async def ascores_table(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.scores_table)
