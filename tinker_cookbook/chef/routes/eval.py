"""Eval benchmark API routes."""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from tinker_cookbook.stores import RunRegistry


def create_router(resolve_registry: Callable[..., RunRegistry]) -> APIRouter:
    router = APIRouter(prefix="/api/eval", tags=["eval"])

    @router.get("/runs")
    def list_eval_runs(source: list[str] = Query(default=[])) -> list[dict[str, Any]]:
        registry = resolve_registry(source)
        store = registry.get_eval_store()
        if store is None:
            return []
        result = []
        for run in store.list_runs():
            benchmarks = store.list_benchmarks(run.run_id)
            result.append(
                {
                    "eval_run_id": run.run_id,
                    "model_name": run.model_name,
                    "checkpoint_path": run.checkpoint_path,
                    "checkpoint_name": run.checkpoint_name,
                    "timestamp": run.timestamp,
                    "benchmarks": benchmarks,
                    "scores": run.scores,
                }
            )
        return result

    @router.get("/runs/{eval_run_id}")
    def get_eval_run(eval_run_id: str, source: list[str] = Query(default=[])) -> dict[str, Any]:
        registry = resolve_registry(source)
        store = registry.get_eval_store()
        if store is None:
            raise HTTPException(status_code=404, detail="No eval data found")
        try:
            metadata = store.read_run(eval_run_id)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"Eval run '{eval_run_id}' not found"
            ) from exc

        benchmarks = store.list_benchmarks(eval_run_id)
        results: dict[str, Any] = {}
        for benchmark in benchmarks:
            result = store.read_result(eval_run_id, benchmark)
            if result:
                results[benchmark] = asdict(result)

        return {
            "eval_run_id": eval_run_id,
            "metadata": metadata.to_dict(),
            "benchmarks": benchmarks,
            "results": results,
        }

    @router.get("/runs/{eval_run_id}/{benchmark}/trajectories")
    def get_eval_trajectories(
        eval_run_id: str,
        benchmark: str,
        correct_only: bool = Query(False),
        incorrect_only: bool = Query(False),
        errors_only: bool = Query(False),
        source: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        if sum([correct_only, incorrect_only, errors_only]) > 1:
            raise HTTPException(status_code=400, detail="Only one filter flag can be set at a time")
        registry = resolve_registry(source)
        store = registry.get_eval_store()
        if store is None:
            raise HTTPException(status_code=404, detail="No eval data found")

        trajectories = store.read_trajectories(
            eval_run_id,
            benchmark,
            correct_only=correct_only,
            incorrect_only=incorrect_only,
            errors_only=errors_only,
        )

        summaries = [
            {
                "idx": t.idx,
                "example_id": t.example_id,
                "reward": t.reward,
                "num_turns": len(t.turns),
                "time_seconds": t.time_seconds,
                "error": t.error,
                "logs": t.logs,
            }
            for t in trajectories
        ]

        return {
            "eval_run_id": eval_run_id,
            "benchmark": benchmark,
            "total": len(summaries),
            "trajectories": summaries,
        }

    @router.get("/runs/{eval_run_id}/{benchmark}/trajectories/{idx}")
    def get_eval_trajectory_detail(
        eval_run_id: str,
        benchmark: str,
        idx: int,
        source: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        registry = resolve_registry(source)
        store = registry.get_eval_store()
        if store is None:
            raise HTTPException(status_code=404, detail="No eval data found")
        traj = store.read_single_trajectory(eval_run_id, benchmark, idx)
        if traj is None:
            raise HTTPException(
                status_code=404, detail=f"Trajectory {idx} not found in {benchmark}"
            )
        return traj.to_dict()

    @router.get("/scores")
    def get_scores_table(source: list[str] = Query(default=[])) -> list[dict[str, Any]]:
        registry = resolve_registry(source)
        store = registry.get_eval_store()
        if store is None:
            return []
        table = []
        for run in store.list_runs():
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

    return router
