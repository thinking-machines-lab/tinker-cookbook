"""Rollout browser API routes."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from tinker_cookbook.chef.routes._helpers import require_run
from tinker_cookbook.stores import RunRegistry


def _to_base_name(split: str, label: str | None) -> str:
    """Convert split/label query params to the store's base_name."""
    return f"eval_{label}" if split != "train" and label else split


def create_router(resolve_registry: Callable[..., RunRegistry]) -> APIRouter:
    router = APIRouter(prefix="/api/runs", tags=["rollouts"])

    @router.get("/{run_id}/iterations/{iteration}/rollouts")
    def get_rollouts(
        run_id: str,
        iteration: int,
        split: str = Query("train"),
        label: str | None = Query(None),
        tag: str | None = Query(None),
        min_reward: float | None = Query(None),
        max_reward: float | None = Query(None),
        limit: int | None = Query(None, description="Max rollouts to return"),
        offset: int = Query(0, description="Number of rollouts to skip"),
        source: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        store = registry.get_training_store(run_id)
        base_name = _to_base_name(split, label)
        all_rollouts = store.read_rollouts(iteration, base_name)

        all_tags: set[str] = set()
        for r in all_rollouts:
            all_tags.update(r.get("tags", []))

        filtered = all_rollouts
        if tag is not None:
            filtered = [r for r in filtered if tag in r.get("tags", [])]
        if min_reward is not None:
            filtered = [r for r in filtered if r.get("total_reward", 0) >= min_reward]
        if max_reward is not None:
            filtered = [r for r in filtered if r.get("total_reward", 0) <= max_reward]

        summaries = [
            {
                "group_idx": r.get("group_idx"),
                "traj_idx": r.get("traj_idx"),
                "tags": r.get("tags", []),
                "total_reward": r.get("total_reward"),
                "final_reward": r.get("final_reward"),
                "num_steps": len(r.get("steps", [])),
                "total_tokens": sum(s.get("ac_len", 0) for s in r.get("steps", [])),
                "final_ob_len": r.get("final_ob_len"),
                "sampling_client_step": r.get("sampling_client_step"),
                "status": r.get("status"),
                "error_type": r.get("error_type"),
                "stop_reason": r.get("stop_reason"),
            }
            for r in filtered
        ]

        total = len(summaries)
        if offset:
            summaries = summaries[offset:]
        if limit is not None:
            summaries = summaries[:limit]

        return {
            "run_id": run_id,
            "iteration": iteration,
            "split": split,
            "total": total,
            "available_tags": sorted(all_tags),
            "rollouts": summaries,
        }

    @router.get("/{run_id}/iterations/{iteration}/rollouts/{group_idx}/{traj_idx}")
    def get_rollout_detail(
        run_id: str,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        split: str = Query("train"),
        label: str | None = Query(None),
        source: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        base_name = _to_base_name(split, label)
        rollout = registry.get_training_store(run_id).read_single_rollout(
            iteration, group_idx, traj_idx, base_name
        )
        if rollout is None:
            raise HTTPException(
                status_code=404,
                detail=f"Rollout ({group_idx}, {traj_idx}) not found at iteration {iteration}",
            )
        return rollout

    @router.get("/{run_id}/iterations/{iteration}/groups/{group_idx}")
    def get_group_rollouts(
        run_id: str,
        iteration: int,
        group_idx: int,
        split: str = Query("train"),
        label: str | None = Query(None),
        source: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        """Return all rollout details for a specific group."""
        registry = resolve_registry(source)
        require_run(registry, run_id)
        base_name = _to_base_name(split, label)
        group = registry.get_training_store(run_id).read_rollouts_by_group(
            iteration, group_idx, base_name
        )
        return {
            "run_id": run_id,
            "iteration": iteration,
            "group_idx": group_idx,
            "total": len(group),
            "rollouts": group,
        }

    @router.get("/{run_id}/iterations/{iteration}/logtree")
    def get_logtree(
        run_id: str,
        iteration: int,
        base_name: str = Query("train"),
        source: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        logtree = registry.get_training_store(run_id).read_logtree(iteration, base_name)
        if logtree is None:
            raise HTTPException(
                status_code=404, detail=f"Logtree not found for iteration {iteration}"
            )
        return logtree

    return router
