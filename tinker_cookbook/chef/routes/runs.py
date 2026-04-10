"""Run discovery and detail API routes."""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from tinker_cookbook.chef.routes._helpers import require_run
from tinker_cookbook.stores import RunRegistry


def _build_eval_index(eval_store: Any) -> dict[str, dict[str, float]]:
    """Build checkpoint_path → best eval scores index."""
    result: dict[str, dict[str, float]] = {}
    if eval_store is None:
        return result
    for eval_run in eval_store.list_runs():
        if eval_run.checkpoint_path and eval_run.scores:
            existing = result.get(eval_run.checkpoint_path, {})
            for bench, score in eval_run.scores.items():
                if bench not in existing or score > existing[bench]:
                    existing[bench] = score
            result[eval_run.checkpoint_path] = existing
    return result


def create_router(resolve_registry: Callable[..., RunRegistry]) -> APIRouter:
    router = APIRouter(prefix="/api/runs", tags=["runs"])

    # Per-registry eval index cache (lives in the router closure, not module-level)
    _eval_cache: dict[int, dict[str, dict[str, float]]] = {}

    @router.get("")
    def list_runs(source: list[str] = Query(default=[])) -> list[dict[str, Any]]:
        registry = resolve_registry(source)
        runs = registry.get_runs()

        # Pre-compute eval scores (cached per registry instance)
        registry_id = id(registry)
        if registry_id not in _eval_cache:
            _eval_cache[registry_id] = _build_eval_index(registry.get_eval_store())
        eval_by_ckpt = _eval_cache[registry_id]

        result = []
        for run in runs:
            info = asdict(run)
            store = registry.get_training_store(run.run_id)
            config = store.read_config()
            if config:
                info["config_summary"] = _extract_config_summary(config)

            # Use incremental read + latest_metric to avoid copying all records
            store.read_metrics()
            latest = store.latest_metric()
            if latest is not None:
                info["latest_step"] = latest.get("step")

            # Find best eval scores across this run's checkpoints
            if eval_by_ckpt:
                best_scores: dict[str, float] = {}
                for ckpt in store.read_checkpoints():
                    path = ckpt.get("state_path") or ckpt.get("sampler_path")
                    if path and path in eval_by_ckpt:
                        for bench, score in eval_by_ckpt[path].items():
                            if bench not in best_scores or score > best_scores[bench]:
                                best_scores[bench] = score
                if best_scores:
                    info["eval_scores"] = best_scores

            result.append(info)
        return result

    @router.get("/{run_id}")
    def get_run(run_id: str, source: list[str] = Query(default=[])) -> dict[str, Any]:
        registry = resolve_registry(source)
        run = require_run(registry, run_id)
        info = asdict(run)
        store = registry.get_training_store(run_id)
        config = store.read_config()
        if config:
            info["config"] = config
        metrics = store.read_metrics()
        if metrics:
            info["latest_step"] = metrics[-1].get("step")
            info["total_steps"] = len(metrics)
        return info

    @router.get("/{run_id}/config")
    def get_config(run_id: str, source: list[str] = Query(default=[])) -> dict[str, Any]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        config = registry.get_training_store(run_id).read_config()
        if config is None:
            raise HTTPException(status_code=404, detail="No config.json found")
        return config

    @router.get("/{run_id}/iterations")
    def list_iterations(run_id: str, source: list[str] = Query(default=[])) -> list[dict[str, Any]]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        return [asdict(it) for it in registry.get_training_store(run_id).list_iterations()]

    @router.get("/{run_id}/checkpoints")
    def get_checkpoints(run_id: str, source: list[str] = Query(default=[])) -> list[dict[str, Any]]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        return registry.get_training_store(run_id).read_checkpoints()

    @router.get("/{run_id}/eval-scores")
    def get_eval_scores(run_id: str, source: list[str] = Query(default=[])) -> list[dict[str, Any]]:
        """Get eval scores matched to this run's checkpoints."""
        registry = resolve_registry(source)
        require_run(registry, run_id)

        checkpoints = registry.get_training_store(run_id).read_checkpoints()
        eval_store = registry.get_eval_store()
        if eval_store is None:
            return []

        # Build a map: checkpoint_path -> step
        ckpt_step_map: dict[str, int] = {}
        for ckpt in checkpoints:
            path = ckpt.get("state_path") or ckpt.get("sampler_path")
            step = ckpt.get("batch") or ckpt.get("loop_state", {}).get("batch", 0)
            if path and step is not None:
                ckpt_step_map[path] = step

        # Match eval runs to checkpoints
        matched = []
        for eval_run in eval_store.list_runs():
            step = None
            if eval_run.checkpoint_path:
                step = ckpt_step_map.get(eval_run.checkpoint_path)
            if step is None and eval_run.checkpoint_name:
                # Try matching by name (e.g., "000050" -> batch 50)
                for ckpt in checkpoints:
                    if ckpt.get("name") == eval_run.checkpoint_name:
                        step = ckpt.get("batch")
                        break

            matched.append(
                {
                    "eval_run_id": eval_run.run_id,
                    "checkpoint_name": eval_run.checkpoint_name,
                    "checkpoint_path": eval_run.checkpoint_path,
                    "step": step,
                    "scores": eval_run.scores,
                    "benchmarks": eval_run.benchmarks,
                    "timestamp": eval_run.timestamp,
                }
            )

        return sorted(matched, key=lambda x: x.get("step") or 0)

    return router


def _extract_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("model_name", "learning_rate", "batch_size", "n_batches", "lora_rank"):
        if key in config:
            summary[key] = config[key]
    return summary
