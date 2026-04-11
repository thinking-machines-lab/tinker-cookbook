"""Typed data access for a single training run.

Provides both read and write methods for all training artifacts:
metrics, config, timing spans, checkpoints, rollouts, and logtrees.
All I/O goes through the ``Storage`` protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tinker_cookbook.stores._base import BaseStore
from tinker_cookbook.stores._incremental import IncrementalReader
from tinker_cookbook.stores.storage import Storage

logger = logging.getLogger(__name__)

_ITERATION_RE = re.compile(r"^iteration_(\d+)$")


class _Unset:
    """Sentinel that survives pickle (unlike bare object())."""

    _instance: _Unset | None = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


_UNSET = _Unset()


@dataclass
class IterationInfo:
    """Metadata about a single training iteration directory.

    Populated by :meth:`TrainingRunStore.list_iterations` by scanning
    the contents of each ``iteration_NNNNNN/`` directory.
    """

    iteration: int
    """Iteration number (from directory name ``iteration_NNNNNN``)."""
    has_train_rollouts: bool = False
    """Whether ``train_rollout_summaries.jsonl`` exists."""
    has_train_logtree: bool = False
    """Whether ``train_logtree.json`` exists."""
    eval_labels: list[str] = field(default_factory=list)
    """Eval labels found (e.g., ``['gsm8k']`` from ``eval_gsm8k_rollout_summaries.jsonl``)."""


class TrainingRunStore(BaseStore):
    """Typed read/write access to one training run's data.

    All file I/O goes through the ``Storage`` protocol — no direct
    ``Path``/``open()`` usage. Pickle-serializable when freshly
    constructed (lazy reader init).
    """

    def __init__(self, storage: Storage, prefix: str = "") -> None:
        super().__init__(storage, prefix)
        # Lazy — created on first access to keep pickle-safe
        self._metrics: IncrementalReader | None = None
        self._timing: IncrementalReader | None = None
        self._config: Any = _UNSET
        self._checkpoints: Any = _UNSET
        self._checkpoints_size: int = -1  # File size at last read, for invalidation
        self._rollout_cache: dict[str, list[dict[str, Any]]] = {}
        self._rollout_cache_sizes: dict[str, int] = {}  # File sizes for invalidation

    def url(self, path: str = "") -> str:
        """Return a human-readable URI for a path within this run.

        Useful for logging in distributed workers::

            logger.info("Writing metrics to %s", store.url("metrics.jsonl"))
        """
        return self.storage.url(self._path(path))

    @staticmethod
    def _iter_dir(iteration: int) -> str:
        return f"iteration_{iteration:06d}"

    def _read_jsonl_typed(
        self, from_dict: Callable[[dict[str, Any]], Any], *parts: str
    ) -> list[Any]:
        """Read JSONL → typed objects. Skips corrupted records."""
        records = []
        for d in self._read_jsonl(*parts):
            try:
                records.append(from_dict(d))
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping corrupted record in %s: %s", self._path(*parts), e)
        return records

    # ── Config ────────────────────────────────────────────────────────

    def read_config(self) -> dict[str, Any] | None:
        """Read config.json (cached after first read)."""
        if self._config is not _UNSET:
            return self._config
        self._config = self._read_json("config.json")
        return self._config

    # ── Metrics (incremental) ─────────────────────────────────────────

    def _get_metrics(self) -> IncrementalReader:
        if self._metrics is None:
            self._metrics = IncrementalReader(self.storage, self._path("metrics.jsonl"))
        return self._metrics

    def read_metrics(self) -> list[dict[str, Any]]:
        """Read all metrics (incremental — only new data from disk)."""
        reader = self._get_metrics()
        reader.read()
        return reader.records

    def read_new_metrics(self) -> list[dict[str, Any]]:
        """Read only metrics added since last call."""
        return self._get_metrics().read()

    def metric_count(self) -> int:
        """Number of metric records read so far."""
        return self._get_metrics().total_read

    def latest_metric(self) -> dict[str, Any] | None:
        """Return the last metric record, or ``None`` if empty."""
        return self._get_metrics().latest

    def metric_keys(self) -> set[str]:
        """All metric keys seen so far (excluding 'step')."""
        return self._get_metrics().known_keys

    # ── Rollouts (typed, cached) ──────────────────────────────────────

    @staticmethod
    def _rollout_filename(base_name: str) -> str:
        return f"{base_name}_rollout_summaries.jsonl"

    def read_rollouts(self, iteration: int, base_name: str = "train") -> list[dict[str, Any]]:
        """Read rollout summaries for an iteration as raw dicts.

        Cached per (iteration, base_name), invalidated when file size changes.

        Args:
            iteration: Training iteration number.
            base_name: Prefix for the JSONL file (e.g. ``"train"``,
                ``"eval_gsm8k"``). Matches the naming used by
                ``rollout_summaries_jsonl_path()`` in RL training.
        """
        filename = self._rollout_filename(base_name)
        cache_key = f"{iteration}/{filename}"
        if cache_key in self._rollout_cache:
            # Invalidate if file size changed (e.g., external writer added rollouts)
            stat = self.storage.stat(self._path(self._iter_dir(iteration), filename))
            current_size = stat.size if stat else 0
            if current_size == self._rollout_cache_sizes.get(cache_key, -1):
                return self._rollout_cache[cache_key]

        path = self._path(self._iter_dir(iteration), filename)
        stat = self.storage.stat(path)
        records = self._read_jsonl(self._iter_dir(iteration), filename)
        self._rollout_cache[cache_key] = records
        self._rollout_cache_sizes[cache_key] = stat.size if stat else 0
        return records

    def read_single_rollout(
        self,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        base_name: str = "train",
    ) -> dict[str, Any] | None:
        """Find one rollout by group and trajectory index, or ``None``."""
        for r in self.read_rollouts(iteration, base_name):
            if r.get("group_idx") == group_idx and r.get("traj_idx") == traj_idx:
                return r
        return None

    def read_rollouts_by_group(
        self, iteration: int, group_idx: int, base_name: str = "train"
    ) -> list[dict[str, Any]]:
        """Return all rollouts for a specific group index."""
        return [
            r for r in self.read_rollouts(iteration, base_name) if r.get("group_idx") == group_idx
        ]

    # ── Checkpoints (typed) ───────────────────────────────────────────

    def read_checkpoints(self) -> list[dict[str, Any]]:
        """Read checkpoints.jsonl (cached, invalidated when file size changes)."""
        if self._checkpoints is not _UNSET:
            stat = self.storage.stat(self._path("checkpoints.jsonl"))
            current_size = stat.size if stat else 0
            if current_size == self._checkpoints_size:
                return self._checkpoints
        stat = self.storage.stat(self._path("checkpoints.jsonl"))
        self._checkpoints_size = stat.size if stat else 0
        self._checkpoints = self._read_jsonl("checkpoints.jsonl")
        return self._checkpoints

    def read_checkpoint_records(self) -> list[Any]:
        """Read checkpoints.jsonl as ``CheckpointRecord`` objects."""
        from tinker_cookbook.checkpoint_utils import CheckpointRecord

        return self._read_jsonl_typed(CheckpointRecord.from_dict, "checkpoints.jsonl")

    # ── Timing (incremental) ──────────────────────────────────────────

    def _get_timing(self) -> IncrementalReader:
        if self._timing is None:
            self._timing = IncrementalReader(self.storage, self._path("timing_spans.jsonl"))
        return self._timing

    def read_timing(self) -> list[dict[str, Any]]:
        """Read all timing records (incremental — only new data from disk)."""
        reader = self._get_timing()
        reader.read()
        return reader.records

    def flatten_timing_spans(
        self,
        *,
        step: int | None = None,
        step_start: int | None = None,
        step_end: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract flat spans from timing records, optionally filtered by step range."""
        spans: list[dict[str, Any]] = []
        for record in self.read_timing():
            s = record.get("step", 0)
            if step is not None and s != step:
                continue
            if step_start is not None and s < step_start:
                continue
            if step_end is not None and s > step_end:
                continue
            if "spans" in record:
                for span in record["spans"]:
                    spans.append({"step": s, **span})
            else:
                spans.append(record)
        return spans

    def build_timing_tree(self, step: int) -> dict[str, Any]:
        """Build a hierarchical span tree for a specific step.

        Groups spans by ``group_idx`` attribute and nests children within
        parents based on wall-time containment.
        """
        spans = self.flatten_timing_spans(step=step)
        if not spans:
            return {"step": step, "root": None}
        sorted_spans = sorted(spans, key=lambda s: (s.get("wall_start", 0), -s.get("duration", 0)))
        nodes: list[dict[str, Any]] = [
            {
                "name": s.get("name", "?"),
                "duration": s.get("duration", 0),
                "wall_start": s.get("wall_start", 0),
                "wall_end": s.get("wall_end", 0),
                "attributes": s.get("attributes", {}),
                "children": [],
            }
            for s in sorted_spans
        ]
        eps = 0.01

        grouped_spans: dict[int, list[dict[str, Any]]] = {}
        ungrouped: list[dict[str, Any]] = []
        for node in nodes:
            gidx = node.get("attributes", {}).get("group_idx")
            if gidx is not None:
                grouped_spans.setdefault(gidx, []).append(node)
            else:
                ungrouped.append(node)

        ungrouped.sort(key=lambda s: (s.get("wall_start", 0), -s.get("duration", 0)))
        root_children: list[dict[str, Any]] = []
        stack: list[dict[str, Any]] = []
        for node in ungrouped:
            while stack and stack[-1]["wall_end"] + eps < node["wall_start"]:
                stack.pop()
            if stack and node["wall_end"] <= stack[-1]["wall_end"] + eps:
                stack[-1]["children"].append(node)
            else:
                while stack and node["wall_end"] > stack[-1]["wall_end"] + eps:
                    stack.pop()
                if stack:
                    stack[-1]["children"].append(node)
                else:
                    root_children.append(node)
            stack.append(node)

        if grouped_spans:
            all_grouped = [s for spans in grouped_spans.values() for s in spans]
            group_start = min(s["wall_start"] for s in all_grouped)
            group_end = max(s["wall_end"] for s in all_grouped)

            def find_parent(
                children: list[dict[str, Any]],
            ) -> dict[str, Any] | None:
                for child in children:
                    if (
                        child["wall_start"] <= group_start + eps
                        and child["wall_end"] >= group_end - eps
                    ):
                        deeper = find_parent(child.get("children", []))
                        return deeper if deeper else child
                return None

            parent = find_parent(root_children)
            target = parent["children"] if parent else root_children

            for gidx in sorted(grouped_spans.keys()):
                gspans = sorted(
                    grouped_spans[gidx],
                    key=lambda s: (s["wall_start"], -s["duration"]),
                )
                group_node: dict[str, Any] = {
                    "name": f"group {gidx}",
                    "duration": max(s["wall_end"] for s in gspans)
                    - min(s["wall_start"] for s in gspans),
                    "wall_start": min(s["wall_start"] for s in gspans),
                    "wall_end": max(s["wall_end"] for s in gspans),
                    "attributes": {"group_idx": gidx},
                    "children": [],
                }
                gstack: list[dict[str, Any]] = []
                for node in gspans:
                    while gstack and gstack[-1]["wall_end"] + eps < node["wall_start"]:
                        gstack.pop()
                    if gstack and node["wall_end"] <= gstack[-1]["wall_end"] + eps:
                        gstack[-1]["children"].append(node)
                    else:
                        group_node["children"].append(node)
                    gstack.append(node)
                target.append(group_node)

            target.sort(key=lambda s: s.get("wall_start", 0))

        all_starts = [n["wall_start"] for n in nodes]
        all_ends = [n["wall_end"] for n in nodes]
        total = max(all_ends) - min(all_starts) if all_starts else 0
        return {
            "step": step,
            "total_duration": total,
            "root": {
                "name": "iteration",
                "duration": total,
                "wall_start": min(all_starts) if all_starts else 0,
                "wall_end": max(all_ends) if all_ends else 0,
                "attributes": {},
                "children": root_children,
            },
        }

    # ── Logtree ───────────────────────────────────────────────────────

    def read_logtree(self, iteration: int, base_name: str = "train") -> dict[str, Any] | None:
        """Read a logtree JSON file for an iteration, or ``None`` if missing."""
        return self._read_json(self._iter_dir(iteration), f"{base_name}_logtree.json")

    def list_logtrees(self, iteration: int) -> list[str]:
        """List logtree base names for an iteration (e.g. ``["train", "eval_gsm8k"]``)."""
        items = self.storage.list_dir(self._path(self._iter_dir(iteration)))
        return sorted(n[: -len("_logtree.json")] for n in items if n.endswith("_logtree.json"))

    # ── Iterations ────────────────────────────────────────────────────

    def list_iterations(self) -> list[IterationInfo]:
        """List all iteration directories with metadata about their contents."""
        iterations: list[IterationInfo] = []
        for child in self.storage.list_dir(self._path()):
            match = _ITERATION_RE.match(child)
            if not match:
                continue
            info = IterationInfo(iteration=int(match.group(1)))
            iter_prefix = self._path(child)
            for f in self.storage.list_dir(iter_prefix):
                if f == "train_rollout_summaries.jsonl":
                    info.has_train_rollouts = True
                elif f == "train_logtree.json":
                    info.has_train_logtree = True
                elif f.startswith("eval_") and f.endswith("_rollout_summaries.jsonl"):
                    info.eval_labels.append(f[len("eval_") : -len("_rollout_summaries.jsonl")])
            iterations.append(info)
        iterations.sort(key=lambda x: x.iteration)
        return iterations

    # ── Writes ────────────────────────────────────────────────────────

    def write_config(self, config: dict[str, Any]) -> None:
        """Write config.json (overwrites if exists, updates cache)."""
        self._write_json(config, "config.json")
        self._config = config

    def write_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Append one metrics record to metrics.jsonl.

        The record is ``{"step": step, ...metrics}`` if step is given,
        otherwise just the metrics dict.
        """
        record: dict[str, Any] = {"step": step} if step is not None else {}
        record.update(metrics)
        self._append_jsonl(record, "metrics.jsonl")

    def write_timing_spans(self, step: int, spans: list[dict[str, Any]]) -> None:
        """Append one timing record to timing_spans.jsonl.

        Each span dict should have keys: ``name``, ``duration``,
        ``wall_start``, ``wall_end``.
        """
        if not spans:
            return
        self._append_jsonl({"step": step, "spans": spans}, "timing_spans.jsonl")

    def write_checkpoint(self, record: dict[str, Any]) -> None:
        """Append one checkpoint record to checkpoints.jsonl.

        Accepts a raw dict (e.g. from ``CheckpointRecord.to_dict()``).
        Must contain at least a ``"name"`` key.
        """
        self._append_jsonl(record, "checkpoints.jsonl")
        self._checkpoints = _UNSET  # Invalidate cache

    def write_rollouts(
        self,
        iteration: int,
        records: list[dict[str, Any]],
        base_name: str = "train",
    ) -> None:
        """Write rollout summaries for an iteration (overwrites).

        Args:
            iteration: Training iteration number.
            records: List of trajectory dicts to write.
            base_name: Prefix for the JSONL file (e.g. ``"train"``,
                ``"eval_gsm8k"``). Must match the ``base_name`` used
                in ``read_rollouts()``.
        """
        filename = self._rollout_filename(base_name)
        lines = [json.dumps(r) for r in records]
        data = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
        self.storage.write(self._path(self._iter_dir(iteration), filename), data)

        # Invalidate read cache
        cache_key = f"{iteration}/{filename}"
        self._rollout_cache.pop(cache_key, None)

    def write_logtree(self, iteration: int, data: dict[str, Any], base_name: str = "train") -> None:
        """Write a logtree JSON file for an iteration (overwrites)."""
        self._write_json(data, self._iter_dir(iteration), f"{base_name}_logtree.json")

    def write_code_diff(self, diff: str) -> None:
        """Write code.diff (overwrites)."""
        self.storage.write(self._path("code.diff"), diff.encode("utf-8"))

    # ── Async variants ────────────────────────────────────────────────

    async def aread_config(self) -> dict[str, Any] | None:
        """Async version of :meth:`read_config`."""
        return await asyncio.to_thread(self.read_config)

    async def aread_metrics(self) -> list[dict[str, Any]]:
        """Async version of :meth:`read_metrics`."""
        return await asyncio.to_thread(self.read_metrics)

    async def aread_new_metrics(self) -> list[dict[str, Any]]:
        """Async version of :meth:`read_new_metrics`."""
        return await asyncio.to_thread(self.read_new_metrics)

    async def aread_rollouts(
        self, iteration: int, base_name: str = "train"
    ) -> list[dict[str, Any]]:
        """Async version of :meth:`read_rollouts`."""
        return await asyncio.to_thread(self.read_rollouts, iteration, base_name)

    async def aread_checkpoints(self) -> list[dict[str, Any]]:
        """Async version of :meth:`read_checkpoints`."""
        return await asyncio.to_thread(self.read_checkpoints)

    async def aread_timing(self) -> list[dict[str, Any]]:
        """Async version of :meth:`read_timing`."""
        return await asyncio.to_thread(self.read_timing)

    async def aread_logtree(
        self, iteration: int, base_name: str = "train"
    ) -> dict[str, Any] | None:
        """Async version of :meth:`read_logtree`."""
        return await asyncio.to_thread(self.read_logtree, iteration, base_name)

    async def awrite_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Async version of :meth:`write_metrics`."""
        await asyncio.to_thread(self.write_metrics, metrics, step)

    async def awrite_checkpoint(self, record: dict[str, Any]) -> None:
        """Async version of :meth:`write_checkpoint`."""
        await asyncio.to_thread(self.write_checkpoint, record)
