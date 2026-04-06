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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from tinker_cookbook.stores._incremental import IncrementalReader
from tinker_cookbook.stores.storage import Storage, storage_join

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

Status = Literal["running", "completed", "idle"]
TrainingType = Literal["rl", "sl", "dpo"]
_ACTIVE_THRESHOLD_SECONDS = 120


@dataclass
class IterationInfo:
    """Metadata about a single training iteration directory."""

    iteration: int
    has_train_rollouts: bool = False
    has_train_logtree: bool = False
    eval_labels: list[str] = field(default_factory=list)


class TrainingRunStore:
    """Typed read/write access to one training run's data.

    All file I/O goes through the ``Storage`` protocol — no direct
    ``Path``/``open()`` usage. Pickle-serializable when freshly
    constructed (lazy reader init).
    """

    def __init__(self, storage: Storage, prefix: str = "") -> None:
        self.storage = storage
        self.prefix = prefix
        # Lazy — created on first access to keep pickle-safe
        self._metrics: IncrementalReader | None = None
        self._timing: IncrementalReader | None = None
        self._config: Any = _UNSET
        self._rollout_cache: dict[str, list[dict[str, Any]]] = {}

    def url(self, path: str = "") -> str:
        """Return a human-readable URI for a path within this run.

        Useful for logging in distributed workers::

            logger.info("Writing metrics to %s", store.url("metrics.jsonl"))
        """
        return self.storage.url(self._path(path))

    def _path(self, *parts: str) -> str:
        return storage_join(self.prefix, *parts)

    @staticmethod
    def _iter_dir(iteration: int) -> str:
        return f"iteration_{iteration:06d}"

    # ── JSON/JSONL helpers ────────────────────────────────────────────

    def _read_json(self, *parts: str) -> dict[str, Any] | None:
        try:
            data = self.storage.read(self._path(*parts))
            return json.loads(data)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read JSON %s: %s", self._path(*parts), e)
            return None

    def _read_jsonl(self, *parts: str) -> list[dict[str, Any]]:
        try:
            data = self.storage.read(self._path(*parts))
        except FileNotFoundError:
            return []
        except OSError as e:
            logger.warning("Failed to read JSONL %s: %s", self._path(*parts), e)
            return []

        records: list[dict[str, Any]] = []
        for line in data.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in %s", self._path(*parts))
        return records

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

    def metric_keys(self) -> set[str]:
        """All metric keys seen so far (excluding 'step')."""
        return self._get_metrics().known_keys

    # ── Rollouts (typed, cached) ──────────────────────────────────────

    @staticmethod
    def _rollout_filename(base_name: str) -> str:
        return f"{base_name}_rollout_summaries.jsonl"

    def read_rollouts(self, iteration: int, base_name: str = "train") -> list[dict[str, Any]]:
        """Read rollout summaries for an iteration as raw dicts.

        Args:
            iteration: Training iteration number.
            base_name: Prefix for the JSONL file (e.g. ``"train"``,
                ``"eval_gsm8k"``). Matches the naming used by
                ``rollout_summaries_jsonl_path()`` in RL training.
        """
        filename = self._rollout_filename(base_name)
        cache_key = f"{iteration}/{filename}"
        if cache_key in self._rollout_cache:
            return self._rollout_cache[cache_key]

        records = self._read_jsonl(self._iter_dir(iteration), filename)
        self._rollout_cache[cache_key] = records
        return records

    def read_single_rollout(
        self,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        base_name: str = "train",
    ) -> dict[str, Any] | None:
        for r in self.read_rollouts(iteration, base_name):
            if r.get("group_idx") == group_idx and r.get("traj_idx") == traj_idx:
                return r
        return None

    # ── Checkpoints (typed) ───────────────────────────────────────────

    def read_checkpoints(self) -> list[dict[str, Any]]:
        """Read checkpoints.jsonl."""
        return self._read_jsonl("checkpoints.jsonl")

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

    def read_timing_flat(self) -> list[dict[str, Any]]:
        """All spans flattened with step annotation."""
        flat: list[dict[str, Any]] = []
        for record in self.read_timing():
            step = record.get("step", 0)
            if "spans" in record:
                for span in record["spans"]:
                    flat.append({"step": step, **span})
            else:
                flat.append(record)
        return flat

    def build_timing_tree(self, step: int) -> dict[str, Any]:
        """Build hierarchical span tree from time containment."""
        spans = self._get_spans_for_step(step)
        if not spans:
            return {"step": step, "root": None}

        sorted_spans = sorted(spans, key=lambda s: (s.get("wall_start", 0), -s.get("duration", 0)))

        nodes: list[dict[str, Any]] = []
        for s in sorted_spans:
            nodes.append(
                {
                    "name": s.get("name", "?"),
                    "duration": s.get("duration", 0),
                    "wall_start": s.get("wall_start", 0),
                    "wall_end": s.get("wall_end", 0),
                    "attributes": s.get("attributes", {}),
                    "children": [],
                }
            )

        EPS = 0.01
        root_children: list[dict[str, Any]] = []
        stack: list[dict[str, Any]] = []

        for node in nodes:
            while stack and stack[-1]["wall_end"] + EPS < node["wall_start"]:
                stack.pop()
            if stack and node["wall_end"] <= stack[-1]["wall_end"] + EPS:
                stack[-1]["children"].append(node)
            else:
                while stack and node["wall_end"] > stack[-1]["wall_end"] + EPS:
                    stack.pop()
                if stack:
                    stack[-1]["children"].append(node)
                else:
                    root_children.append(node)
            stack.append(node)

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

    def get_concurrency(self, step: int) -> dict[str, Any]:
        spans = self._get_spans_for_step(step)
        if not spans:
            return {"step": step, "spans": [], "max_concurrency": 0, "timeline": []}

        sorted_spans = sorted(spans, key=lambda s: s.get("wall_start", 0))
        events: list[tuple[float, int]] = []
        for s in sorted_spans:
            ws = s.get("wall_start", 0)
            we = s.get("wall_end", ws + s.get("duration", 0))
            events.append((ws, 1))
            events.append((we, -1))
        events.sort(key=lambda e: (e[0], e[1]))

        max_c = 0
        current = 0
        timeline: list[dict[str, Any]] = []
        for t, delta in events:
            current += delta
            max_c = max(max_c, current)
            timeline.append({"time": t, "concurrency": current})

        return {"step": step, "spans": sorted_spans, "max_concurrency": max_c, "timeline": timeline}

    def _get_spans_for_step(self, step: int) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        for record in self.read_timing():
            if record.get("step") != step:
                continue
            if "spans" in record:
                spans.extend(record["spans"])
            else:
                spans.append(record)
        return spans

    # ── Logtree ───────────────────────────────────────────────────────

    def read_logtree(self, iteration: int, base_name: str = "train") -> dict[str, Any] | None:
        return self._read_json(self._iter_dir(iteration), f"{base_name}_logtree.json")

    def list_logtrees(self, iteration: int) -> list[str]:
        items = self.storage.list_dir(self._path(self._iter_dir(iteration)))
        return sorted(n[: -len("_logtree.json")] for n in items if n.endswith("_logtree.json"))

    # ── Iterations ────────────────────────────────────────────────────

    def list_iterations(self) -> list[IterationInfo]:
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

    # ── Status detection ──────────────────────────────────────────────

    def detect_status(self) -> tuple[Status, float | None]:
        stat = self.storage.stat(self._path("metrics.jsonl"))
        if stat is None:
            return "idle", None
        age = time.time() - stat.mtime
        if age < _ACTIVE_THRESHOLD_SECONDS:
            return "running", stat.mtime
        for ckpt in reversed(self._read_jsonl("checkpoints.jsonl")):
            if ckpt.get("final"):
                return "completed", stat.mtime
        return "idle", stat.mtime

    def infer_training_type(self) -> TrainingType | None:
        config = self.read_config()
        if config is None:
            return None
        if "dpo_beta" in config:
            return "dpo"
        if "loss_fn" in config:
            return "rl"
        if "num_epochs" in config:
            return "sl"
        db = config.get("dataset_builder")
        if isinstance(db, dict):
            dt = db.get("__type__", "")
            if "RL" in dt:
                return "rl"
            if "Supervised" in dt or "SL" in dt:
                return "sl"
        return None

    # ── Writes ────────────────────────────────────────────────────────

    def _write_json(self, data: dict[str, Any], *parts: str) -> None:
        self.storage.write(self._path(*parts), json.dumps(data, indent=2).encode("utf-8"))

    def _append_jsonl(self, record: dict[str, Any], *parts: str) -> None:
        self.storage.append(self._path(*parts), (json.dumps(record) + "\n").encode("utf-8"))

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
        return await asyncio.to_thread(self.read_config)

    async def aread_metrics(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.read_metrics)

    async def aread_new_metrics(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.read_new_metrics)

    async def aread_rollouts(
        self, iteration: int, base_name: str = "train"
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.read_rollouts, iteration, base_name)

    async def aread_checkpoints(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.read_checkpoints)

    async def aread_timing(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.read_timing)

    async def aread_logtree(
        self, iteration: int, base_name: str = "train"
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.read_logtree, iteration, base_name)

    async def awrite_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        await asyncio.to_thread(self.write_metrics, metrics, step)

    async def awrite_checkpoint(self, record: dict[str, Any]) -> None:
        await asyncio.to_thread(self.write_checkpoint, record)
