"""Memory tracing helpers for the rendering / data-prep phase.

We trace memory along three axes that the kernel exposes separately:

  * **cgroup.current / cgroup.max** — what the K8s OOM killer cares about.
  * **main process RSS** — what the orchestrator's Python heap holds.
  * **worker RSS** (sum of children) — what the multiprocessing pool holds.

Anything cgroup sees that is not main+workers is bucketed as
"unaccounted", which in practice equals OS page cache (e.g. dirty pages
from writes to a ``DiskBackedDatumStore``). Page cache is reclaimable and
will not cause OOM as long as the workload running above it is not also
allocating committed RAM faster than reclaim can keep up.

This module was extracted from ``sft_loop`` after the orchestrator OOM
investigation; see docs/engineering/sft-v2-orchestrator-oom-debug.md for
the story.
"""

from __future__ import annotations

import glob
import logging
import os
import resource as _resource
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level readers (cgroup v1 + v2, /proc/<pid>/status)
# ---------------------------------------------------------------------------


def read_cgroup_mem() -> int:
    """Return current cgroup memory usage in bytes (0 when not in a cgroup)."""
    for path in (
        "/sys/fs/cgroup/memory.current",                     # cgroup v2
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",       # cgroup v1
    ):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            continue
    return 0


def read_cgroup_limit() -> int:
    """Return cgroup memory hard limit in bytes (0 when unlimited / unknown)."""
    for path in (
        "/sys/fs/cgroup/memory.max",                          # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",        # cgroup v1
    ):
        try:
            with open(path) as f:
                val = f.read().strip()
                if val == "max":
                    return 0
                return int(val)
        except (FileNotFoundError, ValueError):
            continue
    return 0


def proc_vmrss_gi(status_path: str) -> float:
    """Return VmRSS for a /proc/<pid>/status path, in GiB."""
    try:
        with open(status_path) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)  # KiB -> GiB
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return 0.0


def worker_rss_info(main_pid: int) -> Tuple[List[Tuple[int, float]], float]:
    """Return ``(per_worker, total_gi)`` for child processes of ``main_pid``.

    Walks ``/proc/*/status``, filters processes whose ``PPid`` is ``main_pid``,
    and reports their ``VmRSS`` in GiB.
    """
    workers: List[Tuple[int, float]] = []
    for status_path in glob.glob("/proc/*/status"):
        try:
            pid_str = status_path.split("/")[2]
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == main_pid:
                continue
            with open(status_path) as f:
                content = f.read()
            ppid = 0
            rss_kib = 0
            for line in content.split("\n"):
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss_kib = int(line.split()[1])
            if ppid == main_pid and rss_kib > 0:
                workers.append((pid, rss_kib / (1024 * 1024)))
        except (FileNotFoundError, ProcessLookupError, PermissionError,
                ValueError, IndexError):
            continue
    workers.sort()
    return workers, sum(r for _, r in workers)


# ---------------------------------------------------------------------------
# High-level tracer
# ---------------------------------------------------------------------------


class MemTracer:
    """Logs ``[mem] <stage> i/total | cgroup=... | main=... | workers=...``.

    Construct once before the rendering loop and call :py:meth:`log` at
    checkpoints (e.g. every 1% of progress). Pass an optional
    ``store_callback`` returning ``(item_count, disk_bytes)`` to attach
    ``DiskBackedDatumStore`` stats to each log line.
    """

    def __init__(
        self,
        *,
        main_pid: int | None = None,
        store_callback: Callable[[], Tuple[int, int]] | None = None,
        log: logging.Logger | None = None,
    ):
        self._main_pid = main_pid if main_pid is not None else os.getpid()
        self._store_callback = store_callback
        self._log = log or logger

    def log(self, stage: str, i: int, total: int, *, pool: object | None = None) -> None:
        peak_kib = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        peak_gi = peak_kib / (1024 ** 2)  # ru_maxrss is KiB on Linux
        main_cur_gi = proc_vmrss_gi(f"/proc/{self._main_pid}/status")
        cgroup_gi = read_cgroup_mem() / (1024 ** 3)
        limit_bytes = read_cgroup_limit()
        limit_gi = limit_bytes / (1024 ** 3) if limit_bytes else 0.0

        workers, workers_total = worker_rss_info(self._main_pid)
        workers_str = ",".join(f"{pid}:{gi:.1f}" for pid, gi in workers)

        pool_info = ""
        if pool is not None:
            try:
                cache_size = len(pool._cache)  # type: ignore[attr-defined]
                pool_info = f" | pool_cache={cache_size}"
            except Exception:
                pass

        unaccounted = cgroup_gi - main_cur_gi - workers_total

        store_info = ""
        if self._store_callback is not None:
            try:
                rows, disk_bytes = self._store_callback()
                store_info = f" | store={rows} rows / {disk_bytes / (1024 ** 3):.1f} GiB disk"
            except Exception:
                pass

        pct = 100.0 * i / total if total else 0.0
        self._log.info(
            "[mem] %s %d/%d (%.0f%%) | cgroup=%.1f/%.1f GiB | main=%.1f cur / %.1f peak GiB"
            " | workers=%.1f GiB [%s] | unaccounted=%.1f GiB%s%s",
            stage, i, total, pct,
            cgroup_gi, limit_gi, main_cur_gi, peak_gi,
            workers_total, workers_str, unaccounted, pool_info, store_info,
        )
