"""Tests for automatic executor scope propagation."""

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tinker_cookbook.capture.exporter import CaptureExporter
from tinker_cookbook.capture.instrument import instrument_tinker, uninstrument_tinker
from tinker_cookbook.capture.propagate import instrument_threads, uninstrument_threads
from tinker_cookbook.capture.scope import capture, current_scope


class _Sink:
    def export(self, records: object, timeout: float | None = None) -> None:
        del records, timeout


def _read_scope() -> dict[str, Any]:
    return dict(current_scope())


@pytest.fixture
def auto_propagate() -> Iterator[None]:
    instrument_threads()
    try:
        yield
    finally:
        uninstrument_threads()


@pytest.mark.asyncio
async def test_run_in_executor_propagates_when_instrumented(auto_propagate: None) -> None:
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as pool, capture(run_id="r1", iteration=2):
        result = await loop.run_in_executor(pool, _read_scope)
    assert result == {"run_id": "r1", "iteration": 2}


@pytest.mark.asyncio
async def test_default_executor_propagates_when_instrumented(auto_propagate: None) -> None:
    loop = asyncio.get_running_loop()
    with capture(run_id="r1"):
        result = await loop.run_in_executor(None, _read_scope)
    assert result == {"run_id": "r1"}


def test_thread_start_propagates_when_instrumented(auto_propagate: None) -> None:
    seen: list[dict[str, object]] = []

    def worker() -> None:
        seen.append(dict(current_scope()))

    with capture(run_id="r1", traj_idx=3):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert seen == [{"run_id": "r1", "traj_idx": 3}]


def test_timer_propagates_when_instrumented(auto_propagate: None) -> None:
    seen: list[dict[str, object]] = []
    with capture(run_id="r1"):
        timer = threading.Timer(0, lambda: seen.append(dict(current_scope())))
        timer.start()
        timer.join()
    assert seen == [{"run_id": "r1"}]


def test_thread_pool_submit_kwargs_propagate(auto_propagate: None) -> None:
    def read_with(value: int) -> tuple[dict[str, Any], int]:
        return dict(current_scope()), value

    with ThreadPoolExecutor(max_workers=1) as pool, capture(run_id="r1"):
        result = pool.submit(read_with, value=7).result()
    assert result == ({"run_id": "r1"}, 7)


@pytest.mark.asyncio
async def test_instrument_tinker_does_not_patch_threads() -> None:
    """Thread propagation is deliberately decoupled from SDK instrumentation:
    it is owned by this module, and the store integration (capture_to_store)
    turns it on by default with nesting-aware restore. Direct
    instrument_tinker users opt in via instrument_threads()."""
    exporter = CaptureExporter(_Sink())
    original_start = threading.Thread.start
    loop = asyncio.get_running_loop()
    instrument_tinker(exporter)
    try:
        assert threading.Thread.start is original_start  # threads untouched
        with ThreadPoolExecutor(max_workers=1) as pool, capture(run_id="via-tinker"):
            result = await loop.run_in_executor(pool, _read_scope)
        assert result == {}  # no propagation without instrument_threads()
    finally:
        uninstrument_tinker()
        exporter.shutdown()


@pytest.mark.asyncio
async def test_plain_run_in_executor_still_loses_scope_without_instrumentation() -> None:
    """Without executor instrumentation, the stdlib gap remains."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as pool, capture(run_id="r1"):
        result = await loop.run_in_executor(pool, _read_scope)
    assert result == {}
