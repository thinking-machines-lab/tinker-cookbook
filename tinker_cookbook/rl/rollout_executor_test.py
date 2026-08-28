"""Tests for the rollout-executor fork-start-method guard."""

import multiprocessing
import sys
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import pytest

from tinker_cookbook.rl.rollouts import get_rollout_executor, set_rollout_executor


class _DummyExecutor(Executor):
    """Minimal Executor with no _mp_context attribute."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


@pytest.fixture(autouse=True)
def _reset_executor():
    yield
    set_rollout_executor(None)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork start method not available on this platform",
)
def test_rejects_fork_process_pool():
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("fork"))
    try:
        with pytest.raises(ValueError, match="spawn"):
            set_rollout_executor(pool)
        assert get_rollout_executor() is None
    finally:
        pool.shutdown(wait=False)


@pytest.mark.skipif(
    sys.platform == "linux",
    reason="default context is fork on Linux; covered by the explicit fork test",
)
def test_accepts_default_context_when_not_fork():
    pool = ProcessPoolExecutor(max_workers=1)
    try:
        set_rollout_executor(pool)
        assert get_rollout_executor() is pool
    finally:
        pool.shutdown(wait=False)


def test_accepts_spawn_process_pool():
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    try:
        set_rollout_executor(pool)
        assert get_rollout_executor() is pool
    finally:
        pool.shutdown(wait=False)


def test_accepts_thread_pool_and_none():
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        set_rollout_executor(pool)
        assert get_rollout_executor() is pool
    finally:
        pool.shutdown(wait=False)
    set_rollout_executor(None)
    assert get_rollout_executor() is None


def test_accepts_custom_executor_without_mp_context():
    executor = _DummyExecutor()
    set_rollout_executor(executor)
    assert get_rollout_executor() is executor
