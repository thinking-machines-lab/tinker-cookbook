"""Tests for capture scopes."""

import asyncio
from pathlib import Path

import pytest

from tinker_cookbook.capture.scope import capture, current_scope, replace_scope


def test_empty_scope_by_default() -> None:
    assert dict(current_scope()) == {}


def test_nesting_merges_and_restores() -> None:
    with capture(run_id="r1", iteration=1):
        assert dict(current_scope()) == {"run_id": "r1", "iteration": 1}
        with capture(iteration=2, purpose="eval"):
            assert dict(current_scope()) == {"run_id": "r1", "iteration": 2, "purpose": "eval"}
        assert dict(current_scope()) == {"run_id": "r1", "iteration": 1}
    assert dict(current_scope()) == {}


def test_scope_is_immutable() -> None:
    with capture(run_id="r1") as scope:
        with pytest.raises(TypeError):
            scope["run_id"] = "r2"  # type: ignore[index]


def test_non_scalar_value_rejected() -> None:
    with (
        pytest.raises(TypeError, match="scalar"),
        # The wrong value type is the point of the test (runtime validation).
        capture(bad=[1, 2]),  # pyright: ignore[reportArgumentType]
    ):
        pass


def test_decorator_form() -> None:
    @capture(purpose="eval")
    def inner() -> dict[str, object]:
        return dict(current_scope())

    assert inner() == {"purpose": "eval"}
    assert dict(current_scope()) == {}


@pytest.mark.asyncio
async def test_isolation_across_asyncio_tasks() -> None:
    seen: dict[str, dict[str, object]] = {}

    async def worker(name: str) -> None:
        with capture(traj_idx=name):
            await asyncio.sleep(0.01)
            seen[name] = dict(current_scope())

    with capture(run_id="r1"):
        await asyncio.gather(worker("a"), worker("b"))
        assert dict(current_scope()) == {"run_id": "r1"}

    assert seen["a"] == {"run_id": "r1", "traj_idx": "a"}
    assert seen["b"] == {"run_id": "r1", "traj_idx": "b"}


def test_replace_scope_does_not_merge() -> None:
    with capture(stale="yes", run_id="r1"):
        with replace_scope({"traj_idx": 3}):
            assert dict(current_scope()) == {"traj_idx": 3}  # no stale keys
        assert dict(current_scope()) == {"stale": "yes", "run_id": "r1"}


@pytest.mark.asyncio
async def test_decorator_on_async_function_scopes_the_body() -> None:
    """The scope must be active while the coroutine BODY runs, not merely
    while the coroutine object is created."""

    @capture(purpose="eval", iteration=1)
    async def inner() -> dict[str, object]:
        await asyncio.sleep(0.01)
        return dict(current_scope())

    # Create the coroutine outside any scope, await it later.
    coro = inner()
    assert dict(current_scope()) == {}
    assert await coro == {"purpose": "eval", "iteration": 1}
    assert dict(current_scope()) == {}


@pytest.mark.asyncio
async def test_shared_instance_across_overlapping_tasks() -> None:
    """One capture instance entered by overlapping tasks with out-of-order
    exits must not mix reset tokens across execution contexts."""
    cm = capture(shared="yes")
    a_entered = asyncio.Event()
    b_done = asyncio.Event()
    results: dict[str, object] = {}

    async def task_a() -> None:
        with cm:
            a_entered.set()
            await b_done.wait()  # A exits AFTER B entered and exited
            results["a_inside"] = dict(current_scope())
        results["a_after"] = dict(current_scope())

    async def task_b() -> None:
        await a_entered.wait()
        with cm:
            results["b_inside"] = dict(current_scope())
        b_done.set()

    await asyncio.gather(task_a(), task_b())
    assert results["a_inside"] == {"shared": "yes"}
    assert results["b_inside"] == {"shared": "yes"}
    assert results["a_after"] == {}
    assert dict(current_scope()) == {}


def test_shared_instance_across_threads() -> None:
    import threading

    cm = capture(shared="yes")
    seen: list[dict[str, object]] = []

    def worker() -> None:
        with cm:
            seen.append(dict(current_scope()))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == [{"shared": "yes"}] * 4
    assert dict(current_scope()) == {}


def test_readme_import_lines_are_valid() -> None:
    """Every `from tinker_cookbook... import ...` line in the README (the
    quickstart included) must actually work, so the docs cannot reference
    an API that does not exist at this point in history."""
    readme = Path(__file__).with_name("README.md").read_text()
    import_lines = [
        line.strip()
        for line in readme.splitlines()
        if line.strip().startswith("from tinker_cookbook") and " import " in line
    ]
    assert import_lines, "README quickstart should contain import lines"
    for line in import_lines:
        exec(line, {})  # doc-consistency check
