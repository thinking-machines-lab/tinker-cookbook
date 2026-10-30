import asyncio
from pathlib import Path

import pytest

from tinker_cookbook.sandbox.daytona_sandbox import (
    _cancellation_as_terminated,
    _task_resources,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxTerminatedError


@_cancellation_as_terminated
async def _cancelled_inside() -> None:
    raise asyncio.CancelledError


def test_inner_cancellation_becomes_sandbox_terminated() -> None:
    with pytest.raises(SandboxTerminatedError):
        asyncio.run(_cancelled_inside())


def test_outer_cancellation_propagates() -> None:
    @_cancellation_as_terminated
    async def slow() -> None:
        await asyncio.sleep(10)

    async def main() -> None:
        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())


def test_task_resources_from_legacy_sizes(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text('[environment]\ncpus = 2\nmemory = "4G"\nstorage = "10240M"\n')
    resources = _task_resources(task_toml)
    assert resources is not None
    assert (resources.cpu, resources.memory, resources.disk) == (2, 4, 10)


def test_task_resources_round_up_to_whole_units(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text("[environment]\ncpus = 0.5\nmemory_mb = 1536\n")
    resources = _task_resources(task_toml)
    assert resources is not None
    assert (resources.cpu, resources.memory, resources.disk) == (1, 2, None)


def test_task_resources_absent(tmp_path: Path) -> None:
    assert _task_resources(tmp_path / "missing.toml") is None
    task_toml = tmp_path / "task.toml"
    task_toml.write_text('version = "1.0"\n')
    assert _task_resources(task_toml) is None
