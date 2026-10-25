import asyncio

import pytest

from tinker_cookbook.sandbox.modal_sandbox import _cancellation_as_terminated
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
