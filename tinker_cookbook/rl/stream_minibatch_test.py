"""Stream-minibatch workers must never die without unblocking the trainer.

``do_sync_training_with_stream_minibatch`` fires one task per env group and
then ``await``s ``trajectory_groups_queue.get()`` until a minibatch is full.
If a worker raises (FailFast, sandbox crash, etc.) without putting anything,
that get hangs for the rest of the run. Putting ``None`` is not enough: it is
the filtered-group signal, so FailFast would be swallowed (and on the async
path, which skips None, the trainer still hangs).
"""

from __future__ import annotations

import asyncio

import pytest

from tinker_cookbook.rl.train import _raise_if_worker_failed, _WorkerFailed


class TestRaiseIfWorkerFailed:
    def test_passes_through_normal_items(self):
        assert _raise_if_worker_failed(None) is None
        assert _raise_if_worker_failed("ok") == "ok"

    def test_reraises_original_exception(self):
        with pytest.raises(RuntimeError, match="rollout exploded"):
            _raise_if_worker_failed(_WorkerFailed(RuntimeError("rollout exploded")))


class TestWorkerCrashUnblocksConsumer:
    def test_worker_that_never_puts_hangs_the_consumer(self):
        """Pre-fix pattern: fire-and-forget raise, trainer blocked on get()."""

        async def _test() -> None:
            queue: asyncio.Queue[object] = asyncio.Queue()

            async def worker() -> None:
                raise RuntimeError("rollout exploded")

            async def consumer() -> object:
                return await queue.get()

            worker_task = asyncio.create_task(worker())
            consumer_task = asyncio.create_task(consumer())
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(consumer_task), timeout=0.05)
            worker_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(worker_task, consumer_task, return_exceptions=True)

        asyncio.run(_test())

    def test_none_sentinel_is_skipped_so_failfast_would_still_hang(self):
        """Putting None matches filtered groups; async training skips it."""

        async def _test() -> None:
            queue: asyncio.Queue[object] = asyncio.Queue()
            needed = 2

            async def worker() -> None:
                queue.put_nowait(None)

            async def consumer() -> None:
                got = 0
                while got < needed:
                    item = await queue.get()
                    if item is None:
                        continue
                    got += 1

            await asyncio.create_task(worker())
            consumer_task = asyncio.create_task(consumer())
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(consumer_task), timeout=0.05)
            consumer_task.cancel()
            await asyncio.gather(consumer_task, return_exceptions=True)

        asyncio.run(_test())

    def test_worker_failed_sentinel_unblocks_and_reraises(self):
        """The fix: enqueue _WorkerFailed so get() returns and the trainer crashes."""

        async def _test() -> None:
            queue: asyncio.Queue[object] = asyncio.Queue()

            async def worker() -> None:
                try:
                    raise RuntimeError("rollout exploded")
                except Exception as exc:
                    queue.put_nowait(_WorkerFailed(exc))
                    return

            worker_task = asyncio.create_task(worker())
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            with pytest.raises(RuntimeError, match="rollout exploded"):
                _raise_if_worker_failed(item)
            await worker_task

        asyncio.run(_test())

    def test_streaming_consumer_reraises_instead_of_hanging(self):
        """Mirror do_train_step_streaming: skip None, unwrap _WorkerFailed."""

        async def _test() -> None:
            queue: asyncio.Queue[object] = asyncio.Queue()

            async def worker() -> None:
                try:
                    raise RuntimeError("FailFast")
                except Exception as exc:
                    queue.put_nowait(_WorkerFailed(exc))
                    return

            async def consumer() -> None:
                while True:
                    item = _raise_if_worker_failed(await queue.get())
                    if item is None:
                        continue
                    return

            worker_task = asyncio.create_task(worker())
            with pytest.raises(RuntimeError, match="FailFast"):
                await asyncio.wait_for(consumer(), timeout=1.0)
            await worker_task

        asyncio.run(_test())
