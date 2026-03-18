"""
Tests for the cascading shutdown mechanism in async RL training.

These tests validate that when the dataloader exhausts its data, the shutdown
propagates cleanly through the pipeline without hanging:
  dataloader -> workers -> training loop -> evaluation loop
"""

from __future__ import annotations

import asyncio

from tinker_cookbook.rl.train import _AsyncCounter, _Shutdown


class TestAsyncCounter:
    def test_decrement_and_get(self):
        async def _test():
            counter = _AsyncCounter(3)
            assert await counter.decrement_and_get() == 2
            assert await counter.decrement_and_get() == 1
            assert await counter.decrement_and_get() == 0

        asyncio.run(_test())

    def test_concurrent_decrements(self):
        """Multiple concurrent decrements should each see a unique value."""

        async def _test():
            counter = _AsyncCounter(100)
            results = await asyncio.gather(*[counter.decrement_and_get() for _ in range(100)])
            # Each decrement should produce a unique value from 0 to 99
            assert sorted(results) == list(range(100))

        asyncio.run(_test())


class TestShutdownCascade:
    def test_dataloader_enqueues_shutdown_sentinels(self):
        """When the dataloader finishes, it should enqueue one _Shutdown per worker."""

        async def _test():
            num_workers = 4
            queue: asyncio.Queue[str | _Shutdown] = asyncio.Queue(maxsize=num_workers)

            for _ in range(num_workers):
                await queue.put(_Shutdown())

            for _ in range(num_workers):
                item = await queue.get()
                assert isinstance(item, _Shutdown)

            assert queue.empty()

        asyncio.run(_test())

    def test_last_worker_signals_training_loop(self):
        """The last worker to exit should enqueue a _Shutdown to the training queue."""

        async def _test():
            num_workers = 3
            counter = _AsyncCounter(num_workers)
            training_queue: asyncio.Queue[str | _Shutdown] = asyncio.Queue()

            for _ in range(num_workers):
                num_alive = await counter.decrement_and_get()
                if num_alive == 0:
                    training_queue.put_nowait(_Shutdown())

            assert training_queue.qsize() == 1
            assert isinstance(await training_queue.get(), _Shutdown)

        asyncio.run(_test())

    def test_full_cascade_no_hang(self):
        """
        Full integration test: wire up all four loops with mock rollouts and verify
        the entire pipeline shuts down cleanly without hanging.
        """

        async def _test():
            num_workers = 2
            num_batches = 2
            items_per_batch = 2

            env_queue: asyncio.Queue[int | _Shutdown] = asyncio.Queue(maxsize=num_workers)
            trajectory_queue: asyncio.Queue[int | _Shutdown | None] = asyncio.Queue()
            dataloader_done = asyncio.Event()
            eval_should_shutdown = asyncio.Event()
            worker_counter = _AsyncCounter(num_workers)
            sampling_updated = asyncio.Event()
            sampling_updated.set()

            loops_completed: list[str] = []

            async def dataloader_loop():
                for batch_idx in range(num_batches):
                    for item_idx in range(items_per_batch):
                        await env_queue.put(batch_idx * items_per_batch + item_idx)
                dataloader_done.set()
                for _ in range(num_workers):
                    await env_queue.put(_Shutdown())
                loops_completed.append("dataloader")

            async def worker_loop():
                while True:
                    item = await env_queue.get()
                    if isinstance(item, _Shutdown):
                        break
                    await asyncio.sleep(0.01)
                    trajectory_queue.put_nowait(item)
                num_alive = await worker_counter.decrement_and_get()
                if num_alive == 0:
                    trajectory_queue.put_nowait(_Shutdown())
                loops_completed.append("worker")

            async def training_loop():
                items_consumed = 0
                target = num_batches * items_per_batch
                while items_consumed < target:
                    item = await trajectory_queue.get()
                    if isinstance(item, _Shutdown):
                        break
                    if item is None:
                        continue
                    items_consumed += 1
                    sampling_updated.set()
                eval_should_shutdown.set()
                sampling_updated.set()
                loops_completed.append("training")

            async def evaluation_loop():
                while not eval_should_shutdown.is_set():
                    await sampling_updated.wait()
                    sampling_updated.clear()
                loops_completed.append("evaluation")

            await asyncio.wait_for(
                asyncio.gather(
                    dataloader_loop(),
                    *[worker_loop() for _ in range(num_workers)],
                    training_loop(),
                    evaluation_loop(),
                ),
                timeout=5.0,
            )

            assert "dataloader" in loops_completed
            assert loops_completed.count("worker") == num_workers
            assert "training" in loops_completed
            assert "evaluation" in loops_completed

        asyncio.run(_test())

    def test_cascade_with_early_shutdown(self):
        """
        When the dataloader has fewer items than the training loop expects,
        the _Shutdown sentinel should still propagate and prevent hanging.
        """

        async def _test():
            num_workers = 2
            num_dataloader_batches = 1
            items_per_batch = 2
            training_loop_target = 10  # Expects more than dataloader provides

            env_queue: asyncio.Queue[int | _Shutdown] = asyncio.Queue(maxsize=num_workers)
            trajectory_queue: asyncio.Queue[int | _Shutdown | None] = asyncio.Queue()
            eval_should_shutdown = asyncio.Event()
            worker_counter = _AsyncCounter(num_workers)
            sampling_updated = asyncio.Event()
            sampling_updated.set()

            async def dataloader_loop():
                for batch_idx in range(num_dataloader_batches):
                    for item_idx in range(items_per_batch):
                        await env_queue.put(batch_idx * items_per_batch + item_idx)
                for _ in range(num_workers):
                    await env_queue.put(_Shutdown())

            async def worker_loop():
                while True:
                    item = await env_queue.get()
                    if isinstance(item, _Shutdown):
                        break
                    trajectory_queue.put_nowait(item)
                num_alive = await worker_counter.decrement_and_get()
                if num_alive == 0:
                    trajectory_queue.put_nowait(_Shutdown())

            async def training_loop():
                i_batch = 0
                while i_batch < training_loop_target:
                    item = await trajectory_queue.get()
                    if isinstance(item, _Shutdown):
                        break
                    if item is None:
                        continue
                    i_batch += 1
                eval_should_shutdown.set()
                sampling_updated.set()

            async def evaluation_loop():
                while not eval_should_shutdown.is_set():
                    await sampling_updated.wait()
                    sampling_updated.clear()

            # Should not hang — shutdown cascade terminates all loops
            await asyncio.wait_for(
                asyncio.gather(
                    dataloader_loop(),
                    *[worker_loop() for _ in range(num_workers)],
                    training_loop(),
                    evaluation_loop(),
                ),
                timeout=5.0,
            )

        asyncio.run(_test())

    def test_requeue_skipped_during_shutdown(self):
        """
        When the dataloader is done, stale samples should be discarded
        rather than requeued (to avoid deadlocking on a full bounded queue).
        """
        dataloader_done = asyncio.Event()

        requeue_attempted = False
        discard_count = 0

        def filter_stale(is_stale: bool) -> bool:
            nonlocal requeue_attempted, discard_count
            if is_stale:
                if dataloader_done.is_set():
                    discard_count += 1
                else:
                    requeue_attempted = True
                return False
            return True

        # Before dataloader is done: stale items should attempt requeue
        filter_stale(is_stale=True)
        assert requeue_attempted

        # After dataloader is done: stale items should be discarded
        requeue_attempted = False
        dataloader_done.set()
        filter_stale(is_stale=True)
        assert not requeue_attempted
        assert discard_count == 1

    def test_none_items_pass_through_during_shutdown(self):
        """
        None items (failed rollouts) should be skipped, and _Shutdown should
        still be received even if preceded by None items.
        """

        async def _test():
            queue: asyncio.Queue[int | _Shutdown | None] = asyncio.Queue()

            queue.put_nowait(None)
            queue.put_nowait(None)
            queue.put_nowait(42)
            queue.put_nowait(None)
            queue.put_nowait(_Shutdown())

            received_items = []
            while True:
                item = await queue.get()
                if isinstance(item, _Shutdown):
                    break
                if item is None:
                    continue
                received_items.append(item)

            assert received_items == [42]

        asyncio.run(_test())
