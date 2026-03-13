"""Tests for tinker_cookbook.utils.profiling."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import threading
import time
from typing import Any

import pytest

from tinker_cookbook.utils.profiling import Profiler, profiled


# ---------------------------------------------------------------------------
# 1. Basic functionality: sync
# ---------------------------------------------------------------------------


class TestSyncFunctions:
    def test_basic_timing_and_return_value(self):
        prof = Profiler()

        @profiled("my_func")
        def do_work():
            time.sleep(0.01)
            return 42

        with prof.measure() as profile:
            result = do_work()

        assert result == 42
        assert profile["time/my_func"] >= 0.01
        assert "cpu/my_func" in profile

    def test_cpu_time_less_than_wall_time_during_sleep(self):
        """Sleep doesn't consume CPU — cpu/ should be much less than time/."""
        prof = Profiler()

        @profiled("sleeper")
        def sleeper():
            time.sleep(0.05)

        with prof.measure() as profile:
            sleeper()

        assert profile["time/sleeper"] >= 0.05
        assert profile["cpu/sleeper"] < 0.02  # CPU should be near zero

    def test_auto_key_from_qualname(self):
        prof = Profiler()

        @profiled
        def some_function():
            pass

        with prof.measure() as profile:
            some_function()

        keys = [k for k in profile if k.startswith("time/") and k != "time/total"]
        assert len(keys) == 1
        assert "some_function" in keys[0]

    def test_no_scope_degrades_gracefully(self):
        """Outside prof.measure(), just logs, no error."""

        @profiled("no_scope")
        def do_work():
            return 99

        assert do_work() == 99

    def test_return_value_preserved(self):
        prof = Profiler()

        @profiled("returns")
        def compute():
            return {"a": 1, "b": [2, 3]}

        with prof.measure():
            result = compute()

        assert result == {"a": 1, "b": [2, 3]}

    def test_exception_still_records(self):
        prof = Profiler()

        @profiled("failing")
        def fail():
            time.sleep(0.01)
            raise ValueError("boom")

        with prof.measure() as profile:
            with pytest.raises(ValueError, match="boom"):
                fail()

        assert profile["time/failing"] >= 0.01
        assert "cpu/failing" in profile


# ---------------------------------------------------------------------------
# 2. Basic functionality: async
# ---------------------------------------------------------------------------


class TestAsyncFunctions:
    def test_basic_async(self):
        prof = Profiler()

        @profiled("async_func")
        async def do_async_work():
            await asyncio.sleep(0.01)
            return "done"

        async def run():
            with prof.measure() as profile:
                result = await do_async_work()
            return result, profile

        result, profile = asyncio.run(run())
        assert result == "done"
        assert profile["time/async_func"] >= 0.01
        assert "cpu/async_func" in profile

    def test_async_cpu_less_than_wall_during_io_wait(self):
        """asyncio.sleep simulates I/O wait — CPU time should be near zero."""
        prof = Profiler()

        @profiled("async_sleeper")
        async def async_sleeper():
            await asyncio.sleep(0.05)

        async def run():
            with prof.measure() as profile:
                await async_sleeper()
            return profile

        profile = asyncio.run(run())
        assert profile["time/async_sleeper"] >= 0.05
        assert profile["cpu/async_sleeper"] < 0.02

    def test_async_exception_records(self):
        prof = Profiler()

        @profiled("async_fail")
        async def fail_async():
            await asyncio.sleep(0.01)
            raise RuntimeError("async boom")

        async def run():
            with prof.measure() as profile:
                with pytest.raises(RuntimeError, match="async boom"):
                    await fail_async()
            return profile

        profile = asyncio.run(run())
        assert "time/async_fail" in profile
        assert "cpu/async_fail" in profile

    def test_async_no_scope_degrades_gracefully(self):
        @profiled("async_no_scope")
        async def do_work():
            return "ok"

        result = asyncio.run(do_work())
        assert result == "ok"


# ---------------------------------------------------------------------------
# 3. Aggregation (repeated calls within one window)
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_single_call_no_suffix(self):
        prof = Profiler()

        @profiled("once")
        def work():
            pass

        with prof.measure() as profile:
            work()

        assert "time/once" in profile
        assert "cpu/once" in profile
        assert "time/once:total" not in profile

    def test_multiple_calls_emit_total_and_count(self):
        prof = Profiler()

        @profiled("repeated")
        def work():
            time.sleep(0.01)

        with prof.measure() as profile:
            for _ in range(3):
                work()

        assert profile["time/repeated:count"] == 3
        assert profile["time/repeated:total"] >= 0.03
        assert profile["cpu/repeated:count"] == 3
        assert "cpu/repeated:total" in profile
        assert "time/repeated" not in profile  # no suffix-less key

    def test_builtin_agg_applied_to_both_time_and_cpu(self):
        prof = Profiler()

        @profiled("agg_test", agg=["mean", "max", "min"])
        def work():
            time.sleep(0.01)

        with prof.measure() as profile:
            work()
            work()
            work()

        # time/ aggs
        assert "time/agg_test:mean" in profile
        assert "time/agg_test:max" in profile
        assert "time/agg_test:min" in profile
        # cpu/ aggs
        assert "cpu/agg_test:mean" in profile
        assert "cpu/agg_test:max" in profile
        assert "cpu/agg_test:min" in profile

    def test_custom_agg_lambda(self):
        prof = Profiler()

        @profiled("custom", agg=[("last", lambda durs: durs[-1])])
        def work():
            time.sleep(0.001)

        with prof.measure() as profile:
            work()
            work()
            work()

        assert "time/custom:last" in profile
        assert "cpu/custom:last" in profile
        assert profile["time/custom:count"] == 3

    def test_invalid_builtin_agg_raises_at_decoration_time(self):
        with pytest.raises(ValueError, match="Unknown built-in aggregation"):

            @profiled("bad_agg", agg=["nonexistent"])
            def work():
                pass

    def test_invalid_agg_tuple_raises_at_decoration_time(self):
        with pytest.raises(TypeError, match="Custom aggregation must be"):

            @profiled("bad_tuple", agg=[("name",)])  # type: ignore[list-item]
            def work():
                pass

    def test_invalid_agg_type_raises_at_decoration_time(self):
        with pytest.raises(TypeError, match="Aggregation spec must be"):

            @profiled("bad_type", agg=[123])  # type: ignore[list-item]
            def work():
                pass

    def test_agg_ignored_for_single_call(self):
        prof = Profiler()

        @profiled("once_with_agg", agg=["mean", "max"])
        def work():
            pass

        with prof.measure() as profile:
            work()

        assert "time/once_with_agg" in profile
        assert "time/once_with_agg:mean" not in profile
        assert "time/once_with_agg:total" not in profile

    def test_async_repeated_calls_aggregation(self):
        prof = Profiler()

        @profiled("async_repeated", agg=["mean"])
        async def async_work():
            await asyncio.sleep(0.01)

        async def run():
            with prof.measure() as profile:
                for _ in range(4):
                    await async_work()
            return profile

        profile = asyncio.run(run())
        assert profile["time/async_repeated:count"] == 4
        assert profile["time/async_repeated:total"] >= 0.04
        assert "time/async_repeated:mean" in profile
        assert "cpu/async_repeated:mean" in profile


# ---------------------------------------------------------------------------
# 4. prof.measure() behavior
# ---------------------------------------------------------------------------


class TestMeasure:
    def test_total_auto_recorded(self):
        prof = Profiler()

        with prof.measure() as profile:
            time.sleep(0.01)

        assert profile["time/total"] >= 0.01
        assert "cpu/total" in profile

    def test_profile_dict_only_contains_profiler_keys(self):
        """The profile dict should only have time/ and cpu/ keys."""
        prof = Profiler()

        @profiled("work")
        def work():
            pass

        with prof.measure() as profile:
            work()

        for key in profile:
            assert key.startswith("time/") or key.startswith("cpu/"), f"Unexpected key: {key}"

    def test_nested_measure_windows_are_independent(self):
        prof = Profiler()

        @profiled("inner_work")
        def inner():
            pass

        @profiled("outer_work")
        def outer():
            pass

        with prof.measure() as outer_profile:
            outer()
            with prof.measure() as inner_profile:
                inner()

        assert "time/outer_work" in outer_profile
        assert "time/inner_work" not in outer_profile
        assert "time/inner_work" in inner_profile
        assert "time/outer_work" not in inner_profile

    def test_empty_window_only_has_totals(self):
        prof = Profiler()

        with prof.measure() as profile:
            pass

        assert set(profile.keys()) == {"time/total", "cpu/total"}

    def test_sequential_windows_no_leakage(self):
        prof = Profiler()

        @profiled("work_a")
        def work_a():
            pass

        @profiled("work_b")
        def work_b():
            pass

        with prof.measure() as profile1:
            work_a()

        with prof.measure() as profile2:
            work_b()

        assert "time/work_a" in profile1
        assert "time/work_b" not in profile1
        assert "time/work_b" in profile2
        assert "time/work_a" not in profile2

    def test_multiple_different_functions_in_one_window(self):
        prof = Profiler()

        @profiled("func_a")
        def func_a():
            time.sleep(0.01)

        @profiled("func_b")
        def func_b():
            time.sleep(0.01)

        @profiled("func_c")
        async def func_c():
            await asyncio.sleep(0.01)

        async def run():
            with prof.measure() as profile:
                func_a()
                func_b()
                await func_c()
            return profile

        profile = asyncio.run(run())
        assert "time/func_a" in profile
        assert "time/func_b" in profile
        assert "time/func_c" in profile

    def test_window_cleans_up_after_body_exception(self):
        prof = Profiler()

        @profiled("before_error")
        def work():
            pass

        profile: dict[str, Any] = {}
        with pytest.raises(RuntimeError, match="user error"):
            with prof.measure() as profile:
                work()
                raise RuntimeError("user error")

        assert "time/before_error" in profile

    def test_separate_profiler_instances_are_independent(self):
        prof1 = Profiler()
        prof2 = Profiler()

        @profiled("shared_func")
        def work():
            pass

        with prof1.measure() as profile1:
            work()

        with prof2.measure() as profile2:
            work()

        # Both should capture independently
        assert "time/shared_func" in profile1
        assert "time/shared_func" in profile2


# ---------------------------------------------------------------------------
# 5. Decorator syntax variants
# ---------------------------------------------------------------------------


class TestDecoratorSyntax:
    def test_bare_decorator_no_parens(self):
        prof = Profiler()

        @profiled
        def my_func():
            return 1

        with prof.measure() as profile:
            assert my_func() == 1

        keys = [k for k in profile if k.startswith("time/") and k != "time/total"]
        assert len(keys) == 1
        assert "my_func" in keys[0]

    def test_decorator_with_positional_key(self):
        prof = Profiler()

        @profiled("custom_key")
        def my_func():
            return 2

        with prof.measure() as profile:
            assert my_func() == 2

        assert "time/custom_key" in profile

    def test_decorator_with_keyword_key(self):
        prof = Profiler()

        @profiled(key="kwarg_key")
        def my_func():
            return 3

        with prof.measure() as profile:
            assert my_func() == 3

        assert "time/kwarg_key" in profile

    def test_preserves_sync_function_metadata(self):
        @profiled("test")
        def my_documented_func():
            """My docstring."""
            pass

        assert my_documented_func.__name__ == "my_documented_func"
        assert my_documented_func.__doc__ == "My docstring."

    def test_preserves_async_function_metadata(self):
        @profiled("test")
        async def my_async_func():
            """Async docstring."""
            pass

        assert my_async_func.__name__ == "my_async_func"
        assert my_async_func.__doc__ == "Async docstring."

    def test_composable_with_other_decorators(self):
        prof = Profiler()

        def other_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        @other_decorator
        @profiled("composed")
        def work():
            return "ok"

        with prof.measure() as profile:
            result = work()

        assert result == "ok"
        assert "time/composed" in profile


# ---------------------------------------------------------------------------
# 6. Concurrency: asyncio.gather() and asyncio.create_task()
# ---------------------------------------------------------------------------


class TestAsyncConcurrency:
    def test_gather_bare_coroutines_share_window(self):
        prof = Profiler()

        @profiled("coro_work")
        async def coro_work():
            await asyncio.sleep(0.01)

        async def run():
            with prof.measure() as profile:
                await asyncio.gather(coro_work(), coro_work())
            return profile

        profile = asyncio.run(run())
        assert profile["time/coro_work:count"] == 2

    def test_create_task_children_share_window(self):
        prof = Profiler()

        @profiled("task_work")
        async def task_work():
            await asyncio.sleep(0.01)

        async def run():
            with prof.measure() as profile:
                t1 = asyncio.create_task(task_work())
                t2 = asyncio.create_task(task_work())
                await t1
                await t2
            return profile

        profile = asyncio.run(run())
        assert profile["time/task_work:count"] == 2

    def test_many_concurrent_tasks(self):
        prof = Profiler()

        @profiled("parallel")
        async def parallel_work(i: int):
            await asyncio.sleep(0.001 * (i % 5))

        async def run():
            with prof.measure() as profile:
                tasks = [asyncio.create_task(parallel_work(i)) for i in range(50)]
                await asyncio.gather(*tasks)
            return profile

        profile = asyncio.run(run())
        assert profile["time/parallel:count"] == 50

    def test_concurrent_tasks_with_independent_windows(self):
        """Models RL do_async_training: training_loop and eval_loop
        each with their own prof.measure() window."""
        prof = Profiler()
        training_profiles = {}
        eval_profiles = {}

        @profiled("train_step")
        async def train_step():
            await asyncio.sleep(0.01)

        @profiled("eval_step")
        async def eval_step():
            await asyncio.sleep(0.01)

        async def training_loop():
            for i in range(3):
                with prof.measure() as profile:
                    await train_step()
                training_profiles[i] = dict(profile)

        async def eval_loop():
            for i in range(2):
                with prof.measure() as profile:
                    await eval_step()
                eval_profiles[i] = dict(profile)

        async def run():
            await asyncio.gather(
                asyncio.create_task(training_loop(), name="training_loop"),
                asyncio.create_task(eval_loop(), name="eval_loop"),
            )

        asyncio.run(run())

        for m in training_profiles.values():
            assert "time/train_step" in m
            assert "time/eval_step" not in m

        for m in eval_profiles.values():
            assert "time/eval_step" in m
            assert "time/train_step" not in m

    def test_window_exit_before_child_task_completes(self):
        prof = Profiler()
        child_recorded = asyncio.Event()
        parent_exited = asyncio.Event()

        @profiled("late_work")
        async def late_child():
            await parent_exited.wait()
            await asyncio.sleep(0.01)
            child_recorded.set()

        async def run():
            with prof.measure() as profile:
                task = asyncio.create_task(late_child())
            parent_exited.set()
            await child_recorded.wait()
            await task
            return profile

        profile = asyncio.run(run())
        assert "time/late_work" not in profile


# ---------------------------------------------------------------------------
# 7. Concurrency: threading.Thread and ThreadPoolExecutor
# ---------------------------------------------------------------------------


class TestThreadIsolation:
    def test_child_threads_do_not_see_parent_window(self):
        prof = Profiler()
        child_saw_scope = {}

        @profiled("thread_work")
        def thread_func(thread_id: int):
            time.sleep(0.01)
            child_saw_scope[thread_id] = _scope_state_is_active()

        with prof.measure() as profile:
            threads = [threading.Thread(target=thread_func, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert all(not v for v in child_saw_scope.values())
        assert "time/thread_work" not in profile

    def test_thread_pool_executor_isolated(self):
        prof = Profiler()

        @profiled("pool_work")
        def pool_func():
            time.sleep(0.01)
            return _scope_state_is_active()

        with prof.measure() as profile:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(pool_func) for _ in range(4)]
                results = [f.result() for f in futures]

        assert all(not r for r in results)
        assert "time/pool_work" not in profile


# ---------------------------------------------------------------------------
# 8. Concurrency: asyncio.to_thread()
# ---------------------------------------------------------------------------


class TestToThread:
    def test_to_thread_records_into_parent_window(self):
        prof = Profiler()

        @profiled("to_thread_work")
        def blocking_work():
            time.sleep(0.01)
            return 42

        async def run():
            with prof.measure() as profile:
                result = await asyncio.to_thread(blocking_work)
            return result, profile

        result, profile = asyncio.run(run())
        assert result == 42
        assert "time/to_thread_work" in profile
        assert "cpu/to_thread_work" in profile

    def test_to_thread_concurrent_with_async_tasks(self):
        prof = Profiler()

        @profiled("bg_work")
        def blocking_work():
            time.sleep(0.05)

        @profiled("async_work")
        async def async_work():
            await asyncio.sleep(0.01)

        async def run():
            with prof.measure() as profile:
                await asyncio.gather(
                    asyncio.to_thread(blocking_work),
                    async_work(),
                    async_work(),
                    async_work(),
                )
            return profile

        profile = asyncio.run(run())
        assert "time/bg_work" in profile
        assert profile["time/async_work:count"] == 3

    def test_to_thread_same_key_from_multiple_threads(self):
        prof = Profiler()

        @profiled("shared_key")
        def blocking_work():
            time.sleep(0.01)

        async def run():
            with prof.measure() as profile:
                await asyncio.gather(*[asyncio.to_thread(blocking_work) for _ in range(10)])
            return profile

        profile = asyncio.run(run())
        assert profile["time/shared_key:count"] == 10

    def test_to_thread_different_keys_from_multiple_threads(self):
        prof = Profiler()

        def make_func(name: str):
            @profiled(name)
            def work():
                time.sleep(0.01)

            return work

        funcs = [make_func(f"key_{i}") for i in range(10)]

        async def run():
            with prof.measure() as profile:
                await asyncio.gather(*[asyncio.to_thread(f) for f in funcs])
            return profile

        profile = asyncio.run(run())
        for i in range(10):
            assert f"time/key_{i}" in profile
            assert f"cpu/key_{i}" in profile

    def test_to_thread_and_event_loop_same_key(self):
        prof = Profiler()

        @profiled("dual_key")
        def sync_work():
            time.sleep(0.01)

        @profiled("dual_key")
        async def async_work():
            await asyncio.sleep(0.01)

        async def run():
            with prof.measure() as profile:
                await asyncio.gather(
                    asyncio.to_thread(sync_work),
                    asyncio.to_thread(sync_work),
                    async_work(),
                    async_work(),
                    async_work(),
                )
            return profile

        profile = asyncio.run(run())
        assert profile["time/dual_key:count"] == 5

    def test_to_thread_stress(self):
        prof = Profiler()

        @profiled("blocking")
        def blocking_work():
            time.sleep(0.001)

        @profiled("async_io")
        async def async_io():
            await asyncio.sleep(0.001)

        async def run():
            with prof.measure() as profile:
                thread_coros = [asyncio.to_thread(blocking_work) for _ in range(20)]
                async_coros = [async_io() for _ in range(20)]
                await asyncio.gather(*thread_coros, *async_coros)
            return profile

        profile = asyncio.run(run())
        assert profile["time/blocking:count"] == 20
        assert profile["time/async_io:count"] == 20

    def test_nested_profiled_no_deadlock(self):
        prof = Profiler()

        @profiled("inner")
        def inner_work():
            time.sleep(0.01)

        @profiled("outer")
        def outer_work():
            inner_work()

        with prof.measure() as profile:
            outer_work()

        assert "time/inner" in profile
        assert "time/outer" in profile
        assert profile["time/outer"] >= profile["time/inner"]

    def test_nested_profiled_in_to_thread_no_deadlock(self):
        prof = Profiler()

        @profiled("inner_bg")
        def inner_work():
            time.sleep(0.01)

        @profiled("outer_bg")
        def outer_work():
            inner_work()

        async def run():
            with prof.measure() as profile:
                await asyncio.to_thread(outer_work)
            return profile

        profile = asyncio.run(run())
        assert "time/inner_bg" in profile
        assert "time/outer_bg" in profile


# ---------------------------------------------------------------------------
# 9. Phase nesting
# ---------------------------------------------------------------------------


class TestPhase:
    def test_phase_records_own_duration(self):
        prof = Profiler()

        with prof.measure() as profile:
            with prof.phase("sampling"):
                time.sleep(0.01)

        assert profile["time/sampling"] >= 0.01
        assert "cpu/sampling" in profile

    def test_phase_prefixes_profiled_keys(self):
        prof = Profiler()

        @profiled("work")
        def work():
            pass

        with prof.measure() as profile:
            with prof.phase("train"):
                work()

        assert "time/train/work" in profile
        assert "cpu/train/work" in profile
        assert "time/work" not in profile  # not at top level

    def test_phase_with_aggregation(self):
        prof = Profiler()

        @profiled("rollout", agg=["mean", "max"])
        def do_rollout():
            time.sleep(0.001)

        with prof.measure() as profile:
            with prof.phase("sampling"):
                for _ in range(5):
                    do_rollout()

        assert profile["time/sampling/rollout:count"] == 5
        assert "time/sampling/rollout:mean" in profile
        assert "time/sampling/rollout:max" in profile
        assert profile["time/sampling"] >= 0.005  # phase own duration

    def test_multiple_phases(self):
        prof = Profiler()

        @profiled("step_a")
        def step_a():
            time.sleep(0.01)

        @profiled("step_b")
        def step_b():
            time.sleep(0.01)

        with prof.measure() as profile:
            with prof.phase("eval"):
                step_a()
            with prof.phase("train"):
                step_b()

        assert "time/eval" in profile
        assert "time/eval/step_a" in profile
        assert "time/train" in profile
        assert "time/train/step_b" in profile
        # No cross-contamination
        assert "time/eval/step_b" not in profile
        assert "time/train/step_a" not in profile

    def test_nested_phases(self):
        prof = Profiler()

        @profiled("inner_work")
        def inner_work():
            pass

        with prof.measure() as profile:
            with prof.phase("outer"):
                with prof.phase("inner"):
                    inner_work()

        assert "time/outer" in profile
        assert "time/outer/inner" in profile
        assert "time/outer/inner/inner_work" in profile

    def test_phase_noop_outside_measure(self):
        """phase() outside measure() should be a no-op, not error."""
        prof = Profiler()

        with prof.phase("orphan"):
            pass  # should not raise

    def test_phase_with_async(self):
        prof = Profiler()

        @profiled("async_rollout", agg=["mean"])
        async def async_rollout():
            await asyncio.sleep(0.01)

        async def run():
            with prof.measure() as profile:
                with prof.phase("sampling"):
                    await asyncio.gather(
                        async_rollout(),
                        async_rollout(),
                        async_rollout(),
                    )
            return profile

        profile = asyncio.run(run())
        assert "time/sampling" in profile
        assert profile["time/sampling/async_rollout:count"] == 3
        assert "time/sampling/async_rollout:mean" in profile

    def test_profiled_outside_phase_stays_at_top_level(self):
        """Functions called outside any phase should be at the top level."""
        prof = Profiler()

        @profiled("top_level")
        def top_level():
            pass

        @profiled("nested")
        def nested():
            pass

        with prof.measure() as profile:
            top_level()
            with prof.phase("inner"):
                nested()

        assert "time/top_level" in profile
        assert "time/inner/nested" in profile
        assert "time/nested" not in profile


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _scope_state_is_active() -> bool:
    """Check if there's an active profiler scope (for thread isolation tests)."""
    from tinker_cookbook.utils.profiling import _scope_state

    return _scope_state.get() is not None
