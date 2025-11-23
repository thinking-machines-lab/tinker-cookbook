import asyncio
import json
import tempfile
import threading

from tinker_cookbook.utils.trace import (
    get_scope_context,
    scope,
    update_scope_context,
    trace_init,
    trace_shutdown,
)


@scope
async def foo():
    await asyncio.sleep(0.1)
    context = get_scope_context()
    context.attributes["foo"] = "foo"
    context.attributes["foo2"] = 1
    await bar()


@scope
async def bar():
    await asyncio.sleep(0.05)
    context = get_scope_context()
    context.attributes["bar"] = 1
    await baz()


@scope
def ced():
    pass


@scope
async def baz():
    await asyncio.sleep(0.02)
    update_scope_context({"baz": "baz"})
    ced()


@scope
async def coroutine1():
    await foo()
    await asyncio.sleep(0.05)


@scope
async def coroutine2():
    await asyncio.sleep(0.15)
    await foo()


@scope
def sync_func():
    pass


@scope
async def work(thread_name: str):
    task1 = asyncio.create_task(coroutine1(), name=f"{thread_name}-coroutine-1")
    task2 = asyncio.create_task(coroutine2(), name=f"{thread_name}-coroutine-2")
    sync_func()
    await asyncio.gather(task1, task2)


@scope
async def example_program():
    @scope
    def thread_target():
        asyncio.run(work("secondary_thread"))

    thread = threading.Thread(target=thread_target, name="secondary_thread")
    thread.start()

    await work("main_thread")

    thread.join()


def test_trace():
    with tempfile.NamedTemporaryFile(
        "w+", suffix=".jsonl", prefix="test_events", delete=True
    ) as temp_file:
        trace_init(output_file=temp_file.name)
        asyncio.run(example_program())
        trace_shutdown()

        with open(temp_file.name, "r") as f:
            events = [json.loads(line) for line in f]

        # There should be 2 process metadata events
        num_metadata_pid_events = sum(
            1 for event in events if event["ph"] == "M" and event["tid"] == 0
        )
        assert num_metadata_pid_events == 2
        num_unique_pids = len(set(event["pid"] for event in events if event["ph"] != "M"))
        assert num_unique_pids == 2

        # main thread has 3: main, coroutine-1, coroutine-2
        # secondary thread has 4: thread_target, work, coroutine-1, coroutine-2
        num_metadata_tid_events = sum(
            1 for event in events if event["ph"] == "M" and event["tid"] != 0
        )
        assert num_metadata_tid_events == 7
        num_unique_tids = len(set(event["tid"] for event in events if event["ph"] != "M"))
        assert num_unique_tids == 7

        # Validate that attributes are set correctly
        for event in events:
            if event["ph"] != "E":
                continue
            if event["name"] == "foo":
                assert event["args"]["foo"] == "foo"
                assert event["args"]["foo2"] == 1
            if event["name"] == "bar":
                assert event["args"]["bar"] == 1
            if event["name"] == "baz":
                assert event["args"]["baz"] == "baz"


if __name__ == "__main__":
    trace_init()
    asyncio.run(example_program())
    trace_shutdown()
