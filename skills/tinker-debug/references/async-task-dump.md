# Async Task Dump Diagnostic

Ready-to-paste diagnostic code for investigating hung or slow training sessions. This instruments both the main asyncio event loop and the Tinker SDK's internal event loop to reveal what's blocking.

## What this does

1. **Periodic task dump** — Every second, prints all active asyncio tasks in both the main event loop and the SDK's internal `InternalClientHolder` thread
2. **Thread stack watchdog** — Every second, prints Python stack traces for the main and SDK threads, so you can see exactly where they're blocked even if it's synchronous code
3. **aiomonitor** — Opens telnet/web ports for interactive async debugging (optional, requires `pip install aiomonitor`)

## When to use

- A training step has been running for much longer than expected
- The training loop appears completely hung (no log output)
- You've confirmed with pyinstrument that time is in `epoll.poll` but need to know what the SDK thread is doing

## Setup

```bash
pip install aiomonitor
```

## The diagnostic code

Paste this at the end of the user's script, replacing their `asyncio.run(main(config))` call:

```python
import asyncio

# ── Configuration ──────────────────────────────────────────────────
AIOMONITOR_HOST = "127.0.0.1"
HOLDER_AIOMONITOR = {"port": 20101, "webui_port": 20102, "console_port": 20103}
MAIN_AIOMONITOR = {"port": 21101, "webui_port": 21102, "console_port": 21103}
TASK_DUMP_INTERVAL_SEC = 1.0

# ── Periodic async task dump ──────────────────────────────────────
async def _periodic_dump_tasks(loop, label):
    import time
    while True:
        await asyncio.sleep(TASK_DUMP_INTERVAL_SEC)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        tasks = asyncio.all_tasks(loop)
        print(
            f"\n===== {ts} asyncio task dump [{label}] "
            f"{len(tasks)} task(s) @ interval={TASK_DUMP_INTERVAL_SEC}s ====="
        )
        for t in sorted(tasks, key=lambda x: (x.get_name(), id(x))):
            coro = t.get_coro()
            coro_s = repr(coro) if coro is not None else "None"
            if len(coro_s) > 240:
                coro_s = coro_s[:237] + "..."
            if t.cancelled():
                state = "cancelled"
            elif t.done():
                try:
                    ex = t.exception()
                except asyncio.CancelledError:
                    state = "cancelled"
                else:
                    state = "done" if ex is None else f"done exc={type(ex).__name__}:{ex}"
            else:
                state = "pending"
            print(f"  {t.get_name()!r}: {state}  {coro_s}")
        print(f"===== {ts} end [{label}] =====\n", flush=True)

# ── Thread stack watchdog ─────────────────────────────────────────
_WATCHDOG_INTERVAL_SEC = 1.0
_watchdog_started = False
_watchdog_lock = None
_watchdog_targets = {}

def _watchdog_register_this_thread(label):
    import threading
    global _watchdog_lock, _watchdog_targets
    if _watchdog_lock is None:
        _watchdog_lock = threading.Lock()
    t = threading.current_thread()
    with _watchdog_lock:
        _watchdog_targets[t.ident] = label

def _ensure_thread_stack_watchdog():
    import sys, threading, time, traceback
    global _watchdog_started, _watchdog_lock
    if _watchdog_lock is None:
        _watchdog_lock = threading.Lock()

    def watchdog_loop():
        while True:
            time.sleep(_WATCHDOG_INTERVAL_SEC)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            buf = [
                f"\n{'#' * 72}",
                f"# {ts} stacks: main + holder threads only",
                f"{'#' * 72}\n",
            ]
            with _watchdog_lock:
                targets = dict(_watchdog_targets)
            if not targets:
                buf.append("(no threads registered yet)\n")
                print("\n".join(buf), flush=True)
                continue
            frames = sys._current_frames()
            for ident, label in sorted(targets.items(), key=lambda kv: kv[1]):
                fr = frames.get(ident)
                if fr is None:
                    buf.append(f"--- [{label}] ident={ident} (no frame) ---\n")
                    continue
                buf.append(f"--- [{label}] ident={ident} ---")
                buf.append("".join(traceback.format_stack(fr, limit=80)))
                buf.append("")
            print("\n".join(buf), flush=True)

    with _watchdog_lock:
        if _watchdog_started:
            return
        _watchdog_started = True
        threading.Thread(target=watchdog_loop, name="stack-watchdog", daemon=True).start()

# ── Patch SDK internal thread to add monitoring ───────────────────
def _patch_internal_client_holder_background_thread():
    import aiomonitor
    from tinker.lib.internal_client_holder import InternalClientHolderThreadSingleton

    def _background_thread_func_with_monitor(self):
        assert self._loop is not None
        _watchdog_register_this_thread("holder")
        print(
            f"aiomonitor [holder thread]: telnet {AIOMONITOR_HOST}:{HOLDER_AIOMONITOR['port']}  "
            f"web http://{AIOMONITOR_HOST}:{HOLDER_AIOMONITOR['webui_port']}  "
            f"console {AIOMONITOR_HOST}:{HOLDER_AIOMONITOR['console_port']}"
        )
        with aiomonitor.start_monitor(self._loop, host=AIOMONITOR_HOST, **HOLDER_AIOMONITOR):
            self._loop.create_task(
                _periodic_dump_tasks(self._loop, "holder"),
                name="periodic-task-dump-holder",
            )
            self._loop.run_forever()

    setattr(
        InternalClientHolderThreadSingleton,
        "_background_thread_func",
        _background_thread_func_with_monitor,
    )

# ── Main runner with monitoring ───────────────────────────────────
def _run_main_with_aiomonitor(task):
    import contextlib
    import aiomonitor

    _watchdog_register_this_thread("main")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    dump_t = None
    try:
        print(
            f"aiomonitor [main]: telnet {AIOMONITOR_HOST}:{MAIN_AIOMONITOR['port']}  "
            f"web http://{AIOMONITOR_HOST}:{MAIN_AIOMONITOR['webui_port']}  "
            f"console {AIOMONITOR_HOST}:{MAIN_AIOMONITOR['console_port']}"
        )
        with aiomonitor.start_monitor(loop, host=AIOMONITOR_HOST, **MAIN_AIOMONITOR):
            dump_t = loop.create_task(
                _periodic_dump_tasks(loop, "main"),
                name="periodic-task-dump-main",
            )
            loop.run_until_complete(task)
    finally:
        if dump_t is not None and not dump_t.done():
            dump_t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(dump_t)
        with contextlib.suppress(RuntimeError):
            loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

# ── Activate everything ───────────────────────────────────────────
_ensure_thread_stack_watchdog()
_patch_internal_client_holder_background_thread()

# Replace: asyncio.run(main(config))
# With:
_run_main_with_aiomonitor(main(config))
```

## Interpreting the output

### Healthy training (no stalls)

You should see both `[main]` and `[holder]` task dumps every second, and the holder thread should cycle through `_forward_backward_async` / `_optim_step_async` / `_result_async` tasks as work is submitted and completed.

### Client-side stall

If the `[main]` thread stops logging for several seconds, but `[holder]` keeps logging, the main thread is blocked on synchronous work. Check the stack watchdog output — it will show exactly where the main thread is stuck (data loading, tokenizer import, etc.).

### SDK serialization stall

If the `[holder]` thread stops logging, the SDK's internal event loop is blocked by synchronous code. The stack watchdog will reveal the exact call — commonly pydantic `model_dump()` or `__repr__` calls during payload serialization.

Example stack trace from a pydantic serialization stall:
```
File ".../tinker/_compat.py", line 143, in model_dump
    return model.model_dump(
File ".../pydantic/main.py", line 479, in model_dump
    return self.__pydantic_serializer__.to_python(
File ".../pydantic/_internal/_serializers.py", line 44, in serialize_sequence_via_list
    v = handler(item, index)
```

This indicates a pydantic version issue — see Step 1 in the main skill.

### Server-side wait

If both threads are logging normally and the holder shows pending `_result_async` tasks, the SDK has submitted work and is waiting for the server. Share the session ID and these logs with the Tinker team.

## Simplified version (no aiomonitor)

If the user can't install aiomonitor, use just the stack watchdog:

```python
import asyncio, sys, threading, time, traceback

_watchdog_targets = {}
_lock = threading.Lock()

def register_thread(label):
    with _lock:
        _watchdog_targets[threading.current_thread().ident] = label

def start_watchdog(interval=1.0):
    def loop():
        while True:
            time.sleep(interval)
            ts = time.strftime("%H:%M:%S")
            with _lock:
                targets = dict(_watchdog_targets)
            frames = sys._current_frames()
            for ident, label in targets.items():
                fr = frames.get(ident)
                if fr:
                    stack = "".join(traceback.format_stack(fr, limit=10))
                    print(f"\n[{ts}] {label}:\n{stack}", flush=True)
    threading.Thread(target=loop, daemon=True).start()

# Add before any tinker imports:
register_thread("main")
start_watchdog()

# Then monkey-patch the holder thread to register itself:
from tinker.lib.internal_client_holder import InternalClientHolderThreadSingleton
_orig = InternalClientHolderThreadSingleton._background_thread_func
def _patched(self):
    register_thread("holder")
    _orig(self)
InternalClientHolderThreadSingleton._background_thread_func = _patched

# ... rest of script ...
```
