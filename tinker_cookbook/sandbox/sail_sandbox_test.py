import asyncio
import atexit
from collections.abc import AsyncIterator, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("sail")

import sail

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.sandbox.sail_sandbox import (
    SailSandbox,
    SailSandboxPool,
    _bash_argv,
    _read_stream_capped,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxTerminatedError


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _fake_process(result: sail.ExecResult, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    process = MagicMock()
    process.__aenter__.return_value = process
    process.stdout_bytes = _chunks(stdout)
    process.stderr_bytes = _chunks(stderr)
    process.wait = AsyncMock(return_value=result)
    return process


def _fake_sailbox(
    process: MagicMock | None = None, exec_error: Exception | None = None
) -> MagicMock:
    sailbox = MagicMock()
    sailbox.sailbox_id = "sb_test"
    sailbox.exec.aio = AsyncMock(return_value=process, side_effect=exec_error)
    sailbox.fs.write.aio = AsyncMock()
    sailbox.fs.write_files.aio = AsyncMock()
    sailbox.terminate.aio = AsyncMock()
    return sailbox


def test_read_stream_capped_keeps_head_and_drains() -> None:
    drained = False

    async def stream() -> AsyncIterator[bytes]:
        nonlocal drained
        yield b"abc"
        yield b"def"
        yield b"ghi"
        drained = True

    assert asyncio.run(_read_stream_capped(stream(), 4)) == "abcd"
    assert drained


def test_bash_argv() -> None:
    assert _bash_argv("echo hi", None) == ["bash", "-lc", "echo hi"]
    assert _bash_argv("echo hi", "/tmp/w d") == [
        "bash",
        "-lc",
        "cd '/tmp/w d' && exec bash -c 'echo hi'",
    ]


def test_run_command_returns_output_and_exit_code() -> None:
    result = sail.ExecResult(stdout="", stderr="", exit_code=3, timed_out=True)
    sailbox = _fake_sailbox(_fake_process(result, stdout=b"out", stderr=b"err"))
    sandbox = SailSandbox(sailbox)

    got = asyncio.run(sandbox.run_command("python run.py", workdir="/w", timeout=7))

    assert (got.stdout, got.stderr, got.exit_code) == ("out", "err", 3)
    assert got.metrics == {"timed_out": True}
    sailbox.exec.aio.assert_awaited_once_with(
        _bash_argv("python run.py", "/w"), timeout=7, user="0:0"
    )


def test_run_command_caps_output() -> None:
    result = sail.ExecResult(stdout="", stderr="", exit_code=0)
    sandbox = SailSandbox(_fake_sailbox(_fake_process(result, stdout=b"x" * 10)))

    assert asyncio.run(sandbox.run_command("true", max_output_bytes=4)).stdout == "xxxx"


def test_run_command_terminated_sailbox() -> None:
    sandbox = SailSandbox(_fake_sailbox(exec_error=sail.SailboxTerminatedError("gone")))

    with pytest.raises(SandboxTerminatedError):
        asyncio.run(sandbox.run_command("true"))


def test_run_command_other_sail_error_becomes_result() -> None:
    sandbox = SailSandbox(_fake_sailbox(exec_error=sail.SailError("boom")))

    got = asyncio.run(sandbox.run_command("true"))

    assert (got.stdout, got.stderr, got.exit_code) == ("", "SailError: boom", -1)


def test_read_file_caps_bytes() -> None:
    stream = MagicMock()
    stream.__aenter__.return_value = stream
    stream.__aiter__.return_value = [b"hel", b"lo world"]
    sailbox = _fake_sailbox()
    sailbox.fs.read_stream.return_value = stream
    sandbox = SailSandbox(sailbox)

    assert asyncio.run(sandbox.read_file("/f", max_bytes=5)).stdout == "hello"
    assert asyncio.run(sandbox.read_file("/f")).stdout == "hello world"


def test_write_file_executable_mode() -> None:
    sailbox = _fake_sailbox()
    sandbox = SailSandbox(sailbox)

    got = asyncio.run(sandbox.write_file("/run.sh", "echo", executable=True))

    assert got.exit_code == 0
    sailbox.fs.write.aio.assert_awaited_once_with("/run.sh", "echo", mode=0o755)


def test_pool_run_in_workdir() -> None:
    result = sail.ExecResult(stdout="ok", stderr="", exit_code=0)
    sailbox = _fake_sailbox(_fake_process(result, stdout=b"ok"))

    class _Pool(SailSandboxPool):
        async def _create(self) -> SailSandbox:
            return SailSandbox(sailbox)

    async def run() -> None:
        pool = _Pool(pool_size=1)
        got = await pool.run_in_workdir({"run.py": "print(1)"}, ["python", "run.py"])
        assert got.stdout == "ok"
        await pool.terminate()

    asyncio.run(run())

    (files,) = sailbox.fs.write_files.aio.await_args.args
    (path,) = files
    workdir, filename = path.rsplit("/", 1)
    assert workdir.startswith("/workspace/") and filename == "run.py"
    sailbox.exec.aio.assert_awaited_once_with(
        _bash_argv("python run.py", workdir), timeout=1200, user="0:0"
    )
    sailbox.terminate.aio.assert_awaited_once()


def test_pool_terminate_waits_for_in_flight_create() -> None:
    sailbox = _fake_sailbox()

    async def run() -> None:
        create_started = asyncio.Event()
        release_create = asyncio.Event()

        class _Pool(SailSandboxPool):
            async def _create(self) -> SailSandbox:
                create_started.set()
                await release_create.wait()
                return SailSandbox(sailbox)

        pool = _Pool(pool_size=1)
        await create_started.wait()
        terminating = asyncio.create_task(pool.terminate())
        await asyncio.sleep(0)
        assert not terminating.done()
        release_create.set()
        await terminating

    asyncio.run(run())
    sailbox.terminate.aio.assert_awaited_once()


def test_pool_stops_creating_when_creation_cannot_succeed() -> None:
    async def run() -> None:
        attempts = 0

        class _Pool(SailSandboxPool):
            async def _create(self) -> SailSandbox:
                nonlocal attempts
                attempts += 1
                raise sail.PermissionDeniedError("bad key")

        pool = _Pool(pool_size=1)
        with pytest.raises(SandboxError, match="bad key"):
            await pool.run_in_workdir({}, ["true"])
        await asyncio.sleep(1.2)  # Long enough for another maintenance step.
        assert attempts == 1
        await pool.terminate()

    asyncio.run(run())


def test_pool_retries_transient_creation_errors() -> None:
    result = sail.ExecResult(stdout="ok", stderr="", exit_code=0)
    sailbox = _fake_sailbox(_fake_process(result, stdout=b"ok"))
    attempts = 0

    class _Pool(SailSandboxPool):
        async def _create(self) -> SailSandbox:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise sail.TransportError("connection reset")
            return SailSandbox(sailbox)

    async def run() -> None:
        pool = _Pool(pool_size=1)
        got = await pool.run_in_workdir({"run.py": "print(1)"}, ["python", "run.py"])
        assert got.stdout == "ok"
        await pool.terminate()

    asyncio.run(run())
    assert attempts == 2


def test_pool_retries_cleanup_after_transient_error() -> None:
    result = sail.ExecResult(stdout="", stderr="", exit_code=0)
    first = _fake_sailbox(_fake_process(result))
    first.terminate.aio.side_effect = [sail.TransportError("connection reset"), None]
    second = _fake_sailbox()
    sailboxes = [first, second]

    class _Pool(SailSandboxPool):
        async def _create(self) -> SailSandbox:
            return SailSandbox(sailboxes.pop(0))

    async def run() -> None:
        pool = _Pool(pool_size=1)
        await pool.run_in_workdir({}, ["true"])
        while first.terminate.aio.await_count == 0:
            await asyncio.sleep(0.05)
        await pool.terminate()

    asyncio.run(run())
    assert first.terminate.aio.await_count == 2
    second.terminate.aio.assert_awaited_once()


def test_pool_terminates_leftover_sandboxes_at_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    result = sail.ExecResult(stdout="", stderr="", exit_code=0)
    sailbox = _fake_sailbox(_fake_process(result))
    exit_hooks: list[Callable[[], None]] = []
    monkeypatch.setattr(atexit, "register", exit_hooks.append)

    class _Pool(SailSandboxPool):
        async def _create(self) -> SailSandbox:
            return SailSandbox(sailbox)

    async def run() -> None:
        pool = _Pool(pool_size=1)
        await pool.run_in_workdir({}, ["true"])

    asyncio.run(run())
    (exit_hook,) = exit_hooks
    exit_hook()
    sailbox.terminate.assert_called_once_with()
    sailbox.terminate.aio.assert_not_awaited()


def test_pool_terminate_fails_waiting_callers() -> None:
    async def run() -> None:
        class _Pool(SailSandboxPool):
            async def _create(self) -> SailSandbox:
                raise sail.TransportError("connection reset")

        pool = _Pool(pool_size=1)
        waiting = asyncio.create_task(pool.run_in_workdir({"run.py": ""}, ["true"]))
        await asyncio.sleep(0)
        await pool.terminate()
        with pytest.raises(SandboxError, match="terminated"):
            await waiting

    asyncio.run(run())


def test_pool_counts_a_taken_sandbox_active_at_once() -> None:
    """terminate() relies on every sandbox being warm or active whenever it looks."""
    result = sail.ExecResult(stdout="", stderr="", exit_code=0)
    process = _fake_process(result)
    sailbox = _fake_sailbox(process)

    class _Pool(SailSandboxPool):
        async def _create(self) -> SailSandbox:
            return SailSandbox(sailbox)

    async def run() -> None:
        release = asyncio.Event()

        async def wait() -> sail.ExecResult:
            await release.wait()
            return result

        process.wait = AsyncMock(side_effect=wait)
        pool = _Pool(pool_size=1)
        while pool._warm_pool.empty():
            await asyncio.sleep(0.05)
        accounted: list[int] = []

        async def watch() -> None:
            while True:
                accounted.append(pool._warm_pool.qsize() + pool._active_count)
                await asyncio.sleep(0)

        watcher = asyncio.create_task(watch())
        running = asyncio.create_task(pool.run_in_workdir({}, ["true"]))
        while pool._active_count == 0:
            await asyncio.sleep(0)
        watcher.cancel()
        release.set()
        await running
        assert set(accounted) == {1}
        await pool.terminate()

    asyncio.run(run())
