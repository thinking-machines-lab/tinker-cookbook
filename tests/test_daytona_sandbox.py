"""Smoke tests for DaytonaSandbox.

Require Daytona authentication and network access; skipped when no
``DAYTONA_API_KEY`` or ``DAYTONA_JWT_TOKEN`` is set in the environment.

Covers the SandboxInterface contract as ModalSandbox implements it (a fresh
shell per run_command, separate stdout/stderr, output caps, timeouts,
background processes, write/read round-trip, missing files, heartbeat,
idempotent cleanup, SandboxTerminatedError once the sandbox is gone), the
Harbor sandbox factory, and the stateless grading helper.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from daytona import AsyncDaytona, Image, SandboxState

from tinker_cookbook.sandbox.daytona_sandbox import (
    DaytonaSandbox,
    daytona_sandbox_factory,
    run_code_in_daytona,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxTerminatedError

_has_daytona_auth = bool(os.environ.get("DAYTONA_API_KEY") or os.environ.get("DAYTONA_JWT_TOKEN"))

pytestmark = [
    pytest.mark.skipif(
        not _has_daytona_auth,
        reason="Daytona not configured (set DAYTONA_API_KEY or DAYTONA_JWT_TOKEN)",
    ),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sandbox():
    """Shared Daytona sandbox for the tests in this module."""
    sb = await DaytonaSandbox.create(timeout=900)
    yield sb
    await sb.cleanup()


async def _timed(coro):
    """Await a coroutine and return (result, elapsed_seconds)."""
    start = time.monotonic()
    result = await coro
    return result, time.monotonic() - start


# ---------------------------------------------------------------------------
# Core SandboxInterface contract
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
async def test_run_command_basic(sandbox):
    """run_command returns structured stdout/exit_code."""
    result = await sandbox.run_command("echo hello")
    assert result.exit_code == 0, f"stderr: {result.stderr}"
    assert "hello" in result.stdout


@pytest.mark.timeout(120)
async def test_run_command_separates_streams_and_exit_code(sandbox):
    """stdout and stderr come back apart, with the command's exit code."""
    result = await sandbox.run_command("echo out; echo err >&2; exit 3")
    assert result.exit_code == 3
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"


@pytest.mark.timeout(120)
async def test_run_command_fresh_shell_per_call(sandbox):
    """Like ModalSandbox, shell state does not carry over between calls."""
    await sandbox.run_command("cd /tmp; export PROBE_VAR=set")
    result = await sandbox.run_command('pwd; echo "var=${PROBE_VAR:-unset}"')
    assert result.exit_code == 0
    assert "/tmp" not in result.stdout
    assert "var=unset" in result.stdout


@pytest.mark.timeout(120)
async def test_run_command_workdir(sandbox):
    """workdir scopes one call; None runs in the image's working directory."""
    default = await sandbox.run_command("pwd")
    scoped = await sandbox.run_command("pwd", workdir="/tmp")
    assert scoped.stdout.strip() == "/tmp"
    after = await sandbox.run_command("pwd")
    assert after.stdout.strip() == default.stdout.strip()


@pytest.mark.timeout(120)
async def test_run_command_quoting(sandbox):
    """Commands reach bash verbatim, quotes and all."""
    result = await sandbox.run_command("""printf '%s|' "a b" 'c"d' $'e\\tf'""")
    assert result.stdout.strip() == 'a b|c"d|e\tf|'


@pytest.mark.timeout(120)
async def test_run_command_caps_output(sandbox):
    """max_output_bytes caps each stream."""
    result = await sandbox.run_command(
        "yes | head -c 100000; yes | head -c 100000 >&2", max_output_bytes=1000
    )
    assert result.exit_code == 0
    assert len(result.stdout.encode()) <= 1000
    assert len(result.stderr.encode()) <= 1000


@pytest.mark.timeout(120)
async def test_run_command_timeout_kills_command(sandbox):
    """A command still running at its timeout is killed and reports it."""
    result, elapsed = await _timed(sandbox.run_command("sleep 100", timeout=3))
    assert result.exit_code in (124, 137)
    assert elapsed < 30


@pytest.mark.timeout(120)
async def test_run_command_timeout_kills_children_that_outlive_sigterm(sandbox):
    """A child that ignores SIGTERM neither delays the result nor outlives the timeout."""
    result, elapsed = await _timed(
        sandbox.run_command("(trap '' TERM; sleep 100); echo after", timeout=3)
    )
    assert result.exit_code == 124
    assert elapsed < 20
    left = await sandbox.run_command("pgrep -f 'sleep 100' || echo none")
    assert left.stdout.strip() == "none"


@pytest.mark.timeout(120)
async def test_run_command_stdin_is_empty(sandbox):
    """A command that reads stdin gets end-of-file instead of waiting for input."""
    result, elapsed = await _timed(sandbox.run_command("cat; read -r line; echo after $?"))
    assert result.stdout.strip() == "after 1"
    assert elapsed < 15


@pytest.mark.slow
@pytest.mark.timeout(900)
async def test_long_command_returns_its_result(sandbox):
    """A command running for minutes still returns its own output and exit code."""
    result = await sandbox.run_command("sleep 420; echo done", timeout=600)
    assert result.exit_code == 0, f"stderr: {result.stderr}"
    assert result.stdout.strip() == "done"


@pytest.mark.timeout(120)
async def test_background_process_survives_the_call(sandbox):
    """A process started in the background keeps running after its command returns."""
    started, elapsed = await _timed(
        sandbox.run_command("nohup sleep 300 > /dev/null 2>&1 & echo $! > /tmp/bg.pid")
    )
    assert started.exit_code == 0, f"stderr: {started.stderr}"
    assert elapsed < 15, f"run_command waited {elapsed:.1f}s for the background process"
    result = await sandbox.run_command("kill -0 $(cat /tmp/bg.pid) && echo alive")
    assert result.stdout.strip() == "alive", f"stderr: {result.stderr}"


@pytest.mark.timeout(120)
async def test_filesystem_round_trip(sandbox):
    """write_file then read_file returns the same content, creating parent directories."""
    content = "hello from tinker-cookbook\n"
    write_result = await sandbox.write_file("/tmp/nested/dir/probe.txt", content)
    assert write_result.exit_code == 0, f"stderr: {write_result.stderr}"

    read_result = await sandbox.read_file("/tmp/nested/dir/probe.txt")
    assert read_result.exit_code == 0
    assert read_result.stdout == content

    head = await sandbox.read_file("/tmp/nested/dir/probe.txt", max_bytes=5)
    assert head.stdout == "hello"


@pytest.mark.timeout(120)
async def test_read_missing_file_fails_without_terminating(sandbox):
    """Reading a missing file is an ordinary failure, not a dead sandbox."""
    result = await sandbox.read_file("/tmp/does-not-exist.txt")
    assert result.exit_code != 0
    assert (await sandbox.run_command("true")).exit_code == 0


@pytest.mark.timeout(120)
async def test_write_file_executable(sandbox):
    """write_file(executable=True) makes the file executable."""
    script = "#!/bin/bash\necho ran from script\n"
    write_result = await sandbox.write_file("/tmp/probe.sh", script, executable=True)
    assert write_result.exit_code == 0, f"stderr: {write_result.stderr}"

    result = await sandbox.run_command("/tmp/probe.sh")
    assert result.stdout.strip() == "ran from script"


@pytest.mark.timeout(60)
async def test_send_heartbeat(sandbox):
    """send_heartbeat succeeds on a live sandbox."""
    await sandbox.send_heartbeat()


# ---------------------------------------------------------------------------
# write_file latency — catches broad performance regressions
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
async def test_write_file_latency(sandbox):
    """write_file should complete in seconds, not minutes.

    The Daytona SDK's ``fs.upload_file`` is a multipart HTTP POST — no
    stdin piping — so the specific drain-hang that bit ModalSandbox does
    not apply here. This test is a generous upper bound (15s) to catch
    future regressions of any cause.
    """
    content = "#!/bin/bash\necho hello world\n"
    result, elapsed = await _timed(
        sandbox.write_file("/tmp/latency_probe.sh", content, executable=True, timeout=30)
    )
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"
    assert elapsed < 15, f"write_file took {elapsed:.1f}s (expected <15s)"


# ---------------------------------------------------------------------------
# Cleanup idempotency and post-cleanup behavior
# ---------------------------------------------------------------------------


@pytest.mark.timeout(180)
async def test_cleanup_is_idempotent():
    """Calling cleanup twice on the same sandbox does not raise."""
    sb = await DaytonaSandbox.create(timeout=300)

    await sb.cleanup()
    # Second call must be a no-op, not raise.
    await sb.cleanup()


@pytest.mark.timeout(180)
async def test_command_after_cleanup_raises_terminated():
    """run_command after cleanup surfaces SandboxTerminatedError."""
    sb = await DaytonaSandbox.create(timeout=300)
    await sb.cleanup()

    with pytest.raises(SandboxTerminatedError):
        await sb.run_command("echo should not run")


@pytest.mark.timeout(300)
async def test_cancelled_create_during_build():
    """A create cancelled while Daytona builds the image leaves no sandbox running for
    long, and does not break the image: the abandoned sandbox finishes the build, then
    stops after a minute of idling, and the next create from the image succeeds."""
    marker = uuid.uuid4().hex
    image = Image.debian_slim("3.12").run_commands(f"echo {marker} && sleep 20")
    create = asyncio.create_task(DaytonaSandbox.create(image=image, labels={"test-run": marker}))
    await asyncio.sleep(8)
    create.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create

    client = AsyncDaytona()
    try:
        for _ in range(10):  # listings can lag the update by a few seconds
            [abandoned] = [sb async for sb in client.list() if sb.labels.get("test-run") == marker]
            handled = abandoned.auto_stop_interval == 1 or abandoned.state in (
                SandboxState.DESTROYING,
                SandboxState.DESTROYED,
            )
            if handled:
                break
            await asyncio.sleep(3)
    finally:
        await client.close()
    assert handled, abandoned

    sb = await DaytonaSandbox.create(image=image, timeout=300)
    try:
        assert (await sb.run_command("true")).exit_code == 0
    finally:
        await sb.cleanup()


@pytest.mark.timeout(180)
async def test_deleted_sandbox_raises_terminated():
    """A sandbox deleted behind the wrapper's back surfaces SandboxTerminatedError."""
    sb = await DaytonaSandbox.create(timeout=300)
    try:
        await sb._client.delete(sb._sandbox, wait=True)
        with pytest.raises(SandboxTerminatedError):
            await sb.run_command("echo should not run")
        with pytest.raises(SandboxTerminatedError):
            await sb.send_heartbeat()
    finally:
        await sb.cleanup()


# ---------------------------------------------------------------------------
# Harbor sandbox factory
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
async def test_harbor_factory_builds_task_environment(tmp_path: Path):
    """The factory builds the task's Dockerfile with environment/ as build context."""
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "data.txt").write_text("from the build context\n")
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.12-slim-bookworm\nWORKDIR /app\nCOPY data.txt /app/data.txt\n"
    )
    (tmp_path / "task.toml").write_text('[environment]\ncpus = 1\nmemory = "2G"\nstorage = "10G"\n')

    sb = await daytona_sandbox_factory(env_dir, 600)
    try:
        result = await sb.run_command("pwd; cat data.txt")
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.split() == ["/app", "from", "the", "build", "context"]
    finally:
        await sb.cleanup()


# ---------------------------------------------------------------------------
# Stateless grading helper (code_rl path)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
async def test_run_code_in_daytona_success():
    """run_code_in_daytona returns (True, {...stdout...}) on a working program."""
    success, response = await run_code_in_daytona(
        code="import numpy; print(2 + 2)",
        files={},
        timeout=60,
    )
    assert success is True, f"response: {response}"
    assert "4" in response["stdout"]
    assert response["exit_code"] == 0


@pytest.mark.timeout(900)
async def test_run_code_in_daytona_failure():
    """run_code_in_daytona returns (False, {...}) when the program raises."""
    success, response = await run_code_in_daytona(
        code="raise ValueError('boom from test')",
        files={},
        timeout=60,
    )
    assert success is False
    assert response["exit_code"] != 0
    assert "boom from test" in response["stderr"]


@pytest.mark.timeout(900)
async def test_run_code_in_daytona_with_files():
    """run_code_in_daytona uploads supporting files alongside the entry point."""
    success, response = await run_code_in_daytona(
        code="print(open('data.txt').read().strip())",
        files={"data.txt": "tinker-cookbook was here"},
        timeout=60,
    )
    assert success is True, f"response: {response}"
    assert "tinker-cookbook was here" in response["stdout"]
