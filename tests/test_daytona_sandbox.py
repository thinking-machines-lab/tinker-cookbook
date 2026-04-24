"""Smoke tests for DaytonaSandbox.

Require Daytona authentication and network access; skipped when no
``DAYTONA_API_KEY`` or ``DAYTONA_JWT_TOKEN`` is set in the environment.

Covers the SandboxInterface contract (run_command state persistence,
write/read round-trip, heartbeat, idempotent cleanup, SandboxTerminatedError
after cleanup) plus the stateless grading helper and a ``write_file``
latency bound to catch future performance regressions.
"""

import os
import time

import pytest
import pytest_asyncio

from tinker_cookbook.sandbox.daytona_sandbox import (
    DaytonaSandbox,
    run_code_in_daytona,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxTerminatedError

_has_daytona_auth = bool(os.environ.get("DAYTONA_API_KEY") or os.environ.get("DAYTONA_JWT_TOKEN"))

requires_daytona = pytest.mark.skipif(
    not _has_daytona_auth,
    reason="Daytona not configured (set DAYTONA_API_KEY or DAYTONA_JWT_TOKEN)",
)


@pytest_asyncio.fixture(scope="module")
async def sandbox():
    """Shared Daytona sandbox for persistent-state tests in this module."""
    sb = await DaytonaSandbox.create(timeout=300)
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


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_run_command_basic(sandbox):
    """run_command returns structured stdout/exit_code."""
    result = await sandbox.run_command("echo hello")
    assert result.exit_code == 0, f"stderr: {result.stderr}"
    assert "hello" in result.stdout


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_run_command_state_persists(sandbox):
    """Shell state (cwd) persists across run_command calls via the session."""
    cd_result = await sandbox.run_command("cd /tmp")
    assert cd_result.exit_code == 0

    pwd_result = await sandbox.run_command("pwd")
    assert pwd_result.exit_code == 0
    assert "/tmp" in pwd_result.stdout


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_filesystem_round_trip(sandbox):
    """write_file then read_file returns the same content."""
    content = "hello from tinker-cookbook\n"
    write_result = await sandbox.write_file("/tmp/probe.txt", content)
    assert write_result.exit_code == 0, f"stderr: {write_result.stderr}"

    read_result = await sandbox.read_file("/tmp/probe.txt")
    assert read_result.exit_code == 0
    assert read_result.stdout == content


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_write_file_executable(sandbox):
    """write_file(executable=True) makes the file executable."""
    script = "#!/bin/bash\necho ran from script\n"
    write_result = await sandbox.write_file("/tmp/probe.sh", script, executable=True)
    assert write_result.exit_code == 0, f"stderr: {write_result.stderr}"

    stat_result = await sandbox.run_command("test -x /tmp/probe.sh && echo yes")
    assert stat_result.stdout.strip() == "yes"


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_send_heartbeat(sandbox):
    """send_heartbeat succeeds on a live sandbox."""
    await sandbox.send_heartbeat()


# ---------------------------------------------------------------------------
# write_file latency — catches broad performance regressions
# ---------------------------------------------------------------------------


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_write_file_latency(sandbox):
    """write_file should complete in seconds, not minutes.

    The Daytona SDK's ``fs.upload_files`` is a multipart HTTP POST — no
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


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cleanup_is_idempotent():
    """Calling cleanup twice on the same sandbox does not raise."""
    sb = await DaytonaSandbox.create(timeout=120)

    await sb.cleanup()
    # Second call must be a no-op, not raise.
    await sb.cleanup()


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_command_after_cleanup_raises_terminated():
    """run_command after cleanup surfaces SandboxTerminatedError."""
    sb = await DaytonaSandbox.create(timeout=120)
    await sb.cleanup()

    with pytest.raises(SandboxTerminatedError):
        await sb.run_command("echo should not run")


# ---------------------------------------------------------------------------
# Stateless grading helper (code_rl path)
# ---------------------------------------------------------------------------


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_run_code_in_daytona_success():
    """run_code_in_daytona returns (True, {...stdout...}) on a working program."""
    success, response = await run_code_in_daytona(
        code="print(2 + 2)",
        files={},
        timeout=60,
    )
    assert success is True, f"response: {response}"
    assert "4" in response["stdout"]
    assert response["exit_code"] == 0


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_run_code_in_daytona_failure():
    """run_code_in_daytona returns (False, {...}) when the program raises."""
    success, response = await run_code_in_daytona(
        code="raise ValueError('boom from test')",
        files={},
        timeout=60,
    )
    assert success is False
    assert response.get("exit_code", 1) != 0 or "error" in response


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_run_code_in_daytona_with_files():
    """run_code_in_daytona uploads supporting files alongside the entry point."""
    success, response = await run_code_in_daytona(
        code="print(open('data.txt').read().strip())",
        files={"data.txt": "tinker-cookbook was here"},
        timeout=60,
    )
    assert success is True, f"response: {response}"
    assert "tinker-cookbook was here" in response["stdout"]
