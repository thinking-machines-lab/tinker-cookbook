"""Smoke tests for SailSandbox.

Require a Sail API key and network access; skipped when SAIL_API_KEY is not set.
Each test talks to a real Sailbox, so the module covers the backend end to end:
file writes and reads, command execution with a working directory and a timeout,
and cleanup() after the Sailbox is already gone.
"""

import asyncio
import contextlib
import os
import time

import pytest
import pytest_asyncio

pytest.importorskip("sail")

import sail

from tinker_cookbook.sandbox.sail_sandbox import SailSandbox
from tinker_cookbook.sandbox.sandbox_interface import SandboxTerminatedError

_has_sail_auth = bool(os.environ.get("SAIL_API_KEY"))

requires_sail = pytest.mark.skipif(not _has_sail_auth, reason="Sail not configured locally")

_SAIL_IMAGE = sail.Image.from_registry("python:3.12-slim")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sandbox():
    """Shared Sailbox for all tests in this module."""
    sb = await SailSandbox.create(image=_SAIL_IMAGE, max_lifetime_seconds=300)
    yield sb
    await sb.cleanup()


async def _timed(coro):
    """Await a coroutine and return (result, elapsed_seconds)."""
    start = time.monotonic()
    result = await coro
    return result, time.monotonic() - start


@requires_sail
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(180)
async def test_write_file_latency(sandbox):
    """write_file should complete in seconds and land intact with the executable bit set."""
    content = "#!/bin/bash\necho hello world\n"

    result, elapsed = await _timed(
        sandbox.write_file("/tmp/test.sh", content, executable=True, timeout=30)
    )
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"
    assert elapsed < 15, f"write_file took {elapsed:.1f}s (expected <15s)"

    read_result = await sandbox.run_command("cat /tmp/test.sh")
    assert read_result.exit_code == 0
    assert read_result.stdout == content

    stat_result = await sandbox.run_command("test -x /tmp/test.sh && echo yes")
    assert stat_result.stdout.strip() == "yes"

    print(f"\nwrite_file latency: {elapsed:.2f}s")


@requires_sail
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(60)
async def test_write_file_binary(sandbox):
    """write_file should handle binary content correctly."""
    content = bytes(range(256))

    result = await sandbox.write_file("/tmp/binary.bin", content, timeout=30)
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"

    size_result = await sandbox.run_command("wc -c < /tmp/binary.bin")
    assert size_result.exit_code == 0
    assert int(size_result.stdout.strip()) == 256


@requires_sail
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(60)
async def test_read_file_and_workdir(sandbox):
    """run_command honors workdir, and read_file returns the file, capped on request."""
    result = await sandbox.run_command("pwd > out.txt", workdir="/tmp")
    assert result.exit_code == 0, result.stderr

    assert (await sandbox.read_file("/tmp/out.txt")).stdout.strip() == "/tmp"
    assert (await sandbox.read_file("/tmp/out.txt", max_bytes=2)).stdout == "/t"
    assert (await sandbox.read_file("/tmp/does-not-exist")).exit_code != 0


@requires_sail
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(60)
async def test_command_timeout(sandbox):
    """A command that outlives its timeout is killed and reported as timed out."""
    result = await sandbox.run_command("sleep 30", timeout=2)
    assert result.exit_code != 0
    assert result.metrics == {"timed_out": True}


# ---------------------------------------------------------------------------
# cleanup() resilience tests
# ---------------------------------------------------------------------------


@requires_sail
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cleanup_after_timeout():
    """cleanup() should not raise even if the Sailbox's lifetime has already elapsed."""
    sb = await SailSandbox.create(image=_SAIL_IMAGE, max_lifetime_seconds=10)

    # Wait for the lifetime to elapse
    await asyncio.sleep(12)

    await sb.cleanup()


@requires_sail
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cleanup_after_terminate():
    """cleanup() should not raise if called twice (Sailbox already terminated)."""
    sb = await SailSandbox.create(image=_SAIL_IMAGE, max_lifetime_seconds=60)

    await sb.cleanup()
    await sb.cleanup()


@requires_sail
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cleanup_after_command_timeout():
    """cleanup() should work after a command outlives the Sailbox's lifetime."""
    sb = await SailSandbox.create(image=_SAIL_IMAGE, max_lifetime_seconds=10)

    # The Sailbox may be reclaimed mid-command, which surfaces as SandboxTerminatedError.
    with contextlib.suppress(SandboxTerminatedError):
        await sb.run_command("sleep 60", timeout=20)

    await sb.cleanup()
