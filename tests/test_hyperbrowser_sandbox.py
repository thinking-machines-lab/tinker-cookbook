"""Smoke tests for HyperbrowserSandbox.

Require a Hyperbrowser API key and network access; skipped when
HYPERBROWSER_API_KEY is not set.

These exercise the paths that unit tests with a fake SDK cannot: real exec
semantics (exit codes, timeouts, working directories), real file transfer, and
cleanup against a sandbox that has already expired. None of them need a local
Docker daemon — they layer packages onto a base image at startup rather than
building one.
"""

import asyncio
import os
import time

import pytest
import pytest_asyncio

from tinker_cookbook.sandbox.hyperbrowser_sandbox import (
    HyperbrowserImage,
    HyperbrowserSandbox,
    HyperbrowserSandboxPool,
    close_client,
)

requires_hyperbrowser = pytest.mark.skipif(
    not os.environ.get("HYPERBROWSER_API_KEY"),
    reason="HYPERBROWSER_API_KEY not set",
)

_IMAGE = HyperbrowserImage.base("python")


@pytest_asyncio.fixture(scope="module")
async def sandbox():
    """Shared Hyperbrowser sandbox for all tests in this module."""
    sb = await HyperbrowserSandbox.create(image=_IMAGE, timeout=300)
    yield sb
    await sb.cleanup()
    await close_client()


async def _timed(coro):
    """Await a coroutine and return (result, elapsed_seconds)."""
    start = time.monotonic()
    result = await coro
    return result, time.monotonic() - start


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_run_command(sandbox):
    result = await sandbox.run_command("echo hello world")
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "hello world"


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_run_command_reports_nonzero_exit(sandbox):
    result = await sandbox.run_command("exit 7")
    assert result.exit_code == 7


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_run_command_captures_stderr(sandbox):
    result = await sandbox.run_command("echo oops >&2; exit 1")
    assert result.exit_code == 1
    assert "oops" in result.stderr


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_run_command_honors_workdir(sandbox):
    await sandbox.run_command("mkdir -p /tmp/wd-probe")
    result = await sandbox.run_command("pwd", workdir="/tmp/wd-probe")
    assert result.stdout.strip() == "/tmp/wd-probe"


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_run_command_timeout_is_not_a_hang(sandbox):
    """A command that outlasts its timeout must return, not block."""
    result, elapsed = await _timed(sandbox.run_command("sleep 30", timeout=5))
    assert result.exit_code != 0
    assert elapsed < 25, f"timeout took {elapsed:.1f}s — expected the 5s cap to be honored"


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_run_command_caps_output(sandbox):
    result = await sandbox.run_command("yes abc | head -c 100000", max_output_bytes=1024)
    assert len(result.stdout) == 1024


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_default_user_can_write_outside_home(sandbox):
    """Harbor's bash tool and grader write to /logs and /tests, so we need root."""
    result = await sandbox.run_command("mkdir -p /logs/verifier && touch /logs/verifier/ok")
    assert result.exit_code == 0, result.stderr


# ---------------------------------------------------------------------------
# File transfer
# ---------------------------------------------------------------------------


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_write_file_latency(sandbox):
    """write_file should complete in seconds, not minutes."""
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


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_write_file_binary(sandbox):
    """write_file should handle binary content correctly."""
    content = bytes(range(256))

    result, elapsed = await _timed(sandbox.write_file("/tmp/binary.bin", content, timeout=30))
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"
    assert elapsed < 15, f"write_file took {elapsed:.1f}s (expected <15s)"

    size_result = await sandbox.run_command("wc -c < /tmp/binary.bin")
    assert size_result.exit_code == 0
    assert int(size_result.stdout.strip()) == 256

    print(f"\nwrite_file (binary) latency: {elapsed:.2f}s")


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_write_file_creates_parent_directories(sandbox):
    result = await sandbox.write_file("/tmp/deeply/nested/dir/file.txt", "content")
    assert result.exit_code == 0, result.stderr
    read_back = await sandbox.read_file("/tmp/deeply/nested/dir/file.txt")
    assert read_back.stdout == "content"


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_read_file_roundtrip_and_truncation(sandbox):
    await sandbox.write_file("/tmp/read-probe.txt", "abcdefghij")

    full = await sandbox.read_file("/tmp/read-probe.txt")
    assert (full.exit_code, full.stdout) == (0, "abcdefghij")

    partial = await sandbox.read_file("/tmp/read-probe.txt", max_bytes=4)
    assert partial.stdout == "abcd"


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_read_missing_file_is_a_nonzero_exit(sandbox):
    """Must not raise — callers treat this like a failed `cat`."""
    result = await sandbox.read_file("/tmp/definitely-not-here.txt")
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Image layers
# ---------------------------------------------------------------------------


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_base_image_layers_apply_without_docker():
    """pip_install on a base image runs at startup, needing no local Docker."""
    image = HyperbrowserImage.base("python").pip_install("six")
    sb = await HyperbrowserSandbox.create(image=image, timeout=300)
    try:
        result = await sb.run_command("python3 -c 'import six; print(six.__name__)'")
        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "six"
    finally:
        await sb.cleanup()


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_pool_run_in_workdir():
    pool = HyperbrowserSandboxPool(pool_size=2, sandbox_timeout_secs=300, image=_IMAGE)
    try:
        result = await pool.run_in_workdir(
            files={"code.py": "print('from the pool')"},
            command=["python3", "code.py"],
            timeout=60,
        )
        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "from the pool"
    finally:
        await pool.terminate()


# ---------------------------------------------------------------------------
# cleanup() resilience
# ---------------------------------------------------------------------------


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_cleanup_is_idempotent():
    """cleanup() should not raise if called twice."""
    sb = await HyperbrowserSandbox.create(image=_IMAGE, timeout=300)
    await sb.cleanup()
    await sb.cleanup()


@requires_hyperbrowser
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_cleanup_after_expiry():
    """cleanup() should not raise once the sandbox lifetime has elapsed."""
    # Hyperbrowser lifetimes are minute-granular, so 1 minute is the floor.
    sb = await HyperbrowserSandbox.create(image=_IMAGE, timeout=60)
    await asyncio.sleep(75)
    await sb.cleanup()
