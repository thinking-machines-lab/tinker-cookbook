"""Smoke tests for FystashSandbox.

Require FYSTASH_API_KEY and network access; skipped when the key is unset.
Mirrors tests/test_modal_sandbox.py for local/partner evidence (not PR CI).
"""

from __future__ import annotations

import os
import time

import pytest
import pytest_asyncio

from tinker_cookbook.sandbox.fystash_sandbox import FystashSandbox

_has_fystash_auth = bool(os.environ.get("FYSTASH_API_KEY"))

requires_fystash = pytest.mark.skipif(
    not _has_fystash_auth, reason="FYSTASH_API_KEY not set"
)


@pytest_asyncio.fixture(scope="module")
async def sandbox():
    """Shared Fystash sandbox for all tests in this module."""
    sb = await FystashSandbox.create(timeout=120)
    yield sb
    await sb.cleanup()


async def _timed(coro):
    """Await a coroutine and return (result, elapsed_seconds)."""
    start = time.monotonic()
    result = await coro
    return result, time.monotonic() - start


@requires_fystash
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_write_file_latency(sandbox):
    """write_file should complete in seconds and round-trip via run_command."""
    content = "#!/bin/bash\necho hello world\n"

    result, elapsed = await _timed(
        sandbox.write_file("/tmp/test.sh", content, executable=True, timeout=30)
    )
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"
    assert elapsed < 30, f"write_file took {elapsed:.1f}s (expected <30s)"

    read_result = await sandbox.run_command("cat /tmp/test.sh")
    assert read_result.exit_code == 0
    assert read_result.stdout == content

    stat_result = await sandbox.run_command("test -x /tmp/test.sh && echo yes")
    assert stat_result.stdout.strip() == "yes"

    print(f"\nwrite_file latency: {elapsed:.2f}s")


@requires_fystash
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_write_file_binary(sandbox):
    """write_file should handle binary content correctly."""
    content = bytes(range(256))

    result, elapsed = await _timed(sandbox.write_file("/tmp/binary.bin", content, timeout=30))
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"
    assert elapsed < 30, f"write_file took {elapsed:.1f}s (expected <30s)"

    size_result = await sandbox.run_command("wc -c < /tmp/binary.bin")
    assert size_result.exit_code == 0
    assert int(size_result.stdout.strip()) == 256

    print(f"\nwrite_file (binary) latency: {elapsed:.2f}s")


@requires_fystash
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cleanup_idempotent():
    """cleanup() should not raise if called twice."""
    sb = await FystashSandbox.create(timeout=120)
    await sb.cleanup()
    await sb.cleanup()


@requires_fystash
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cleanup_after_command():
    """cleanup() should work after a successful run_command."""
    sb = await FystashSandbox.create(timeout=120)
    result = await sb.run_command("echo ok", timeout=30)
    assert result.exit_code == 0
    assert "ok" in result.stdout
    await sb.cleanup()
