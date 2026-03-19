"""Benchmark tests for ModalSandbox operations.

These are smoke tests that require Modal authentication and network access.
They are skipped locally when TINKER_API_KEY is not set (via conftest.py).

The primary goal is to catch latency regressions in write_file — a previous
bug where a missing drain() after write_eof() caused 60s hangs on small files.
"""

import time

import pytest

try:
    import modal  # noqa: F401

    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

requires_modal = pytest.mark.skipif(not HAS_MODAL, reason="modal not installed")

# Modal's debian_slim() defaults to the local Python version, which may not
# be supported. Pin to 3.12 for sandbox creation.
_MODAL_IMAGE = modal.Image.debian_slim(python_version="3.12") if HAS_MODAL else None


async def _create_sandbox(**kwargs):
    """Create a ModalSandbox with a compatible Python image."""
    from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

    return await ModalSandbox.create(image=_MODAL_IMAGE, **kwargs)


async def _timed(coro):
    """Await a coroutine and return (result, elapsed_seconds)."""
    start = time.monotonic()
    result = await coro
    return result, time.monotonic() - start


@requires_modal
@pytest.mark.asyncio
async def test_write_file_latency():
    """write_file should complete in seconds, not minutes.

    Before the drain fix, write_file would hang for ~60s (the full exec timeout)
    because proc.stdin.write_eof() wasn't flushed. This test catches that
    regression by asserting a generous upper bound of 15s.
    """
    sandbox = await _create_sandbox(timeout=120)
    try:
        content = "#!/bin/bash\necho hello world\n"

        result, elapsed = await _timed(
            sandbox.write_file("/tmp/test.sh", content, executable=True, timeout=30)
        )
        assert result.exit_code == 0, f"write_file failed: {result.stderr}"
        assert elapsed < 15, (
            f"write_file took {elapsed:.1f}s — likely stdin EOF hang (expected <15s)"
        )

        # Verify content was written correctly
        read_result = await sandbox.run_command("cat /tmp/test.sh")
        assert read_result.exit_code == 0
        assert read_result.stdout == content

        # Verify executable bit
        stat_result = await sandbox.run_command("test -x /tmp/test.sh && echo yes")
        assert stat_result.stdout.strip() == "yes"

        print(f"\nwrite_file latency: {elapsed:.2f}s")
    finally:
        await sandbox.cleanup()


@requires_modal
@pytest.mark.asyncio
async def test_write_file_binary():
    """write_file should handle binary content correctly."""
    sandbox = await _create_sandbox(timeout=120)
    try:
        content = bytes(range(256))

        result, elapsed = await _timed(
            sandbox.write_file("/tmp/binary.bin", content, timeout=30)
        )
        assert result.exit_code == 0, f"write_file failed: {result.stderr}"
        assert elapsed < 15, (
            f"write_file took {elapsed:.1f}s — likely stdin EOF hang (expected <15s)"
        )

        # Verify size
        size_result = await sandbox.run_command("wc -c < /tmp/binary.bin")
        assert size_result.exit_code == 0
        assert int(size_result.stdout.strip()) == 256

        print(f"\nwrite_file (binary) latency: {elapsed:.2f}s")
    finally:
        await sandbox.cleanup()


@requires_modal
@pytest.mark.asyncio
async def test_run_command_vs_write_file_latency():
    """Compare run_command and write_file latency side-by-side.

    run_command (no stdin) has always been fast. write_file should now be
    comparable after the drain fix. This test documents the relative
    performance so regressions are easy to spot.
    """
    sandbox = await _create_sandbox(timeout=120)
    try:
        # Baseline: run_command (no stdin, known fast)
        _, run_cmd_elapsed = await _timed(
            sandbox.run_command("echo hello > /tmp/via_cmd.txt")
        )

        # Test: write_file (uses stdin + drain)
        content = "hello\n" * 100  # ~600 bytes
        _, write_file_elapsed = await _timed(
            sandbox.write_file("/tmp/via_write.txt", content, timeout=30)
        )

        print(f"\nrun_command latency:  {run_cmd_elapsed:.2f}s")
        print(f"write_file latency:   {write_file_elapsed:.2f}s")
        print(f"ratio:                {write_file_elapsed / max(run_cmd_elapsed, 0.001):.1f}x")

        # write_file may be slightly slower due to stdin overhead, but should
        # be in the same ballpark — not 60x slower
        assert write_file_elapsed < 15, (
            f"write_file took {write_file_elapsed:.1f}s vs run_command {run_cmd_elapsed:.1f}s "
            f"— likely stdin EOF hang"
        )
    finally:
        await sandbox.cleanup()
