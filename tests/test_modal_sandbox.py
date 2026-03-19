"""Smoke tests for ModalSandbox write_file latency.

Require Modal authentication and network access; skipped when Modal is not
configured locally (no MODAL_TOKEN_ID env var and no ~/.modal.toml).

The primary goal is to catch latency regressions in write_file — a previous
bug where a missing drain() after write_eof() caused hangs. For context,
run_command (no stdin) has always been fast; write_file should be comparable
after the drain fix, not 30-60x slower.
"""

import os
import time

import modal
import pytest
import pytest_asyncio

from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

_has_modal_auth = bool(
    os.environ.get("MODAL_TOKEN_ID")
    or os.path.exists(os.path.expanduser("~/.modal.toml"))
)

requires_modal = pytest.mark.skipif(not _has_modal_auth, reason="Modal not configured locally")

# Modal's debian_slim() defaults to the local Python version, which may not
# be supported. Pin to 3.12 for sandbox creation.
_MODAL_IMAGE = modal.Image.debian_slim(python_version="3.12")


@pytest_asyncio.fixture(scope="module")
async def sandbox():
    """Shared Modal sandbox for all tests in this module."""
    sb = await ModalSandbox.create(image=_MODAL_IMAGE, timeout=120)
    yield sb
    await sb.cleanup()


async def _timed(coro):
    """Await a coroutine and return (result, elapsed_seconds)."""
    start = time.monotonic()
    result = await coro
    return result, time.monotonic() - start


@requires_modal
@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_write_file_latency(sandbox):
    """write_file should complete in seconds, not minutes.

    Before the drain fix, write_file would hang for ~60s (the full exec timeout)
    because proc.stdin.write_eof() wasn't flushed. This test catches that
    regression by asserting a generous upper bound of 15s.
    """
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


@requires_modal
@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_write_file_binary(sandbox):
    """write_file should handle binary content correctly."""
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
