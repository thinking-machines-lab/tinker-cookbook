"""Smoke tests for TensorlakeSandbox and TensorlakeSandboxPool.

Require a Tensorlake API key and network access; skipped when
TENSORLAKE_API_KEY is not set.
"""

import asyncio
import os

import pytest
import pytest_asyncio

pytest.importorskip("tensorlake")

from tinker_cookbook.sandbox import SandboxInterface, SandboxTerminatedError
from tinker_cookbook.sandbox.tensorlake_sandbox import (
    TensorlakeSandbox,
    TensorlakeSandboxPool,
    _is_sandbox_terminated,
)

requires_tensorlake = pytest.mark.skipif(
    not os.environ.get("TENSORLAKE_API_KEY"), reason="TENSORLAKE_API_KEY not set"
)


def test_terminated_detection():
    """A missing file must not look like a dead sandbox."""
    assert _is_sandbox_terminated(
        Exception('API error (status 404): {"error":"Sandbox \'abc\' not found or not running"}')
    )
    assert not _is_sandbox_terminated(
        Exception('API error (status 404): {"error":"File not found: /sandbox/x"}')
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sandbox():
    """Shared Tensorlake sandbox for all tests in this module."""
    sb = await TensorlakeSandbox.create(timeout=300)
    yield sb
    await sb.cleanup()


@requires_tensorlake
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(30)
async def test_conforms_to_interface(sandbox):
    assert isinstance(sandbox, SandboxInterface)
    assert sandbox.sandbox_id


@requires_tensorlake
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(30)
async def test_run_command(sandbox):
    result = await sandbox.run_command("echo out; echo err >&2; exit 3")
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"
    assert result.exit_code == 3

    result = await sandbox.run_command("pwd", workdir="/tmp")
    assert result.stdout.strip() == "/tmp"


@requires_tensorlake
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(30)
async def test_run_command_timeout_and_output_cap(sandbox):
    result = await sandbox.run_command("sleep 10", timeout=1)
    assert result.exit_code != 0

    result = await sandbox.run_command("head -c 5000 /dev/zero | tr '\\0' a", max_output_bytes=100)
    assert len(result.stdout) == 100


@requires_tensorlake
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(30)
async def test_write_and_read_file(sandbox):
    content = "#!/bin/bash\necho hello world\n"
    result = await sandbox.write_file("/tmp/new/dir/test.sh", content, executable=True)
    assert result.exit_code == 0, result.stderr

    read = await sandbox.read_file("/tmp/new/dir/test.sh")
    assert read.exit_code == 0
    assert read.stdout == content

    read = await sandbox.read_file("/tmp/new/dir/test.sh", max_bytes=4)
    assert read.stdout == "#!/b"

    run = await sandbox.run_command("/tmp/new/dir/test.sh")
    assert run.stdout.strip() == "hello world"

    binary = bytes(range(256))
    assert (await sandbox.write_file("/tmp/binary.bin", binary)).exit_code == 0
    size = await sandbox.run_command("wc -c < /tmp/binary.bin")
    assert int(size.stdout.strip()) == 256

    missing = await sandbox.read_file("/does/not/exist")
    assert missing.exit_code != 0


@requires_tensorlake
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(60)
async def test_service_and_port(sandbox):
    """A background HTTP server in the sandbox is reachable through open_port."""
    import urllib.request

    # /srv is not writable by the default image user, so this also tests the root fallback.
    assert (await sandbox.write_file("/srv/index.html", "ok")).exit_code == 0
    await sandbox.start_process("python3 -m http.server 8080", workdir="/srv")

    url = await sandbox.open_port(8080)
    body = ""
    for _ in range(20):
        try:
            body = await asyncio.to_thread(
                lambda: urllib.request.urlopen(f"{url}/index.html", timeout=5).read().decode()
            )
            break
        except Exception:
            await asyncio.sleep(0.5)
    assert body == "ok"


@requires_tensorlake
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(120)
async def test_checkpoint_and_restore(sandbox):
    await sandbox.write_file("/tmp/state.txt", "saved")
    snapshot_id = await sandbox.checkpoint()

    restored = await TensorlakeSandbox.create(snapshot_id=snapshot_id, timeout=120)
    try:
        read = await restored.read_file("/tmp/state.txt")
        assert read.stdout == "saved"
    finally:
        await restored.cleanup()


@requires_tensorlake
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(120)
async def test_filesystem_checkpoint_and_restore(sandbox):
    """A filesystem-only snapshot keeps files but not running processes."""
    await sandbox.write_file("/tmp/fs_state.txt", "saved")
    snapshot_id = await sandbox.checkpoint(include_memory=False)

    restored = await TensorlakeSandbox.create(snapshot_id=snapshot_id, timeout=120)
    try:
        read = await restored.read_file("/tmp/fs_state.txt")
        assert read.stdout == "saved"
    finally:
        await restored.cleanup()


@requires_tensorlake
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(30)
async def test_cleanup_twice_and_dead_sandbox():
    sb = await TensorlakeSandbox.create(timeout=60)
    await sb.cleanup()
    await sb.cleanup()
    with pytest.raises(SandboxTerminatedError):
        await sb.run_command("true")


@requires_tensorlake
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(180)
async def test_pool_run_in_workdir():
    pool = TensorlakeSandboxPool(
        max_concurrency=8, setup_command="pip install -q --break-system-packages numpy"
    )
    try:
        results = await asyncio.gather(
            *(
                pool.run_in_workdir(
                    files={"run.py": f"import numpy; print({i} * 2)"},
                    command=["python3", "run.py"],
                    timeout=60,
                )
                for i in range(16)
            )
        )
        for i, r in enumerate(results):
            assert r.exit_code == 0, r.stderr
            assert r.stdout.strip() == str(i * 2)
    finally:
        await pool.terminate()
