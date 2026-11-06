import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tinker_cookbook.sandbox.daytona_sandbox import (
    _CAPTURE_SCRIPT,
    _RUN_SCRIPT,
    _cancellation_as_terminated,
    _task_resources,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxTerminatedError

# The script run_command executes in the sandbox, run here with the local bash and coreutils.
needs_local_shell = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None or shutil.which("timeout") is None,
    reason="needs Linux with bash and coreutils",
)


def _run_script(
    command: str, timeout: int = 10, cap: int = 1000
) -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.monotonic()
    result = subprocess.run(
        ["bash", "-c", _RUN_SCRIPT, "run_command", str(timeout), command, str(cap)]
        + [_CAPTURE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, time.monotonic() - start


@needs_local_shell
def test_script_returns_output_and_exit_code_without_waiting() -> None:
    result, elapsed = _run_script("echo out; echo err >&2; exit 3")
    assert (result.returncode, result.stdout, result.stderr) == (3, "out\n", "err\n")
    assert elapsed < 0.9  # nothing holds the output, so no grace period


@needs_local_shell
def test_script_caps_each_stream() -> None:
    result, _ = _run_script("yes | head -c 5000; yes | head -c 5000 >&2", cap=1000)
    assert (result.returncode, len(result.stdout), len(result.stderr)) == (0, 1000, 1000)


@needs_local_shell
def test_script_returns_while_a_background_job_holds_the_output(tmp_path: Path) -> None:
    # The background bash running ``cd /tmp && sleep 30 ...`` has the command's output as
    # its own, so the output stays open while sleep runs.
    pid_file = tmp_path / "job.pid"
    result, elapsed = _run_script(
        f"cd /tmp && sleep 30 > /dev/null 2>&1 & echo $! > {pid_file}\necho started"
    )
    pid = int(pid_file.read_text())
    try:
        assert (result.returncode, result.stdout) == (0, "started\n")
        assert elapsed < 5
        os.kill(pid, 0)  # the background job is still running
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)


@needs_local_shell
def test_script_kills_the_command_at_its_timeout() -> None:
    result, elapsed = _run_script("sleep 30", timeout=1)
    assert result.returncode == 124
    assert elapsed < 10


@_cancellation_as_terminated
async def _cancelled_inside() -> None:
    raise asyncio.CancelledError


def test_inner_cancellation_becomes_sandbox_terminated() -> None:
    with pytest.raises(SandboxTerminatedError):
        asyncio.run(_cancelled_inside())


def test_outer_cancellation_propagates() -> None:
    @_cancellation_as_terminated
    async def slow() -> None:
        await asyncio.sleep(10)

    async def main() -> None:
        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())


def test_task_resources_from_legacy_sizes(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text('[environment]\ncpus = 2\nmemory = "4G"\nstorage = "10240M"\n')
    resources = _task_resources(task_toml)
    assert resources is not None
    assert (resources.cpu, resources.memory, resources.disk) == (2, 4, 10)


def test_task_resources_round_up_to_whole_units(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text("[environment]\ncpus = 0.5\nmemory_mb = 1536\n")
    resources = _task_resources(task_toml)
    assert resources is not None
    assert (resources.cpu, resources.memory, resources.disk) == (1, 2, None)


def test_task_resources_absent(tmp_path: Path) -> None:
    assert _task_resources(tmp_path / "missing.toml") is None
    task_toml = tmp_path / "task.toml"
    task_toml.write_text('version = "1.0"\n')
    assert _task_resources(task_toml) is None
