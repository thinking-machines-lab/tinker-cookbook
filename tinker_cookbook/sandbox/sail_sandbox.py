"""
Thin wrapper around Sail's Sailbox API.

Sail provides cloud-based sandboxed execution environments called Sailboxes.
Requires a Sail API key: `export SAIL_API_KEY=...`

Configuration via environment variables:
    SAIL_POOL_SIZE: Number of sandboxes in the pool (default: 32)
    SAIL_CREATION_RATE_LIMIT: Max sandboxes created per second (default: 4)

See: https://docs.sailresearch.com/sailbox-sdk
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import functools
import logging
import os
import shlex
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from typing import Any, TypeVar, cast

try:
    import sail
except ImportError:
    raise ImportError(
        "sail is required for SailSandbox. "
        "Install it with: uv pip install 'tinker-cookbook[sail] @ "
        "git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'"
    ) from None

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, SandboxResult]])


async def _read_stream_capped(stream: AsyncIterator[bytes], max_bytes: int) -> str:
    """Keep the first *max_bytes* of *stream* and drain the rest.

    A stream that is being read pauses the command when the reader falls behind,
    so the stream is always consumed to the end.
    """
    chunks: list[bytes] = []
    remaining = max_bytes
    async for chunk in stream:
        if remaining > 0:
            chunks.append(chunk[:remaining])
            remaining -= len(chunks[-1])
    return b"".join(chunks).decode("utf-8", errors="replace")


def _sail_errors_as_results(fn: _F) -> _F:
    """Map Sail errors the way the other backends do.

    A Sailbox that is gone raises ``SandboxTerminatedError``; any other Sail error,
    or a timeout, becomes a ``SandboxResult`` with exit code -1.
    """

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> SandboxResult:
        try:
            return await fn(*args, **kwargs)
        except (sail.SailboxTerminatedError, sail.SailboxHostLostError) as e:
            raise SandboxTerminatedError(str(e)) from e
        except (sail.SailError, TimeoutError) as e:
            return SandboxResult(stdout="", stderr=f"{type(e).__name__}: {e}", exit_code=-1)

    return cast(_F, wrapped)


class SailSandbox:
    """
    Persistent Sail sandbox for code execution. Conforms to SandboxInterface.

    Usage:
        sandbox = await SailSandbox.create()

        await sandbox.write_file("/workspace/code.py", "print('hello')")
        result = await sandbox.run_command("python /workspace/code.py")
        print(result.stdout)

        await sandbox.cleanup()
    """

    def __init__(self, sailbox: sail.Sailbox, max_stream_output_bytes: int = 128 * 1024) -> None:
        self._sailbox = sailbox
        self._max_stream_output_bytes = max_stream_output_bytes

    @classmethod
    async def create(
        cls,
        app_name: str = "tinker-cookbook-runner",
        image: sail.ImageDefinition | None = None,
        image_build_timeout: int = 1800,
        size: sail.SailboxSize = "s",
        memory_limit_gib: int | None = None,
        disk_limit_gib: int | None = None,
        egress_policy: sail.EgressPolicy | Mapping[str, Any] | str | None = None,
        max_lifetime_seconds: int | None = None,
        max_stream_output_bytes: int = 128 * 1024,
    ) -> SailSandbox:
        """Create a new Sailbox.

        Args:
            app_name: Sail app that owns the Sailbox; created on first use.
            image: Image to boot. Defaults to Sail's Debian base image.
            image_build_timeout: Seconds allowed for pulling or building the image before
                create fails.
            size: Resource size, "s", "m", or "l". "s" has the fastest cold starts.
            memory_limit_gib: Memory ceiling in GiB. Defaults to the size's own ceiling.
            disk_limit_gib: Disk ceiling in GiB. Defaults to the size's own ceiling.
            egress_policy: Outbound network policy, in any form sail.Sailbox.create accepts,
                such as sail.EgressPolicy.no_network(). None, the default, leaves outbound
                access unrestricted.
            max_lifetime_seconds: Hard cap on the Sailbox's lifetime. Sail terminates it once
                the cap elapses, even mid-command. None, the default, lets it live until
                cleanup(): an idle Sailbox sleeps, unbilled, and wakes on the next command.
            max_stream_output_bytes: Default cap on stdout and stderr returned by run_command.
        """
        app = await sail.App.find.aio(app_name, mint_if_missing=True)
        sailbox = await sail.Sailbox.create.aio(
            app=app,
            image=image,
            image_build_timeout=image_build_timeout,
            name=f"tinker-cookbook-{uuid.uuid4().hex[:12]}",
            size=size,
            memory_limit_gib=memory_limit_gib,
            disk_limit_gib=disk_limit_gib,
            egress_policy=egress_policy,
            max_lifetime_seconds=max_lifetime_seconds,
        )
        return cls(sailbox, max_stream_output_bytes=max_stream_output_bytes)

    @property
    def sandbox_id(self) -> str:
        return self._sailbox.sailbox_id

    @property
    def sailbox(self) -> sail.Sailbox:
        """The SDK handle, for Sail features that SandboxInterface does not cover."""
        return self._sailbox

    async def send_heartbeat(self, timeout: int = 30) -> None:
        """No-op: a Sailbox lives until cleanup(), or until an opt-in max_lifetime_seconds elapses.

        Sail may put an idle Sailbox to sleep, and wakes it on the next command.
        """

    @_sail_errors_as_results
    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox.

        Commands run under a login bash, like ModalSandbox, and as root, whatever ``USER`` the
        image sets. *workdir* must already exist. A command killed for exceeding *timeout*
        reports ``metrics={"timed_out": True}``.
        """
        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes
        async with await self._sailbox.exec.aio(
            ["bash", "-lc", command], cwd=workdir, timeout=timeout, user="0:0"
        ) as proc:
            stdout, stderr, result = await asyncio.gather(
                _read_stream_capped(proc.stdout_bytes, cap),
                _read_stream_capped(proc.stderr_bytes, cap),
                proc.wait(),
            )
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=result.exit_code,
            metrics={"timed_out": True} if result.timed_out else {},
        )

    @_sail_errors_as_results
    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox."""
        chunks: list[bytes] = []
        remaining = max_bytes
        async with asyncio.timeout(timeout), self._sailbox.fs.read_stream(path) as stream:
            async for chunk in stream:
                if remaining is not None:
                    chunk = chunk[:remaining]
                    remaining -= len(chunk)
                chunks.append(chunk)
                if remaining == 0:
                    break
        content = b"".join(chunks).decode("utf-8", errors="replace")
        return SandboxResult(stdout=content, stderr="", exit_code=0)

    @_sail_errors_as_results
    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Write content to a file in the sandbox, creating parent directories as needed."""
        async with asyncio.timeout(timeout):
            await self._sailbox.fs.write.aio(path, content, mode=0o755 if executable else 0o644)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    @_sail_errors_as_results
    async def write_files(
        self, files: Mapping[str, str | bytes], timeout: int = 60
    ) -> SandboxResult:
        """Write several files in one call, creating parent directories as needed."""
        async with asyncio.timeout(timeout):
            await self._sailbox.fs.write_files.aio(files)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def cleanup(self) -> None:
        """Terminate the Sailbox. Safe to call more than once."""
        await self._sailbox.terminate.aio()


class SailSandboxPool:
    """
    Pool of Sail sandboxes for concurrent execution.

    Each sandbox handles one request at a time. The pool manages
    borrowing and returning sandboxes automatically. Pooled Sailboxes have
    no lifetime cap: a warm one sleeps, unbilled, until it is used.

    Configuration via environment variables:
        SAIL_POOL_SIZE: Number of sandboxes in the pool (default: 32)
        SAIL_CREATION_RATE_LIMIT: Max sandboxes created per second (default: 4)
    """

    def __init__(
        self,
        *,
        pool_size: int | None = None,  # Number of warm sandboxes to maintain during the job run.
        sandbox_timeout_secs: int = 1200,  # Default timeout for a run_in_workdir command.
        image: sail.ImageDefinition | None = None,
        app_name: str = "tinker-cookbook-runner",
    ):
        self._pool_size = pool_size or int(os.getenv("SAIL_POOL_SIZE", "32"))
        self._creation_rate_limit = int(os.getenv("SAIL_CREATION_RATE_LIMIT", "4"))
        self._sandbox_timeout_secs = sandbox_timeout_secs
        self._image = image
        self._app_name = app_name
        self._terminated = False

        self._warm_pool: asyncio.Queue[SailSandbox] = asyncio.Queue()  # Warm pool of sandboxes.
        self._to_terminate: list[SailSandbox] = []  # Sandboxes pending termination.
        self._active_count = 0  # Number of in-use sandboxes.
        self._sandboxes: set[SailSandbox] = set()  # Every sandbox created and not yet terminated.
        # The creation failure that a retry cannot fix; once set, the pool creates nothing more.
        self._creation_error: sail.SailError | None = None

        self._maintainer = asyncio.create_task(self._maintain_pool())
        # Pooled Sailboxes have no lifetime cap, so nothing else reclaims them when the
        # process exits without awaiting terminate().
        atexit.register(self._terminate_at_exit)

    async def _create(self) -> SailSandbox:
        return await SailSandbox.create(app_name=self._app_name, image=self._image)

    async def _maintain_pool(self) -> None:
        """Background task to handle all sandbox creation and termination."""
        while not self._terminated:
            try:
                await self._maintain_pool_step()
            except Exception as e:
                logger.error(f"Error maintaining SailSandboxPool: {e}")
            await asyncio.sleep(1.0)

    async def _maintain_pool_step(self) -> None:
        """Terminate used sandboxes, then top up the pool unless creation cannot succeed."""
        # Batch terminate used sandboxes
        if self._to_terminate:
            to_terminate, self._to_terminate = self._to_terminate, []
            await self._cleanup(to_terminate)

        if self._creation_error is not None:
            return

        # Create new sandboxes in parallel (respecting rate limit)
        total = self._warm_pool.qsize() + self._active_count
        need = min(self._creation_rate_limit, self._pool_size - total)
        if need > 0:
            await asyncio.gather(*(self._add_sandbox() for _ in range(need)))

    async def _add_sandbox(self) -> None:
        """Create one sandbox and own it as soon as it exists, whether or not the other
        creations in its batch have finished."""
        try:
            sandbox = await self._create()
        except Exception as e:
            logger.error(f"Error creating Sail sandbox: {e}")
            if isinstance(e, sail.SailError) and not e.retryable:
                self._creation_error = e
            return
        self._sandboxes.add(sandbox)
        await self._warm_pool.put(sandbox)

    async def _cleanup(self, sandboxes: list[SailSandbox]) -> None:
        """Terminate sandboxes, keeping any whose termination may succeed on a retry."""
        results = await asyncio.gather(*(sb.cleanup() for sb in sandboxes), return_exceptions=True)
        for sandbox, result in zip(sandboxes, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Error terminating Sail sandbox {sandbox.sandbox_id}: {result}")
                if isinstance(result, sail.SailError) and result.retryable:
                    self._to_terminate.append(sandbox)
                    continue
            self._sandboxes.discard(sandbox)

    async def _acquire(self) -> SailSandbox:
        """Take a warm sandbox as active, failing instead of waiting while none can be created."""
        while True:
            if self._terminated:
                raise SandboxError("SailSandboxPool has been terminated.")
            if self._creation_error is not None and self._warm_pool.empty():
                raise SandboxError(
                    f"SailSandboxPool cannot create sandboxes: {self._creation_error}"
                ) from self._creation_error
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(1.0):
                    sandbox = await self._warm_pool.get()
                # asyncio.timeout awaits the queue in this task, so no other task runs
                # between taking the sandbox and counting it, and terminate() always
                # sees every sandbox as either warm or active.
                self._active_count += 1
                return sandbox

    def _release(self, sandbox: SailSandbox) -> None:
        """Return a used sandbox to the maintainer for termination."""
        self._active_count -= 1
        self._to_terminate.append(sandbox)

    async def run_in_workdir(
        self,
        files: Mapping[str, str | bytes],
        command: list[str],
        timeout: int | None = None,
    ) -> SandboxResult:
        """
        Execute command with files using an available sandbox from the pool.
        If all sandboxes are busy, waits until one becomes available.

        Creates an isolated workdir, writes files, and runs the command.

        Args:
            files: Files to write {filename: content}
            command: Command and arguments (e.g., ["python", "run.py"])
            timeout: Execution timeout in seconds

        Raises:
            SandboxError: The pool was terminated, including while this call was waiting for
                a sandbox, or Sail rejected sandbox creation with an error that a retry cannot
                fix (for example a bad API key or image).
        """
        sandbox = await self._acquire()
        try:
            workdir = f"/workspace/{uuid.uuid4().hex[:12]}"
            if files:
                # Writing the files also creates the workdir.
                result = await sandbox.write_files(
                    {f"{workdir}/{filename}": content for filename, content in files.items()}
                )
            else:
                result = await sandbox.run_command(
                    f"mkdir -p {shlex.quote(workdir)}", timeout=timeout or 60
                )
            if result.exit_code != 0:
                return result
            return await sandbox.run_command(
                shlex.join(command), workdir=workdir, timeout=timeout or self._sandbox_timeout_secs
            )
        finally:
            self._release(sandbox)

    async def terminate(self) -> None:
        """Exit the pool and terminate all sandboxes.

        A sandbox whose termination fails with an error a retry may fix is terminated
        again when the process exits.
        """
        self._terminated = True

        # Let an in-flight maintenance step finish so the sandboxes it is
        # creating land in the warm pool and get terminated below.
        await self._maintainer

        # Wait for active sandboxes to finish and be added to _to_terminate
        while self._active_count > 0:
            await asyncio.sleep(0.5)

        # Collect and terminate all sandboxes
        all_sandboxes, self._to_terminate = self._to_terminate, []
        while not self._warm_pool.empty():
            try:
                all_sandboxes.append(self._warm_pool.get_nowait())
            except asyncio.QueueEmpty:
                break
        await self._cleanup(all_sandboxes)

    def _terminate_at_exit(self) -> None:
        """Terminate the sandboxes still alive, with the event loop already gone."""
        self._terminated = True
        sandboxes, self._sandboxes = self._sandboxes, set()
        for sandbox in sandboxes:
            try:
                sandbox.sailbox.terminate()
            except sail.SailError as e:
                logger.error(f"Error terminating Sail sandbox {sandbox.sandbox_id}: {e}")
