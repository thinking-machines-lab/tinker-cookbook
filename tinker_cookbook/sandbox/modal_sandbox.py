"""
Thin wrapper around Modal Sandbox API.

Modal provides cloud-based sandboxed execution environments.
Requires Modal authentication: `modal token new`

Configuration via environment variables:
    MODAL_POOL_SIZE: Number of sandboxes in the pool (default: 32)
    MODAL_CREATION_RATE_LIMIT: Max sandboxes created per second (default: 4)

See: https://modal.com/docs/guide/sandbox
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import uuid

import modal

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

logger = logging.getLogger(__name__)


async def _read_stream_capped(stream: object, max_bytes: int) -> str:
    """Read a Modal async stream up to *max_bytes*, draining the rest to avoid blocking."""
    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in stream:  # type: ignore[union-attr]
            if isinstance(chunk, str):
                chunk = chunk.encode()
            remaining = max_bytes - total
            if remaining <= 0:
                break
            chunks.append(chunk[:remaining])
            total += len(chunk[:remaining])
    except UnicodeDecodeError:
        pass  # Modal internal decoding error — return what we have

    # Drain any remaining data so the process can exit cleanly
    try:
        async for _ in stream:  # type: ignore[union-attr]
            pass
    except (UnicodeDecodeError, Exception):
        pass

    return b"".join(chunks).decode("utf-8", errors="replace")


def _is_sandbox_terminated(e: BaseException) -> bool:
    """Check if an exception indicates the sandbox has died."""
    type_name = type(e).__name__
    if type_name == "NotFoundError":
        return True
    msg = str(e).lower()
    return any(keyword in msg for keyword in ("terminated", "died", "not found"))


class ModalSandbox:
    """
    Persistent Modal sandbox for code execution. Conforms to SandboxInterface.

    Usage:
        sandbox = await ModalSandbox.create()

        await sandbox.write_file("/workspace/code.py", "print('hello')")
        result = await sandbox.run_command("python /workspace/code.py")
        print(result.stdout)

        await sandbox.cleanup()
    """

    def __init__(
        self,
        timeout: int,
        image: modal.Image,
        app: modal.App,
        sandbox: modal.Sandbox,
        max_stream_output_bytes: int = 128 * 1024,
    ) -> None:
        self._timeout = timeout  # Timeout for the entire Sandbox instance
        self._image = image
        self._app = app
        self._sandbox = sandbox
        self._max_stream_output_bytes = max_stream_output_bytes

    @classmethod
    async def create(
        cls,
        app_name: str = "tinker-cookbook-runner",
        timeout: int = 600,
        image: modal.Image | None = None,
        max_stream_output_bytes: int = 128 * 1024,
    ) -> ModalSandbox:
        """Create a new Modal sandbox."""
        image = image or modal.Image.debian_slim()
        app = await modal.App.lookup.aio(app_name, create_if_missing=True)
        sandbox = await modal.Sandbox.create.aio(app=app, image=image, timeout=timeout)
        return cls(
            timeout=timeout,
            image=image,
            app=app,
            sandbox=sandbox,
            max_stream_output_bytes=max_stream_output_bytes,
        )

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.object_id

    async def send_heartbeat(self) -> None:
        await self._sandbox.exec.aio("true")

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox."""
        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes
        try:
            kwargs: dict[str, object] = {"timeout": timeout}
            if workdir is not None:
                kwargs["workdir"] = workdir
            proc = await self._sandbox.exec.aio("bash", "-lc", command, **kwargs)
            stdout, stderr, exit_code = await asyncio.gather(
                _read_stream_capped(proc.stdout, cap),
                _read_stream_capped(proc.stderr, cap),
                proc.wait.aio(),
            )
            return SandboxResult(stdout=stdout, stderr=stderr, exit_code=exit_code)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox."""
        if max_bytes is not None:
            cmd = f"head -c {max_bytes} {shlex.quote(path)}"
        else:
            cmd = f"cat {shlex.quote(path)}"
        return await self.run_command(cmd, timeout=timeout)

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Write content to a file in the sandbox."""
        if isinstance(content, str):
            content = content.encode()

        dir_path = os.path.dirname(path)
        quoted_path = shlex.quote(path)

        cmd = f"mkdir -p {shlex.quote(dir_path)} && cat > {quoted_path}"
        if executable:
            cmd += f" && chmod +x {quoted_path}"

        try:
            proc = await self._sandbox.exec.aio("bash", "-lc", cmd, timeout=timeout)

            # Write content in 2 MiB chunks via stdin
            chunk_size = 2 * 1024 * 1024
            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]
                try:
                    proc.stdin.write(chunk)
                except TypeError:
                    proc.stdin.write(chunk.decode("utf-8", errors="replace"))
                await proc.stdin.drain.aio()
            proc.stdin.write_eof()

            stdout, stderr, exit_code = await asyncio.gather(
                _read_stream_capped(proc.stdout, self._max_stream_output_bytes),
                _read_stream_capped(proc.stderr, self._max_stream_output_bytes),
                proc.wait.aio(),
            )
            return SandboxResult(stdout=stdout, stderr=stderr, exit_code=exit_code)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Terminate the Modal sandbox and wait for it to fully shut down."""
        await self._sandbox.terminate.aio()
        await self._sandbox.wait.aio(raise_on_termination=False)


class ModalSandboxPool:
    """
    Pool of Modal sandboxes for concurrent execution.

    Each sandbox handles one request at a time. The pool manages
    borrowing and returning sandboxes automatically.

    Configuration via environment variables:
        MODAL_POOL_SIZE: Number of sandboxes in the pool (default: 32)
        MODAL_CREATION_RATE_LIMIT: Max sandboxes created per second (default: 4)
    """

    def __init__(
        self,
        *,
        pool_size: int | None = None,  # Number of warm sandboxes to maintain during the job run.
        sandbox_timeout_secs: int = 1200,  # Time after which a sandbox is terminated.
        image: modal.Image | None = None,
        app_name: str = "tinker-cookbook-runner",
    ):
        self._pool_size = pool_size or int(os.getenv("MODAL_POOL_SIZE", "32"))
        self._creation_rate_limit = int(os.getenv("MODAL_CREATION_RATE_LIMIT", "4"))
        self._sandbox_timeout_secs = sandbox_timeout_secs
        self._image = image
        self._app_name = app_name
        self._terminated = False

        self._warm_pool: asyncio.Queue[ModalSandbox] = asyncio.Queue()  # Warm pool of sandboxes.
        self._to_terminate: list[ModalSandbox] = []  # Sandboxes pending termination.
        self._active_count = 0  # Number of in-use sandboxes.

        asyncio.create_task(self._maintain_pool())

    async def _create(self) -> ModalSandbox:
        return await ModalSandbox.create(
            app_name=self._app_name, timeout=self._sandbox_timeout_secs, image=self._image
        )

    async def _maintain_pool(self) -> None:
        """Background task to handle all sandbox creation and termination."""
        while not self._terminated:
            try:
                await self._maintain_pool_step()
            except Exception as e:
                logger.error(f"Error maintaining ModalSandboxPool: {e}")
            await asyncio.sleep(1.0)

    async def _maintain_pool_step(self) -> None:
        """Single iteration of pool maintenance: terminate used sandboxes, create new ones."""
        # Batch terminate used sandboxes
        if self._to_terminate:
            to_terminate, self._to_terminate = self._to_terminate, []
            await asyncio.gather(*(sb.cleanup() for sb in to_terminate))

        # Create new sandboxes in parallel (respecting rate limit)
        total = self._warm_pool.qsize() + self._active_count
        need = min(self._creation_rate_limit, self._pool_size - total)
        if need > 0:
            new_sandboxes = await asyncio.gather(
                *(self._create() for _ in range(need)),
                return_exceptions=True,
            )
            for sb in new_sandboxes:
                if isinstance(sb, BaseException):
                    logger.error(f"Error creating Modal sandbox: {sb}")
                else:
                    await self._warm_pool.put(sb)

    async def run_in_workdir(
        self,
        files: dict[str, str],
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
        """
        if self._terminated:
            raise RuntimeError("ModalSandboxPool has been terminated.")

        sandbox = await self._warm_pool.get()
        self._active_count += 1

        try:
            workdir = f"/workspace/{uuid.uuid4().hex[:12]}"
            result = await sandbox.run_command(
                f"mkdir -p {shlex.quote(workdir)}", timeout=timeout or 60
            )
            if result.exit_code != 0:
                return SandboxResult(
                    stdout="",
                    stderr=f"Failed to create workdir: {workdir}",
                    exit_code=result.exit_code,
                )

            if files:
                await asyncio.gather(
                    *(
                        sandbox.write_file(f"{workdir}/{filename}", content)
                        for filename, content in files.items()
                    )
                )
            return await sandbox.run_command(
                shlex.join(command), workdir=workdir, timeout=timeout or self._sandbox_timeout_secs
            )
        finally:
            self._active_count -= 1
            self._to_terminate.append(sandbox)

    async def terminate(self) -> None:
        """Exit the pool and terminate all sandboxes."""
        self._terminated = True

        # Wait for active sandboxes to finish and be added to _to_terminate
        while self._active_count > 0:
            await asyncio.sleep(0.5)

        # Collect and terminate all sandboxes
        all_sandboxes = list(self._to_terminate)
        while not self._warm_pool.empty():
            try:
                all_sandboxes.append(self._warm_pool.get_nowait())
            except asyncio.QueueEmpty:
                break
        await asyncio.gather(*(sb.cleanup() for sb in all_sandboxes))
