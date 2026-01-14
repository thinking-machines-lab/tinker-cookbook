"""
Thin wrapper around Modal Sandbox API.

Modal provides cloud-based sandboxed execution environments.
Requires Modal authentication: `modal token new`

Configuration via environment variables:
    MODAL_POOL_SIZE: Number of sandboxes in the warm pool (default: 32)

See: https://modal.com/docs/guide/sandbox
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import modal


class ModalSandbox:
    """
    Persistent Modal sandbox for code execution.

    Usage:
        sandbox = await ModalSandbox.create()

        # Manual file write and exec:
        await sandbox.write_file("/workspace/code.py", "print('hello')")
        exit_code, stdout, stderr = await sandbox.exec("python", "code.py", workdir="/workspace")

        # Run in isolated workdir with files:
        exit_code, stdout, stderr = await sandbox.run_in_workdir(
            files={"code.py": "print('hello')"},
            command=["python", "code.py"],
        )
    """

    def __init__(
        self, timeout: int, image: modal.Image, app: modal.App, sandbox: modal.Sandbox
    ) -> None:
        self._timeout = timeout  # Timeout for the entire Sandbox instance
        self._image = image
        self._app = app
        self._sandbox = sandbox
        self._created_at = asyncio.get_running_loop().time()

    @classmethod
    async def create(
        cls,
        app_name: str = "tinker-cookbook-runner",
        timeout: int = 600,
        image: modal.Image | None = None,
    ) -> ModalSandbox:
        # Create the Modal sandbox
        image = image or modal.Image.debian_slim()
        app = await modal.App.lookup.aio(app_name, create_if_missing=True)
        sandbox = await modal.Sandbox.create.aio(app=app, image=image, timeout=timeout)
        return cls(timeout=timeout, image=image, app=app, sandbox=sandbox)

    async def write_file(self, path: str, content: str) -> None:
        """Write a file into the sandbox filesystem."""
        async with await self._sandbox.open.aio(path, "w") as f:
            await f.write.aio(content)

    async def exec(
        self,
        *args: str,
        workdir: str = "/workspace",
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """
        Execute a command in the sandbox.

        Args:
            *args: Command and arguments (e.g., "python", "script.py")
            workdir: Working directory for the command
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        try:
            proc = await self._sandbox.exec.aio(
                *args, workdir=workdir, timeout=timeout or self._timeout
            )
            stdout = await proc.stdout.read.aio()
            stderr = await proc.stderr.read.aio()
            exit_code = await proc.wait.aio()

            # Decode bytes to str if needed
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

            return exit_code, stdout, stderr
        except Exception as e:
            return -1, "", str(e)

    async def run_in_workdir(
        self,
        files: dict[str, str],
        command: list[str],
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """
        Execute a command in an isolated workdir with the given files.

        Args:
            files: Files to write {filename: content}
            command: Command and arguments (e.g., ["python", "run.py"])
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        workdir = f"/workspace/{uuid.uuid4().hex[:12]}"

        proc = await self._sandbox.exec.aio("mkdir", "-p", workdir)
        ret = await proc.wait.aio()
        if ret != 0:
            return ret, "", f"Failed to create workdir: {workdir}"

        if files:
            await asyncio.gather(
                *(
                    self.write_file(f"{workdir}/{filename}", content)
                    for filename, content in files.items()
                )
            )
        exit_code, stdout, stderr = await self.exec(*command, workdir=workdir, timeout=timeout)
        return exit_code, stdout, stderr

    async def terminate(self) -> None:
        """Terminate the Modal sandbox."""
        await self._sandbox.terminate.aio()


class ModalSandboxPool:
    """
    Pool of Modal sandboxes for concurrent execution.

    Each sandbox handles one request at a time. The pool manages
    borrowing and returning sandboxes automatically.
    """

    def __init__(
        self,
        *,
        pool_size: int | None = None,  # Number of warm sandboxes to maintain during the job run.
        pool_recycle_timeout_secs: float = 600,  # Time after which a sandbox is too old and removed from pool.
        pool_sandbox_timeout_secs: float = 1200,  # Time after which a sandbox is terminated.
        image: modal.Image | None = None,
        app_name: str = "tinker-cookbook-runner",
    ):
        self._pool_size = pool_size or int(os.getenv("MODAL_POOL_SIZE", "32"))

        # The difference between these timeouts is the guaranteed "remaining
        # lifetime" of a sandbox when borrowed from the pool.
        self._pool_recycle_timeout_secs = pool_recycle_timeout_secs
        self._pool_sandbox_timeout_secs = pool_sandbox_timeout_secs

        self._image = image
        self._app_name = app_name
        self._terminated = False

        self._warm_pool: list[ModalSandbox] = []
        asyncio.create_task(self._maintain_pool())

    async def _create(self) -> ModalSandbox:
        return await ModalSandbox.create(
            app_name=self._app_name, timeout=self._pool_sandbox_timeout_secs, image=self._image
        )

    async def _maintain_pool(self) -> None:
        while not self._terminated:
            try:
                await self._maintain_pool_step()
            except Exception as e:
                print(f"Error maintaining ModalSandboxPool: {e}", file=sys.stderr)
            await asyncio.sleep(2.0)

    async def _maintain_pool_step(self) -> None:
        # Atomically recycle old sandboxes, removing them from the pool.
        now = asyncio.get_running_loop().time()
        new_pool: list[ModalSandbox] = []
        for sandbox in self._warm_pool:
            age = now - sandbox._created_at
            if age < self._pool_recycle_timeout_secs:
                new_pool.append(sandbox)
            else:
                # Sandbox is too old, remove from pool and terminate so we don't
                # continue paying for it unnecessarily.
                asyncio.create_task(sandbox.terminate())
        self._warm_pool = new_pool

        if len(self._warm_pool) >= self._pool_size or self._terminated:
            return

        # Fill pool with new sandbox instances
        new_sandboxes = await asyncio.gather(
            *(self._create() for _ in range(self._pool_size - len(self._warm_pool))),
            return_exceptions=True,
        )
        failures: list[BaseException] = []
        for sb in new_sandboxes:
            if isinstance(sb, BaseException):
                failures.append(sb)
            else:
                self._warm_pool.append(sb)
        if failures:
            raise BaseExceptionGroup(f"Errors creating {len(failures)} Modal sandboxes", failures)

    async def run_in_workdir(
        self,
        files: dict[str, str],
        command: list[str],
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """
        Execute command with files using an available sandbox from the pool.
        If all sandboxes are busy, waits until one becomes available.

        Args:
            files: Files to write {filename: content}
            command: Command and arguments (e.g., ["python", "run.py"])
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        if self._terminated:
            raise RuntimeError("ModalSandboxPool has been terminated and cannot run new tasks.")
        if self._warm_pool:
            sandbox = self._warm_pool.pop(0)
        else:
            sandbox = await self._create()
        try:
            return await sandbox.run_in_workdir(files, command, timeout)
        finally:
            asyncio.create_task(sandbox.terminate())  # Don't reuse sandboxes after intial use

    async def terminate(self) -> None:
        """Exit the pool and terminate all sandboxes."""
        pool = self._warm_pool
        self._terminated = True
        self._warm_pool = []
        await asyncio.gather(*(sb.terminate() for sb in pool))
