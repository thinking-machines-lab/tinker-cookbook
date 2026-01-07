"""
Thin wrapper around Modal Sandbox API.

Modal provides cloud-based sandboxed execution environments.
Requires Modal authentication: `modal token new`

Configuration via environment variables:
    MODAL_POOL_SIZE: Number of sandboxes in the pool (default: 32)

See: https://modal.com/docs/guide/sandbox
"""

from __future__ import annotations

import asyncio
import os
import uuid

import modal


class ModalSandbox:
    """
    Persistent Modal sandbox for code execution.

    Usage:
        sandbox = ModalSandbox()

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
        self,
        app_name: str = "tinker-cookbook-runner",
        default_timeout: int = 240,
        image: modal.Image | None = None,
    ):
        self._app_name = app_name
        self._default_timeout = default_timeout
        self._image = image or modal.Image.debian_slim()

        # Create the Modal sandbox
        self._app = modal.App.lookup(self._app_name, create_if_missing=True)
        self._sandbox = modal.Sandbox.create(app=self._app, image=self._image)

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
        timeout = timeout if timeout is not None else self._default_timeout

        try:
            proc = await self._sandbox.exec.aio(*args, workdir=workdir, timeout=timeout)
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

        Creates a unique workdir, writes files, runs command, then cleans up.

        Args:
            files: Files to write {filename: content}
            command: Command and arguments (e.g., ["python", "run.py"])
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        workdir = f"/workspace/{uuid.uuid4().hex[:12]}"
        try:
            # Create workdir and check return code
            proc = await self._sandbox.exec.aio("mkdir", "-p", workdir)
            ret = await proc.wait.aio()
            if ret != 0:
                return ret, "", f"Failed to create workdir: {workdir}"

            for filename, content in files.items():
                await self.write_file(f"{workdir}/{filename}", content)
            exit_code, stdout, stderr = await self.exec(*command, workdir=workdir, timeout=timeout)
            return exit_code, stdout, stderr
        finally:
            proc = await self._sandbox.exec.aio("rm", "-rf", workdir)
            await proc.wait.aio()


class ModalSandboxPool:
    """
    Pool of Modal sandboxes for concurrent execution.

    Each sandbox handles one request at a time. The pool manages
    borrowing and returning sandboxes automatically.
    """

    def __init__(
        self,
        pool_size: int | None = None,
        image: modal.Image | None = None,
        app_name: str = "tinker-cookbook-runner",
        default_timeout: int = 240,
    ):
        self._pool_size = pool_size or int(os.getenv("MODAL_POOL_SIZE", "32"))
        self._image = image
        self._app_name = app_name
        self._default_timeout = default_timeout

        # Fill pool with sandbox instances
        self._sandboxes = [
            ModalSandbox(
                app_name=app_name,
                image=image,
                default_timeout=default_timeout,
            )
            for _ in range(self._pool_size)
        ]

        # Queue for borrowing/returning sandboxes
        self._queue: asyncio.Queue[ModalSandbox] = asyncio.Queue()
        for sb in self._sandboxes:
            self._queue.put_nowait(sb)

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
        sandbox = await self._queue.get()
        try:
            return await sandbox.run_in_workdir(files, command, timeout)
        finally:
            await self._queue.put(sandbox)
