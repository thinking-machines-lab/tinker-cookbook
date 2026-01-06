"""
Thin wrapper around Modal Sandbox API.

Modal provides cloud-based sandboxed execution environments.
Requires Modal authentication: `modal token new`

See: https://modal.com/docs/guide/sandbox
"""

from __future__ import annotations

import asyncio
import uuid

import modal


class ModalSandbox:
    """
    Persistent Modal sandbox for code execution.

    Usage:
        sandbox = ModalSandbox()

        # Manual file write and exec:
        sandbox.write_file("/workspace/code.py", "print('hello')")
        exit_code, stdout, stderr = sandbox.exec("python", "code.py", workdir="/workspace")

        # Run in isolated workdir with files:
        exit_code, stdout, stderr = sandbox.run_in_workdir(
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

    def write_file(self, path: str, content: str) -> None:
        """Write a file into the sandbox filesystem."""
        with self._sandbox.open(path, "w") as f:
            f.write(content)

    def exec(
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
            proc = self._sandbox.exec(*args, workdir=workdir, timeout=timeout)
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            exit_code = proc.wait()

            # Decode bytes to str if needed
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

            return exit_code, stdout, stderr
        except Exception as e:
            return -1, "", str(e)

    def run_in_workdir(
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
            self._sandbox.exec("mkdir", "-p", workdir).wait()
            for filename, content in files.items():
                self.write_file(f"{workdir}/{filename}", content)
            exit_code, stdout, stderr = self.exec(*command, workdir=workdir, timeout=timeout)
            return exit_code, stdout, stderr
        finally:
            self._sandbox.exec("rm", "-rf", workdir).wait()


class ModalSandboxPool:
    """
    Pool of Modal sandboxes for concurrent execution.

    Each sandbox handles one request at a time. The pool manages
    borrowing and returning sandboxes automatically.
    """

    def __init__(
        self,
        pool_size: int = 8,
        image: modal.Image | None = None,
        app_name: str = "tinker-cookbook-runner",
        default_timeout: int = 240,
    ):
        self._pool_size = pool_size
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
            for _ in range(pool_size)
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
            return await asyncio.to_thread(sandbox.run_in_workdir, files, command, timeout)
        finally:
            await self._queue.put(sandbox)
