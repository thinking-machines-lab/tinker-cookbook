"""
Thin wrapper around Modal Sandbox API.

Modal provides cloud-based sandboxed execution environments.
Requires Modal authentication: `modal token new`

See: https://modal.com/docs/guide/sandbox
"""

from __future__ import annotations

import uuid

import modal


class ModalSandbox:
    """
    Persistent Modal sandbox for code execution.

    The sandbox is created on first use and reused across executions.

    Usage:
        sandbox = ModalSandbox()
        sandbox.write_file("/workspace/code.py", "print('hello')")
        exit_code, stdout, stderr = sandbox.exec("python", "code.py", workdir="/workspace")
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
        self._app: modal.App | None = None
        self._sandbox: modal.Sandbox | None = None

    def _ensure_sandbox(self) -> modal.Sandbox:
        """Get or create the persistent sandbox."""
        if self._sandbox is None:
            self._app = modal.App.lookup(self._app_name, create_if_missing=True)
            self._sandbox = modal.Sandbox.create(app=self._app, image=self._image)
        return self._sandbox

    def write_file(self, path: str, content: str) -> None:
        """Write a file into the sandbox filesystem."""
        sandbox = self._ensure_sandbox()
        with sandbox.open(path, "w") as f:
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
        sandbox = self._ensure_sandbox()
        timeout = timeout if timeout is not None else self._default_timeout

        try:
            proc = sandbox.exec(*args, workdir=workdir, timeout=timeout)
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
        sandbox = self._ensure_sandbox()
        workdir = f"/workspace/{uuid.uuid4().hex[:12]}"
        try:
            sandbox.exec("mkdir", "-p", workdir).wait()
            for filename, content in files.items():
                self.write_file(f"{workdir}/{filename}", content)
            return self.exec(*command, workdir=workdir, timeout=timeout)
        finally:
            sandbox.exec("rm", "-rf", workdir).wait()

