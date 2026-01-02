"""
Thin wrapper around Modal Sandbox API.

Modal provides cloud-based sandboxed execution environments.
Requires Modal authentication: `modal token new`

See: https://modal.com/docs/guide/sandbox
"""

from __future__ import annotations

import modal


class ModalSandbox:
    """
    Persistent Modal sandbox for code execution.

    The sandbox is created on first use and reused across executions.
    Call close() to terminate the sandbox when done.

    Usage:
        sandbox = ModalSandbox()
        sandbox.write_file("/workspace/code.py", "print('hello')")
        exit_code, stdout, stderr = sandbox.exec("python", "code.py", workdir="/workspace")
        sandbox.close()
    """

    def __init__(
        self,
        app_name: str = "tinker-cookbook-runner",
        default_timeout: int = 240,
    ):
        self._app_name = app_name
        self._default_timeout = default_timeout
        self._app: modal.App | None = None
        self._sandbox: modal.Sandbox | None = None

    def _ensure_sandbox(self) -> modal.Sandbox:
        """Get or create the persistent sandbox."""
        if self._sandbox is None:
            self._app = modal.App.lookup(self._app_name, create_if_missing=True)
            self._sandbox = modal.Sandbox.create(app=self._app)
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

    def cleanup(self, path: str) -> None:
        """
        Remove a file or directory from the sandbox.

        Raises RuntimeError if the cleanup fails.
        """
        sandbox = self._ensure_sandbox()
        proc = sandbox.exec("rm", "-rf", path)
        _ = proc.stdout.read()
        exit_code = proc.wait()
        if exit_code != 0:
            stderr = proc.stderr.read()
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to cleanup {path}: {stderr}")

    def mkdir(self, path: str) -> None:
        """
        Create a directory in the sandbox.

        Raises RuntimeError if mkdir fails.
        """
        sandbox = self._ensure_sandbox()
        proc = sandbox.exec("mkdir", "-p", path)
        _ = proc.stdout.read()
        exit_code = proc.wait()
        if exit_code != 0:
            stderr = proc.stderr.read()
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to mkdir {path}: {stderr}")

    def close(self) -> None:
        """Terminate the sandbox and release resources."""
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
            except Exception:
                pass
            self._sandbox = None
            self._app = None
