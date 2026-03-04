"""Sandbox interface for pluggable code execution backends."""

from typing import Any, Protocol, runtime_checkable

import chz

DEFAULT_MAX_OUTPUT_BYTES: int = 128 * 1024  # 128 KB per stream


@chz.chz
class SandboxResult:
    """Result from a sandbox operation."""

    stdout: str
    stderr: str
    exit_code: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)


class SandboxTerminatedError(Exception):
    """Raised when a sandbox has been terminated or died unexpectedly."""

    pass


@runtime_checkable
class SandboxInterface(Protocol):
    """Interface for a sandbox.

    Implementations must provide: run_command, read_file, write_file,
    send_heartbeat, and cleanup.
    """

    @staticmethod
    def resolve_max_output_bytes(max_output_bytes: int | None) -> int:
        """Resolve an optional max_output_bytes to a concrete value, falling back to DEFAULT_MAX_OUTPUT_BYTES."""
        return max_output_bytes if max_output_bytes is not None else DEFAULT_MAX_OUTPUT_BYTES

    async def send_heartbeat(self) -> None:
        """Send a heartbeat to keep the sandbox alive.

        If the sandbox server does not support heartbeat, this method can be a no-op.
        """
        ...

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a command in the sandbox.

        Setting ``workdir=None`` will run the command in the default WORKDIR set
        in the container image (Dockerfile).

        Args:
            command: Shell command string to execute.
            workdir: Working directory for the command.
            timeout: Timeout in seconds.
            max_output_bytes: Cap stdout/stderr at this many bytes. When None,
                implementation uses its default (e.g. 128 KB).
        """
        ...

    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
        """Read the content of a file from the sandbox.

        Args:
            path: Path to the file in the sandbox.
            max_bytes: If set, only read up to this many bytes from the file.
            timeout: Timeout in seconds for the read operation.
        """
        ...

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        """Write content to a file in the sandbox.

        Args:
            path: Destination path inside the sandbox.
            content: File content (str or bytes).
            executable: If True, make the file executable.
            timeout: Timeout in seconds.
        """
        ...

    async def cleanup(self) -> None:
        """Clean up the sandbox."""
        ...


class SandboxResource:
    """Resource wrapping a SandboxInterface."""

    def __init__(self, sandbox: SandboxInterface):
        self.sandbox: SandboxInterface = sandbox

    async def send_heartbeat(self) -> None:
        await self.sandbox.send_heartbeat()

    async def cleanup(self) -> None:
        await self.sandbox.cleanup()
