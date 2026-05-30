"""
Daytona sandbox implementation for tinker-cookbook.

Implements SandboxInterface using the Daytona Python SDK (async).
Requires: DAYTONA_API_KEY environment variable.

Usage:
    sandbox = await DaytonaSandbox.create(dockerfile_path="/path/to/Dockerfile")
    result = await sandbox.run_command("echo hello")
    await sandbox.cleanup()
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from uuid import uuid4

from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    Image,
    Resources,
    SessionExecuteRequest,
)

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

logger = logging.getLogger(__name__)

MAX_STREAM_OUTPUT_BYTES = 128 * 1024


class DaytonaSandbox:
    """
    Daytona sandbox that conforms to tinker-cookbook's SandboxInterface.

    Lifecycle:
    1. Create sandbox from Dockerfile or base image (Daytona builds remotely)
    2. Execute commands via session-based process API
    3. Read/write files via filesystem API
    4. Delete sandbox on cleanup
    """

    def __init__(
        self,
        sandbox: AsyncSandbox,
        client: AsyncDaytona,
    ):
        self._sandbox = sandbox
        self._client = client

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.id

    @classmethod
    async def create(
        cls,
        dockerfile_path: str | None = None,
        base_image: str | None = None,
        timeout: int = 600,
        cpu: int = 4,
        memory: int = 8,
        disk: int = 10,
        auto_stop_interval: int = 0,
    ) -> DaytonaSandbox:
        """Create a new Daytona sandbox.

        Args:
            dockerfile_path: Path to Dockerfile. Daytona builds it remotely (no local Docker needed).
            base_image: Alternative to dockerfile — use a pre-built image (e.g., "python:3.13-slim").
            timeout: Build timeout in seconds.
            cpu: Number of CPUs.
            memory: Memory in GB.
            disk: Disk in GB.
            auto_stop_interval: Minutes of inactivity before auto-stop. 0 = no auto-stop.
        """
        if not dockerfile_path and not base_image:
            raise ValueError("Either dockerfile_path or base_image must be provided")

        client = AsyncDaytona()

        if dockerfile_path:
            image = Image.from_dockerfile(dockerfile_path)
        else:
            image = Image.base(base_image)

        params = CreateSandboxFromImageParams(
            image=image,
            resources=Resources(cpu=cpu, memory=memory, disk=disk),
            auto_stop_interval=auto_stop_interval,
        )

        logger.info("Creating Daytona sandbox (image=%s, cpu=%d, mem=%dGB)",
                     base_image or dockerfile_path, cpu, memory)

        sandbox = await client.create(params=params, timeout=timeout)
        logger.info("Daytona sandbox created: %s", sandbox.id)

        return cls(sandbox=sandbox, client=client)

    async def send_heartbeat(self) -> None:
        """No-op — Daytona uses auto_stop_interval instead of heartbeats."""
        pass

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the Daytona sandbox."""
        cap = max_output_bytes or MAX_STREAM_OUTPUT_BYTES

        session_id = str(uuid4())
        try:
            await self._sandbox.process.create_session(session_id)

            full_command = f"bash -c {shlex.quote(command)}"
            if timeout:
                full_command = f"timeout {timeout} {full_command}"
            if workdir:
                full_command = f"cd {shlex.quote(workdir)} && {full_command}"

            response = await self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=full_command, run_async=True),
                timeout=timeout,
            )

            if response.cmd_id is None:
                return SandboxResult(stdout="", stderr="No command ID returned", exit_code=-1)

            # Poll for completion
            cmd_id = response.cmd_id
            while True:
                cmd_status = await self._sandbox.process.get_session_command(
                    session_id, cmd_id
                )
                if cmd_status.exit_code is not None:
                    break
                await asyncio.sleep(0.5)

            # Get output
            logs = await self._sandbox.process.get_session_command_logs(
                session_id, cmd_id
            )

            stdout = (logs.stdout or "")[:cap]
            stderr = (logs.stderr or "")[:cap]

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=int(cmd_status.exit_code),
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "terminated" in error_msg:
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the Daytona sandbox."""
        try:
            # Use run_command as a reliable fallback since the SDK download
            # writes to a local file rather than returning content
            if max_bytes:
                cmd = f"head -c {max_bytes} {shlex.quote(path)}"
            else:
                cmd = f"cat {shlex.quote(path)}"
            return await self.run_command(cmd, workdir="/", timeout=timeout)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        """Write a file to the Daytona sandbox."""
        try:
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")

            # Write via base64 to handle special characters safely
            import base64
            b64 = base64.b64encode(content.encode()).decode()
            cmd = f"mkdir -p $(dirname {shlex.quote(path)}) && echo {shlex.quote(b64)} | base64 -d > {shlex.quote(path)}"
            if executable:
                cmd += f" && chmod +x {shlex.quote(path)}"

            return await self.run_command(cmd, workdir="/", timeout=timeout)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Delete the Daytona sandbox."""
        try:
            await self._sandbox.delete()
            logger.info("Deleted Daytona sandbox: %s", self._sandbox.id)
        except Exception as e:
            logger.warning("Failed to delete sandbox %s: %s", self._sandbox.id, e)
        finally:
            try:
                await self._client.close()
            except Exception:
                pass
