"""Together Sandbox backend for tinker-cookbook.

Implements SandboxInterface using the ``together-sandbox`` SDK for sandbox
lifecycle, command execution, and file operations.

Prerequisites:
    pip install together-sandbox
    export TOGETHER_API_KEY=<your-key>

Usage:
    sandbox = await TogetherSandbox.create(
        dockerfile_path="path/to/Dockerfile",
        snapshot_alias="harbor@my-task",
    )
    result = await sandbox.run_command("echo hello")
    await sandbox.cleanup()
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path

from together_sandbox import (
    TogetherSandbox as TogetherSandboxSDK,
    Sandbox as SandboxConnection,
    CreateContextSnapshotParams,
    HttpError,
)

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 128 * 1024


class TogetherSandbox:
    """Together Sandbox that conforms to tinker-cookbook's SandboxInterface.

    Lifecycle:
    1. Build Docker image and create a snapshot via the SDK
    2. Create an ephemeral sandbox from the snapshot
    3. Start the sandbox
    4. Execute commands / read / write files via the SDK
    5. Shut down sandbox on cleanup
    """

    def __init__(
        self,
        sandbox_id: str,
        sdk: TogetherSandboxSDK,
        sandbox: SandboxConnection,
    ):
        self._sandbox_id = sandbox_id
        self._sdk = sdk
        self._sandbox = sandbox

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    @classmethod
    async def create(
        cls,
        dockerfile_path: str,
        context_dir: str | None = None,
        timeout: int = 3600,
        cpu: int = 2,
        memory_mb: int = 2048,
        disk_mb: int = 10240,
        snapshot_alias: str | None = None,
        api_key: str | None = None,
    ) -> TogetherSandbox:
        """Create a sandbox from a Dockerfile.

        Uses the together-sandbox SDK to build the image, create a snapshot,
        and start an ephemeral sandbox. Snapshot aliases enable caching so
        repeated runs skip the build step.

        Args:
            api_key: Optional API key. Falls back to ``TOGETHER_API_KEY`` env var.
        """
        sdk = TogetherSandboxSDK(api_key=api_key)

        # Check if snapshot already exists (cache hit)
        snapshot_id = None
        if snapshot_alias:
            try:
                existing = await sdk.snapshots.get_by_alias(snapshot_alias)
                snapshot_id = str(existing.id)
                logger.info("Reusing cached snapshot %s -> %s", snapshot_alias, snapshot_id)
            except (HttpError, RuntimeError):
                pass

        if not snapshot_id:
            # Build image and create snapshot via the SDK
            ctx = context_dir or str(Path(dockerfile_path).parent)
            dockerfile = Path(dockerfile_path)
            # Only pass dockerfile param if it's not the default location
            dockerfile_param = (
                str(dockerfile)
                if dockerfile != Path(ctx) / "Dockerfile"
                else None
            )

            logger.info("Building snapshot from %s", ctx)
            result = await sdk.snapshots.create(
                CreateContextSnapshotParams(
                    context=ctx,
                    dockerfile=dockerfile_param,
                    alias=snapshot_alias,
                )
            )
            snapshot_id = result.snapshot_id
            logger.info("Created snapshot: %s", snapshot_id)

        # Create and start sandbox
        sandbox_model = await sdk.sandboxes.create(
            snapshot_id=snapshot_id,
            ephemeral=True,
            millicpu=cpu * 1000,
            memory_bytes=memory_mb * 1024 * 1024,
            disk_bytes=disk_mb * 1024 * 1024,
        )
        sandbox_id_str = sandbox_model.id
        logger.info("Created sandbox: %s", sandbox_id_str)

        sandbox = await sdk.sandboxes.start(sandbox_id_str)
        logger.info("Sandbox running: %s", sandbox_id_str)

        instance = cls(sandbox_id=sandbox_id_str, sdk=sdk, sandbox=sandbox)

        # Configure DNS (sandboxes don't inherit host DNS)
        # TODO: remove once platform DNS is fixed
        await instance.run_command('echo "nameserver 1.1.1.1" > /etc/resolv.conf', workdir="/")

        return instance

    async def send_heartbeat(self, timeout: int = 30) -> None:
        """No-op — Together Sandbox uses hibernation timeout instead of heartbeats."""
        pass

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Execute a command in the sandbox."""
        max_bytes = max_output_bytes or MAX_OUTPUT_BYTES

        # Use login shell so PATH and other env vars from the image are available
        if workdir:
            wrapped = f"cd {shlex.quote(workdir)} && {command}"
        else:
            wrapped = command

        try:
            # Stream output to separate stdout from stderr
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            exit_code: int | None = None

            exec_item = await self._sandbox.execs.create(
                command="bash",
                args=["-lc", wrapped],
                autostart=True,
            )

            async def _stream():
                nonlocal exit_code
                async for event in self._sandbox.execs.stream_output(exec_item.id):
                    text = event.get("output", "")
                    if event.get("type") == "stderr":
                        stderr_parts.append(text)
                    elif text:
                        stdout_parts.append(text)
                    if isinstance(event.get("exitCode"), int):
                        exit_code = event["exitCode"]
                        return

            try:
                await asyncio.wait_for(_stream(), timeout=timeout)
            except asyncio.TimeoutError:
                return SandboxResult(stdout="", stderr="Command timed out", exit_code=-1)

            if exit_code is None:
                exit_code = -1

            stdout = "".join(stdout_parts)[:max_bytes]
            stderr = "".join(stderr_parts)[:max_bytes]

            return SandboxResult(stdout=stdout, stderr=stderr, exit_code=exit_code)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        try:
            content = await self._sandbox.files.read(path)
            if max_bytes:
                content = content[:max_bytes]
            return SandboxResult(stdout=content, stderr="", exit_code=0)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        try:
            await self._sandbox.files.create(path, content)
            if executable:
                await self.run_command(f"chmod +x {shlex.quote(path)}", workdir="/", timeout=10)
            return SandboxResult(stdout="", stderr="", exit_code=0)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

    async def cleanup(self) -> None:
        try:
            await self._sdk.sandboxes.shutdown(self._sandbox_id)
            logger.info("Sandbox %s shut down", self._sandbox_id)
        except Exception as e:
            logger.warning("Sandbox cleanup failed for %s: %s", self._sandbox_id, e)
