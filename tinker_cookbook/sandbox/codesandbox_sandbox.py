"""CodeSandbox (Bartender) sandbox backend for tinker-cookbook.

Implements SandboxInterface using the Bartender API for sandbox lifecycle
and the Pint protocol for in-sandbox command execution and file operations.

Prerequisites:
    export CSB_API_KEY=<your-key>
    export CSB_BASE_URL=https://api.bartender.codesandbox.stream  # optional

Usage:
    sandbox = await CodeSandboxSandbox.create(
        dockerfile_path="path/to/Dockerfile",
        snapshot_alias="harbor@my-task",
    )
    result = await sandbox.run_command("echo hello")
    await sandbox.cleanup()
"""

from __future__ import annotations

import base64
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import httpx

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

logger = logging.getLogger(__name__)

BARTENDER_BASE_URL = "https://api.bartender.codesandbox.stream"
LEGACY_BASE_URL = "https://api.codesandbox.stream"
API_TIMEOUT = 60.0
MAX_OUTPUT_BYTES = 128 * 1024


async def _retry_request(fn, retries: int = 3, delay: float = 2.0):
    """Retry an async HTTP request with exponential backoff."""
    import asyncio

    for attempt in range(retries):
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                await asyncio.sleep(delay * (2**attempt))
                continue
            raise


# ---------------------------------------------------------------------------
# Bartender API client
# ---------------------------------------------------------------------------


class _BartenderClient:
    """HTTP client for the Bartender sandbox orchestration API."""

    def __init__(self, api_key: str, base_url: str = BARTENDER_BASE_URL):
        self._api_key = api_key
        self._base_url = base_url

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if extra:
            headers.update(extra)
        return headers

    async def get_meta_info(self) -> dict:
        """Get team ID from the legacy API (Bartender doesn't expose /meta/info)."""

        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.get(
                    f"{LEGACY_BASE_URL}/meta/info", headers=self._headers()
                )
                r.raise_for_status()
                return r.json()

        return await _retry_request(_do)

    async def get_snapshot_by_alias(self, alias: str) -> dict | None:
        """Look up a snapshot by alias. Returns None if not found."""
        try:
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.get(
                    f"{self._base_url}/api/v1/snapshots/@{alias}",
                    headers=self._headers(),
                )
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def create_snapshot(
        self,
        registry: str,
        repository: str,
        name: str,
        tag: str,
        architecture: str = "amd64",
    ) -> dict:
        body = {
            "image": {
                "registry": registry,
                "repository": repository,
                "name": name,
                "tag": tag,
                "architecture": architecture,
            }
        }

        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.post(
                    f"{self._base_url}/api/v1/snapshots/",
                    headers=self._headers({"Content-Type": "application/json"}),
                    json=body,
                )
                r.raise_for_status()
                return r.json()

        return await _retry_request(_do)

    async def assign_snapshot_alias(self, snapshot_id: str, alias: str) -> None:
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.post(
                    f"{self._base_url}/api/v1/snapshots/{snapshot_id}/aliases",
                    headers=self._headers({"Content-Type": "application/json"}),
                    json={"alias": alias},
                )
                r.raise_for_status()

        await _retry_request(_do)

    async def create_sandbox(
        self,
        snapshot_alias: str,
        millicpu: int = 2000,
        memory_bytes: int = 2 * 1024 * 1024 * 1024,
        disk_bytes: int = 10 * 1024 * 1024 * 1024,
    ) -> dict:
        body = {
            "snapshot_alias": snapshot_alias,
            "ephemeral": True,
            "millicpu": millicpu,
            "memory_bytes": memory_bytes,
            "disk_bytes": disk_bytes,
        }

        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.post(
                    f"{self._base_url}/api/v1/sandboxes",
                    headers=self._headers({"Content-Type": "application/json"}),
                    json=body,
                )
                r.raise_for_status()
                return r.json()

        return await _retry_request(_do)

    async def start_sandbox(self, sandbox_id: str) -> dict:
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.post(
                    f"{self._base_url}/api/v1/sandboxes/{sandbox_id}/start",
                    headers=self._headers(),
                )
                r.raise_for_status()
                return r.json()

        return await _retry_request(_do)

    async def wait_for_sandbox(self, sandbox_id: str) -> dict:
        async def _do():
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    f"{self._base_url}/api/v1/sandboxes/{sandbox_id}/wait",
                    headers=self._headers(),
                )
                r.raise_for_status()
                return r.json()

        return await _retry_request(_do)

    async def stop_sandbox(self, sandbox_id: str) -> None:
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.post(
                    f"{self._base_url}/api/v1/sandboxes/{sandbox_id}/stop",
                    headers=self._headers({"Content-Type": "application/json"}),
                    json={"stop_type": "shutdown"},
                )
                r.raise_for_status()

        await _retry_request(_do)


# ---------------------------------------------------------------------------
# Pint client (in-sandbox exec and file operations)
# ---------------------------------------------------------------------------


class _PintClient:
    """HTTP client for the Pint in-sandbox process manager."""

    def __init__(self, pint_url: str, pint_token: str):
        self._base_url = pint_url
        self._token = pint_token

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._token}"}
        if extra:
            headers.update(extra)
        return headers

    async def exec(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> dict:
        """Start a command execution. Returns immediately with an exec ID."""
        body: dict[str, Any] = {"command": command, "autorun": True}
        if args:
            body["args"] = args
        if cwd:
            body["cwd"] = cwd

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{self._base_url}/api/v1/execs",
                headers=self._headers({"Content-Type": "application/json"}),
                json=body,
            )
            r.raise_for_status()
            return r.json()

    async def get_exec_status(self, exec_id: str) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{self._base_url}/api/v1/execs/{exec_id}",
                headers=self._headers(),
            )
            r.raise_for_status()
            return r.json()

    async def get_exec_output(self, exec_id: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{self._base_url}/api/v1/execs/{exec_id}/io",
                headers=self._headers(),
            )
            r.raise_for_status()
            return r.json()

    async def write_file(self, path: str, content: str | bytes) -> None:
        if isinstance(content, str):
            content = content.encode()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{self._base_url}/api/v1/files/{path.lstrip('/')}",
                headers=self._headers({"Content-Type": "application/octet-stream"}),
                content=content,
            )
            r.raise_for_status()

    async def read_file(self, path: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{self._base_url}/api/v1/files/{path.lstrip('/')}",
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()
            return data.get("content", "")


# ---------------------------------------------------------------------------
# CodeSandboxSandbox (implements SandboxInterface)
# ---------------------------------------------------------------------------


class CodeSandboxSandbox:
    """CodeSandbox sandbox that conforms to tinker-cookbook's SandboxInterface.

    Lifecycle:
    1. Build Docker image locally, push to CSB registry
    2. Create a snapshot from the pushed image
    3. Create an ephemeral sandbox from the snapshot
    4. Start the sandbox and wait for running state
    5. Execute commands / read / write files via Pint
    6. Stop sandbox on cleanup
    """

    def __init__(
        self,
        sandbox_id: str,
        pint_client: _PintClient,
        bartender_client: _BartenderClient,
    ):
        self._sandbox_id = sandbox_id
        self._pint = pint_client
        self._bartender = bartender_client

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
    ) -> CodeSandboxSandbox:
        """Create a sandbox from a Dockerfile.

        Builds the image locally, pushes to the CSB registry, creates a
        snapshot, and starts an ephemeral sandbox. Uses snapshot aliases
        for caching so repeated runs skip the build step.
        """
        api_key = os.environ.get("CSB_API_KEY")
        if not api_key:
            raise ValueError("CSB_API_KEY environment variable not set")

        base_url = os.environ.get("CSB_BASE_URL", BARTENDER_BASE_URL)
        registry = os.environ.get("CSB_REGISTRY", "registry.codesandbox.stream")
        architecture = os.environ.get("CSB_IMAGE_ARCH", "amd64")

        bartender = _BartenderClient(api_key=api_key, base_url=base_url)

        # Get team ID (needed for registry repository path)
        meta = await bartender.get_meta_info()
        team_id = meta.get("auth", {}).get("team")
        if not team_id:
            raise ValueError("Failed to get team ID from CSB API")

        # Check if snapshot already exists
        snapshot_id = None
        if snapshot_alias:
            existing = await bartender.get_snapshot_by_alias(snapshot_alias)
            if existing:
                snapshot_id = existing.get("id")
                logger.info("Reusing cached snapshot %s -> %s", snapshot_alias, snapshot_id)

        if not snapshot_id:
            # Build and push Docker image
            dockerfile = Path(dockerfile_path)
            ctx = context_dir or str(dockerfile.parent)
            repo = base64.b32encode(team_id.encode()).decode().lower().rstrip("=")
            image_name = dockerfile.parent.parent.name.lower().replace("_", "-")
            image_tag = os.environ.get("CSB_IMAGE_TAG", "latest")
            full_ref = f"{registry}/{repo}/{image_name}:{image_tag}"

            # Docker login
            subprocess.run(
                ["docker", "login", registry, "-u", "harbor", "--password", api_key],
                capture_output=True, text=True, check=True,
            )

            # Build
            logger.info("Building Docker image: %s", full_ref)
            build = subprocess.run(
                ["docker", "build", "--platform", f"linux/{architecture}",
                 "-t", full_ref, "-f", str(dockerfile), ctx],
                capture_output=True, text=True,
            )
            if build.returncode != 0:
                raise RuntimeError(f"Docker build failed (exit {build.returncode}):\n{build.stderr}")

            # Push
            logger.info("Pushing image to CSB registry")
            push = subprocess.run(
                ["docker", "push", full_ref], capture_output=True, text=True,
            )
            if push.returncode != 0:
                raise RuntimeError(f"Docker push failed (exit {push.returncode}):\n{push.stderr}")

            # Create snapshot
            snapshot_data = await bartender.create_snapshot(
                registry=registry, repository=repo, name=image_name,
                tag=image_tag, architecture=architecture,
            )
            snapshot_id = snapshot_data.get("id")
            logger.info("Created snapshot: %s", snapshot_id)

            # Assign alias for future reuse
            if snapshot_alias:
                await bartender.assign_snapshot_alias(snapshot_id, snapshot_alias)

        # Create sandbox
        sandbox_data = await bartender.create_sandbox(
            snapshot_alias=snapshot_alias or f"snapshot-{snapshot_id}",
            millicpu=cpu * 1000,
            memory_bytes=memory_mb * 1024 * 1024,
            disk_bytes=disk_mb * 1024 * 1024,
        )
        sandbox_id = sandbox_data["id"]
        logger.info("Created sandbox: %s", sandbox_id)

        # Start and wait
        await bartender.start_sandbox(sandbox_id)
        wait_data = await bartender.wait_for_sandbox(sandbox_id)
        logger.info("Sandbox running: %s", sandbox_id)

        pint = _PintClient(
            pint_url=wait_data.get("agent_url"),
            pint_token=wait_data.get("agent_token"),
        )

        instance = cls(sandbox_id=sandbox_id, pint_client=pint, bartender_client=bartender)

        # Configure DNS (sandboxes don't inherit host DNS)
        await instance.run_command('echo "nameserver 1.1.1.1" > /etc/resolv.conf', workdir="/")

        return instance

    async def send_heartbeat(self) -> None:
        """No-op — CSB uses hibernation timeout instead of heartbeats."""
        pass

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Execute a command in the sandbox."""
        import asyncio

        max_bytes = max_output_bytes or MAX_OUTPUT_BYTES

        # Use login shell so PATH and other env vars from the image are available
        if workdir:
            wrapped = f"cd {shlex.quote(workdir)} && {command}"
        else:
            wrapped = command

        exec_data = await self._pint.exec(
            command="bash", args=["-lc", wrapped], timeout=float(timeout)
        )
        exec_id = exec_data.get("id")
        if not exec_id:
            return SandboxResult(stdout="", stderr="No exec ID returned", exit_code=-1)

        # Poll for completion
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            status = await self._pint.get_exec_status(exec_id)
            if status.get("status") == "EXITED":
                break
            await asyncio.sleep(1.0)
        else:
            return SandboxResult(stdout="", stderr="Command timed out", exit_code=-1)

        exit_code = status.get("exitCode", -1)

        # Collect output
        output_events = await self._pint.get_exec_output(exec_id)
        stdout_parts, stderr_parts = [], []
        for event in output_events:
            text = event.get("output", "")
            if event.get("type") == "stderr":
                stderr_parts.append(text)
            elif text:
                stdout_parts.append(text)

        stdout = "".join(stdout_parts)[:max_bytes]
        stderr = "".join(stderr_parts)[:max_bytes]

        return SandboxResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        try:
            content = await self._pint.read_file(path)
            if max_bytes:
                content = content[:max_bytes]
            return SandboxResult(stdout=content, stderr="", exit_code=0)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        try:
            await self._pint.write_file(path, content)
            if executable:
                await self.run_command(f"chmod +x {shlex.quote(path)}", workdir="/", timeout=10)
            return SandboxResult(stdout="", stderr="", exit_code=0)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

    async def cleanup(self) -> None:
        try:
            await self._bartender.stop_sandbox(self._sandbox_id)
            logger.info("Sandbox %s stopped", self._sandbox_id)
        except Exception as e:
            logger.warning("Sandbox cleanup failed for %s: %s", self._sandbox_id, e)
