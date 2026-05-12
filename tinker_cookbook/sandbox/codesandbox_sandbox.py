"""
CodeSandbox sandbox implementation for tinker-cookbook.

Implements SandboxInterface using the CodeSandbox Bartender API + Pint protocol.
Requires: CSB_API_KEY environment variable and Docker for image builds.

This adapter bridges the tinker-cookbook SandboxInterface with CodeSandbox's
two-layer API:
  - CodeSandboxClient: lifecycle via Bartender API (snapshots, sandboxes, start/stop)
  - PintClient: execution (run commands, read/write files inside the VM)

Usage:
    sandbox = await CodeSandboxSandbox.create(dockerfile_path="/path/to/Dockerfile")
    result = await sandbox.run_command("echo hello")
    await sandbox.cleanup()
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

import httpx

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

logger = logging.getLogger(__name__)

MAX_STREAM_OUTPUT_BYTES = 128 * 1024
API_TIMEOUT = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
MAX_RETRIES = 3
RETRY_DELAY = 2.0

BARTENDER_BASE_URL = "https://api.bartender.codesandbox.stream"
LEGACY_BASE_URL = "https://api.codesandbox.stream"


async def _retry_request(fn, retries=MAX_RETRIES):
    """Retry an async function on transient network errors."""
    for attempt in range(retries):
        try:
            return await fn()
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.PoolTimeout) as e:
            if attempt == retries - 1:
                raise
            delay = RETRY_DELAY * (attempt + 1)
            logger.warning("Request failed (%s), retrying in %.1fs (%d/%d)",
                           type(e).__name__, delay, attempt + 1, retries)
            await asyncio.sleep(delay)


class _PintClient:
    """Client for Pint protocol (command execution and file I/O inside a CSB VM)."""

    def __init__(self, pint_url: str, pint_token: str):
        if pint_url.startswith("wss://"):
            base_url = pint_url.replace("wss://", "https://")
        elif pint_url.startswith("ws://"):
            base_url = pint_url.replace("ws://", "http://")
        else:
            base_url = pint_url

        self._base_url = base_url
        self._token = pint_token
        self._host = None

        # DEVBOX mode support
        if os.getenv("DEVBOX", "false").lower() == "true":
            from urllib.parse import urlparse

            gateway_ip = os.getenv("DEVBOX_GATEWAY_IP")
            if not gateway_ip:
                raise ValueError("DEVBOX mode enabled but DEVBOX_GATEWAY_IP not set")
            parsed = urlparse(base_url)
            self._host = parsed.netloc
            path = parsed.path
            query = f"?{parsed.query}" if parsed.query else ""
            self._base_url = f"http://{gateway_ip}{path}{query}"

    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self._token}"}
        if self._host:
            headers["Host"] = self._host
        if extra:
            headers.update(extra)
        return headers

    async def read_file(self, path: str) -> str:
        path = path.lstrip("/")
        url = f"{self._base_url}/api/v1/files/{path}"
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response = await client.get(url, headers=self._headers())
                response.raise_for_status()
                return response.json().get("content", "")
        return await _retry_request(_do)

    async def create_file(self, path: str, content: str | bytes) -> None:
        path = path.lstrip("/")
        url = f"{self._base_url}/api/v1/files/{path}"
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response = await client.post(
                    url,
                    headers=self._headers({"Content-Type": "application/json"}),
                    json={"content": content},
                )
                response.raise_for_status()
        return await _retry_request(_do)

    async def execute_command(self, command: str, args: list[str]) -> dict:
        url = f"{self._base_url}/api/v1/execs"
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response = await client.post(
                    url,
                    headers=self._headers({"Content-Type": "application/json"}),
                    json={"command": command, "args": args, "autorun": True},
                )
                response.raise_for_status()
                return response.json()
        return await _retry_request(_do)

    async def get_exec_status(self, exec_id: str) -> dict:
        url = f"{self._base_url}/api/v1/execs/{exec_id}"
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response = await client.get(
                    url, headers=self._headers({"Accept": "application/json"})
                )
                response.raise_for_status()
                return response.json()
        return await _retry_request(_do)

    async def get_exec_output(self, exec_id: str) -> list[dict]:
        url = f"{self._base_url}/api/v1/execs/{exec_id}/io"
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response = await client.get(
                    url, headers=self._headers({"Accept": "*/*"})
                )
                response.raise_for_status()
                if not response.text or response.text.strip() == "":
                    return []
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    return response.json()
                elif "text/plain" in content_type:
                    return [{"type": "stdout", "output": response.text}]
                else:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        return [{"type": "stdout", "output": response.text}]
        return await _retry_request(_do)


class _CSBApiClient:
    """Client for CodeSandbox Bartender API (sandbox lifecycle management)."""

    def __init__(self, api_key: str, base_url: str = BARTENDER_BASE_URL):
        self._api_key = api_key
        self._base_url = base_url

    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if extra:
            headers.update(extra)
        return headers

    async def get_meta_info(self) -> dict:
        # Meta endpoint remains on the legacy API
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.get(
                    f"{LEGACY_BASE_URL}/meta/info", headers=self._headers()
                )
                r.raise_for_status()
                return r.json()
        return await _retry_request(_do)

    async def get_snapshot_by_alias(self, alias: str) -> dict | None:
        """Look up a snapshot by its alias. Returns None if not found."""
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.get(
                    f"{self._base_url}/api/v1/snapshots/@{alias}",
                    headers=self._headers(),
                )
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.json()
        return await _retry_request(_do)

    async def create_snapshot(
        self,
        registry: str,
        repository: str,
        name: str,
        tag: str,
        architecture: str | None = None,
    ) -> dict:
        body = {
            "image": {
                "registry": registry,
                "repository": repository,
                "name": name,
                "tag": tag,
            },
        }
        if architecture:
            body["image"]["architecture"] = architecture
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

    async def assign_snapshot_alias(self, snapshot_id: str, alias: str) -> dict:
        async def _do():
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                r = await client.post(
                    f"{self._base_url}/api/v1/snapshots/{snapshot_id}/aliases",
                    headers=self._headers({"Content-Type": "application/json"}),
                    json={"alias": alias},
                )
                r.raise_for_status()
                # 204 No Content is the expected success response
                if r.status_code == 204 or not r.content:
                    return {"status": "ok"}
                return r.json()
        return await _retry_request(_do)

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
        return await _retry_request(_do)


class CodeSandboxSandbox:
    """
    CodeSandbox sandbox that conforms to tinker-cookbook's SandboxInterface.

    Lifecycle:
    1. Build Docker image from task Dockerfile, push to CSB registry
    2. Create a CSB snapshot from that image
    3. Create an ephemeral sandbox from the snapshot
    4. Start the sandbox and get Pint connection details
    5. Execute commands / read / write files via Pint
    6. Stop sandbox on cleanup
    """

    def __init__(
        self,
        sandbox_id: str,
        pint_client: _PintClient,
        csb_client: _CSBApiClient,
    ):
        self._sandbox_id = sandbox_id
        self._pint = pint_client
        self._csb = csb_client

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
        """Create a new CodeSandbox sandbox from a Dockerfile.

        Args:
            dockerfile_path: Path to the Dockerfile.
            context_dir: Docker build context directory (defaults to Dockerfile's parent).
            timeout: VM timeout in seconds (not used by Bartender API directly).
            cpu: Number of CPUs (converted to millicpu).
            memory_mb: Memory in MB (converted to bytes).
            disk_mb: Disk in MB (converted to bytes).
            snapshot_alias: Optional alias like "harbor@task-name" for snapshot caching.
        """
        api_key = os.environ.get("CSB_API_KEY")
        if not api_key:
            raise ValueError("CSB_API_KEY environment variable not set")

        base_url = os.environ.get("CSB_BASE_URL", BARTENDER_BASE_URL)
        registry = os.environ.get("CSB_REGISTRY", "registry.codesandbox.stream")
        architecture = os.environ.get("CSB_IMAGE_ARCH", "amd64")

        csb = _CSBApiClient(api_key=api_key, base_url=base_url)

        # Get team ID
        meta = await csb.get_meta_info()
        team_id = meta.get("auth", {}).get("team")
        if not team_id:
            raise ValueError("Failed to get team ID from CSB API")

        # Check if snapshot already exists (via alias)
        snapshot_id = None
        if snapshot_alias:
            existing = await csb.get_snapshot_by_alias(snapshot_alias)
            if existing:
                snapshot_id = existing.get("id")
                logger.info("Reusing existing snapshot %s -> %s", snapshot_alias, snapshot_id)

        if not snapshot_id:
            # Docker login
            dockerfile = Path(dockerfile_path)
            ctx = context_dir or str(dockerfile.parent)

            login_cmd = ["docker", "login", registry, "-u", "harbor", "--password", api_key]
            subprocess.run(login_cmd, capture_output=True, text=True, check=True)

            # Build and push
            repo = base64.b32encode(team_id.encode()).decode().lower().rstrip("=")
            task_dir = dockerfile.parent.parent
            image_name = task_dir.name.lower().replace("_", "-")
            image_tag = os.environ.get("CSB_IMAGE_TAG", "latest")
            full_ref = f"{registry}/{repo}/{image_name}:{image_tag}"

            logger.info("Building Docker image: %s", full_ref)
            build_result = subprocess.run(
                ["docker", "build", "--platform", f"linux/{architecture}",
                 "-t", full_ref, "-f", str(dockerfile), ctx],
                capture_output=True, text=True,
            )
            if build_result.returncode != 0:
                logger.error("Docker build failed:\n%s\n%s",
                             build_result.stdout, build_result.stderr)
                raise RuntimeError(
                    f"Docker build failed (exit {build_result.returncode}):\n"
                    f"{build_result.stderr}"
                )

            logger.info("Pushing image to CSB registry")
            push_result = subprocess.run(
                ["docker", "push", full_ref],
                capture_output=True, text=True,
            )
            if push_result.returncode != 0:
                logger.error("Docker push failed:\n%s\n%s",
                             push_result.stdout, push_result.stderr)
                raise RuntimeError(
                    f"Docker push failed (exit {push_result.returncode}):\n"
                    f"{push_result.stderr}"
                )

            # Create snapshot
            snapshot_data = await csb.create_snapshot(
                registry=registry, repository=repo, name=image_name, tag=image_tag,
                architecture=architecture,
            )
            snapshot_id = snapshot_data.get("id")
            logger.info("Created CSB snapshot: %s", snapshot_id)

            # Assign alias for future reuse
            if snapshot_alias:
                await csb.assign_snapshot_alias(snapshot_id, snapshot_alias)

        # Create sandbox with explicit resources
        millicpu = cpu * 1000
        memory_bytes = memory_mb * 1024 * 1024
        disk_bytes = disk_mb * 1024 * 1024

        alias = snapshot_alias or f"snapshot-{snapshot_id}"
        sandbox_data = await csb.create_sandbox(
            snapshot_alias=alias,
            millicpu=millicpu,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
        )
        sandbox_id = sandbox_data["id"]
        logger.info("Created sandbox: %s", sandbox_id)

        # Start the sandbox and wait for it to reach running state
        await csb.start_sandbox(sandbox_id)
        wait_data = await csb.wait_for_sandbox(sandbox_id)
        logger.info("Sandbox started and running: %s", sandbox_id)

        # Bartender API returns agent_url/agent_token
        pint_url = wait_data.get("pint_url") or wait_data.get("agent_url")
        pint_token = wait_data.get("pint_token") or wait_data.get("agent_token")

        pint = _PintClient(
            pint_url=pint_url,
            pint_token=pint_token,
        )

        instance = cls(sandbox_id=sandbox_id, pint_client=pint, csb_client=csb)

        # Configure DNS
        await instance.run_command('echo "nameserver 1.1.1.1" > /etc/resolv.conf', workdir="/")

        return instance

    async def send_heartbeat(self) -> None:
        """No-op — CSB VMs use hibernation timeout instead of heartbeats."""
        pass

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the CodeSandbox VM."""
        cap = max_output_bytes or MAX_STREAM_OUTPUT_BYTES

        wrapped = command
        if workdir:
            wrapped = f"cd {shlex.quote(workdir)} && {command}"

        try:
            exec_resp = await self._pint.execute_command("bash", ["-c", wrapped])
            exec_id = exec_resp.get("id")
            if not exec_id:
                return SandboxResult(stdout="", stderr="No exec ID returned", exit_code=-1)

            # Poll for completion
            start_time = asyncio.get_event_loop().time()
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    return SandboxResult(
                        stdout="", stderr=f"Command timed out after {timeout}s", exit_code=-1
                    )

                status = await self._pint.get_exec_status(exec_id)
                if status.get("status") == "EXITED":
                    exit_code = status.get("exitCode", -1)
                    break

                await asyncio.sleep(0.5)

            # Collect output
            output_events = await self._pint.get_exec_output(exec_id)
            stdout_parts = []
            stderr_parts = []
            for event in output_events:
                if event.get("type") == "stdout":
                    stdout_parts.append(event.get("output", ""))
                elif event.get("type") == "stderr":
                    stderr_parts.append(event.get("output", ""))

            stdout = "".join(stdout_parts)[:cap]
            stderr = "".join(stderr_parts)[:cap]

            return SandboxResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

        except httpx.HTTPStatusError as e:
            if "not found" in str(e).lower() or "terminated" in str(e).lower():
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the CodeSandbox VM."""
        try:
            content = await self._pint.read_file(path)
            if max_bytes is not None:
                content = content[:max_bytes]
            return SandboxResult(stdout=content, stderr="", exit_code=0)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return SandboxResult(stdout="", stderr=f"File not found: {path}", exit_code=1)
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        """Write a file to the CodeSandbox VM."""
        try:
            await self._pint.create_file(path, content)
            if executable:
                await self.run_command(f"chmod +x {shlex.quote(path)}", workdir="/", timeout=10)
            return SandboxResult(stdout="", stderr="", exit_code=0)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Stop the CodeSandbox sandbox."""
        try:
            await self._csb.stop_sandbox(self._sandbox_id)
            logger.info("Stopped CSB sandbox: %s", self._sandbox_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("Sandbox already stopped: %s", self._sandbox_id)
            else:
                logger.warning("Failed to stop sandbox %s: %s", self._sandbox_id, e)
        except Exception as e:
            logger.warning("Cleanup error for %s: %s", self._sandbox_id, e)
