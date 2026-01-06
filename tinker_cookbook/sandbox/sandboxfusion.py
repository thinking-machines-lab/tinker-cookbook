"""
Thin wrapper around SandboxFusion HTTP API.

SandboxFusion is a Docker-based code execution sandbox. Run it locally with:

    docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609

Configuration via environment variables:
    SANDBOX_URL: Endpoint URL (default: http://localhost:8080/run_code)
    SANDBOX_MAX_CONCURRENCY: Max concurrent requests (default: 4)
"""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Any

import aiohttp


class SandboxFusionClient:
    """
    Async HTTP client for SandboxFusion code execution.

    Usage:
        client = SandboxFusionClient()
        success, response = await client.run(
            code="print('hello')",
            files={"data.txt": "some content"},
            timeout=30,
        )
        await client.close()
    """

    def __init__(
        self,
        url: str | None = None,
        max_concurrency: int | None = None,
    ):
        self._url = url or os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
        self._max_concurrency = max_concurrency or int(os.getenv("SANDBOX_MAX_CONCURRENCY", "4"))
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create shared HTTP session with connection pooling.

        The TCPConnector limits concurrent connections to max_concurrency.
        When all connections are busy, additional requests automatically wait
        in a queue until a connection becomes available.
        """
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=self._max_concurrency,
                    limit_per_host=self._max_concurrency,
                )
                timeout = aiohttp.ClientTimeout(total=6000)
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                )
            return self._session

    async def run(
        self,
        code: str,
        files: dict[str, str],
        timeout: float,
        language: str = "python",
    ) -> tuple[bool, dict[str, Any]]:
        """
        Execute code with supporting files in the sandbox.

        Args:
            code: Main code to execute (entry point)
            files: Additional files to include {filename: content}
            timeout: Execution timeout in seconds
            language: Programming language (default: python)

        Returns:
            Tuple of (success: bool, response: dict)
            - success is True only if status == "Success"
            - response contains the full API response or error details
        """
        encoded_files = {
            k: base64.b64encode(v.encode("utf-8")).decode("utf-8") for k, v in files.items()
        }

        payload = {
            "code": code,
            "language": language,
            "run_timeout": int(timeout),
            "files": encoded_files,
        }

        try:
            session = await self._get_session()
            async with session.post(self._url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return False, {"error": f"HTTP {resp.status}: {error_text}"}

                data: dict[str, Any] = await resp.json()

                if data.get("status") == "SandboxError":
                    return False, {"error": data.get("message", "SandboxError"), **data}

                success = data.get("status") == "Success"
                return success, data

        except Exception as e:
            return False, {"error": str(e)}

    async def close(self) -> None:
        """Close the HTTP session and release resources."""
        async with self._session_lock:
            if self._session is not None and not self._session.closed:
                await self._session.close()
                self._session = None
