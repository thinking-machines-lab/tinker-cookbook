"""Fystash cloud sandbox backend for Tinker (SandboxInterface peer of ModalSandbox).

Warm Firecracker rooms via the Fystash Room HTTP API (stdlib urllib — no extra deps).

Honesty / v1 limits:

- Does **not** build Harbor task ``environment/Dockerfile`` (Modal default does).
- Template path by default (``FYSTASH_TEMPLATE_ID``, usually ``default``).
- Optional DinD: set ``FYSTASH_DOCKER_IMAGE`` → ``template_id=docker`` + pull + run.

Docs: https://fystash.ai · https://docs.fystash.ai
Requires ``FYSTASH_API_KEY``. Optional: ``FYSTASH_API`` (default ``https://api.fystash.ai``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

_CTR_NAME = "tinker-task"
_PULL_TIMEOUT_S = 1800.0
_DOCKER_READY_S = 120.0
_MAX_EXEC_TIMEOUT_MS = 3_600_000
_HTTP_TIMEOUT_S = 3900.0


class FystashApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Fystash API HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class _RoomApi:
    """Sync Room API client (urllib)."""

    def __init__(
        self, base_url: str, api_key: str, *, timeout: float = _HTTP_TIMEOUT_S
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def close(self) -> None:
        return

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        url = f"{self._base}{path}"
        if params:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(params)}"
        data: bytes | None = None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        if json_body is not None:
            data = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw.decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                parsed = json.loads(detail)
                detail = str(parsed.get("detail", detail))
            except Exception:  # noqa: BLE001
                pass
            raise FystashApiError(exc.code, detail) from exc

    def create_room(self, room_id: str) -> dict[str, Any]:
        return self._request("POST", "/v1/rooms", json_body={"room_id": room_id})

    def create_from_template(
        self,
        room_id: str,
        agent_id: str,
        *,
        guest_cid: int,
        template_id: str,
        memory_mib: int,
        vcpu_count: int,
        attach_fabric: bool = True,
    ) -> dict[str, Any]:
        body = {
            "agent_id": agent_id,
            "guest_cid": guest_cid,
            "template_id": template_id,
            "vcpu_count": vcpu_count,
            "memory_mib": memory_mib,
            "enable_guest_net": True,
            "attach_fabric": attach_fabric,
        }
        last_exc: Exception | None = None
        for attempt in range(1, 13):
            try:
                return self._request(
                    "POST",
                    f"/v1/rooms/{room_id}/sandboxes/from-template",
                    json_body=body,
                )
            except FystashApiError as exc:
                last_exc = exc
                if exc.status not in (502, 503) or attempt >= 12:
                    raise
                detail = exc.detail.lower()
                if exc.status == 502 and not any(
                    tok in detail
                    for tok in ("timed out", "connection refused", "reset by peer")
                ):
                    raise
                time.sleep(min(30.0, 5.0 * attempt))
        if last_exc is None:
            raise RuntimeError("create_from_template exhausted retries")
        raise last_exc

    def exec(
        self,
        room_id: str,
        agent_id: str,
        argv: list[str],
        *,
        cwd: str | None = None,
        timeout_ms: int = 120_000,
        stdin: bytes | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"argv": argv, "timeout_ms": timeout_ms}
        if cwd is not None:
            body["cwd"] = cwd
        if stdin is not None:
            body["stdin_b64"] = base64.b64encode(stdin).decode()
        if env:
            body["env"] = env
        return self._request(
            "POST",
            f"/v1/rooms/{room_id}/sandboxes/{agent_id}/exec",
            json_body=body,
        )

    def destroy(self, room_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/rooms/{room_id}")

    def create_episode_batch(
        self,
        *,
        count: int,
        template_id: str = "default",
        agent_id: str = "tinker",
        room_id_prefix: str = "tkp",
        memory_mib: int = 256,
        strategy: str = "wave",
        attach_fabric: bool = True,
        labels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "count": int(count),
            "template_id": template_id,
            "agent_id": agent_id,
            "room_id_prefix": room_id_prefix,
            "memory_mib": int(memory_mib),
            "strategy": strategy,
            "attach_fabric": attach_fabric,
        }
        if labels is not None:
            body["labels"] = labels
        return self._request("POST", "/v1/episodes/batch", json_body=body)

    def destroy_episode_batch(self, batch_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/episodes/batch/{batch_id}")

    def reserve_capacity(
        self,
        *,
        count: int,
        template_id: str = "default",
        ttl_s: int = 3600,
        kind: str = "hard_l1_fence",
        labels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "count": int(count),
            "template_id": template_id,
            "ttl_s": int(ttl_s),
            "kind": kind,
        }
        if labels is not None:
            body["labels"] = labels
        return self._request("POST", "/v1/capacity/reserve", json_body=body)

    def release_capacity(self, reservation_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/capacity/reservations/{reservation_id}")


def _decode_b64(value: str | None) -> str:
    if not value:
        return ""
    return base64.b64decode(value).decode(errors="replace")


def _memory_for_template(template_id: str) -> int:
    return {
        "default": 256,
        "docker": 2048,
        "browser": 2048,
        "desktop": 2048,
    }.get(template_id, 256)


class FystashSandbox:
    """Persistent Fystash room sandbox. Conforms to SandboxInterface."""

    def __init__(
        self,
        *,
        api: _RoomApi,
        room_id: str,
        agent_id: str,
        timeout: int,
        template_id: str,
        docker_image: str | None,
        create_wall_ms: float | None,
        dind: bool,
        max_stream_output_bytes: int = 128 * 1024,
        owns_api: bool = True,
        on_cleanup: Any | None = None,
    ) -> None:
        self._api = api
        self._room_id = room_id
        self._agent_id = agent_id
        self._timeout = timeout
        self._template_id = template_id
        self._docker_image = docker_image
        self._dind = dind
        self._max_stream_output_bytes = max_stream_output_bytes
        self.create_wall_ms = create_wall_ms
        self._cleaned = False
        self._owns_api = owns_api
        self._on_cleanup = on_cleanup

    @classmethod
    def from_existing(
        cls,
        api: _RoomApi,
        *,
        room_id: str,
        agent_id: str,
        timeout: int = 600,
        template_id: str = "default",
        docker_image: str | None = None,
        create_wall_ms: float | None = None,
        dind: bool = False,
        max_stream_output_bytes: int = 128 * 1024,
        on_cleanup: Any | None = None,
    ) -> FystashSandbox:
        """Wrap an already-running room (episode batch / pool replenish)."""
        return cls(
            api=api,
            room_id=room_id,
            agent_id=agent_id,
            timeout=timeout,
            template_id=template_id,
            docker_image=docker_image,
            create_wall_ms=create_wall_ms,
            dind=dind,
            max_stream_output_bytes=max_stream_output_bytes,
            owns_api=False,
            on_cleanup=on_cleanup,
        )

    @classmethod
    async def create(
        cls,
        *,
        timeout: int = 600,
        template_id: str | None = None,
        docker_image: str | None = None,
        agent_id: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        max_stream_output_bytes: int = 128 * 1024,
    ) -> FystashSandbox:
        """Create a room and start a sandbox (template or DinD)."""
        key = api_key or os.environ.get("FYSTASH_API_KEY") or ""
        if not key:
            raise RuntimeError(
                "FYSTASH_API_KEY is required for FystashSandbox. "
                "Signup: https://fystash.ai/signup"
            )
        base = (
            api_url or os.environ.get("FYSTASH_API") or "https://api.fystash.ai"
        ).rstrip("/")
        image = (
            docker_image
            if docker_image is not None
            else (os.environ.get("FYSTASH_DOCKER_IMAGE") or None)
        )
        if image:
            tpl = "docker"
            dind = True
        else:
            tpl = template_id or os.environ.get("FYSTASH_TEMPLATE_ID") or "default"
            dind = False
        aid = agent_id or os.environ.get("FYSTASH_AGENT_ID") or "tinker"
        room_id = f"tk-{uuid.uuid4().hex[:10]}"
        guest_cid = 9600 + (int(uuid.uuid4().hex[:4], 16) % 300)
        api = _RoomApi(base, key)
        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(api.create_room, room_id)
            created = await asyncio.to_thread(
                api.create_from_template,
                room_id,
                aid,
                guest_cid=guest_cid,
                template_id=tpl,
                memory_mib=_memory_for_template(tpl),
                vcpu_count=1 if not dind else 2,
                attach_fabric=True,
            )
            create_wall_ms = (
                float(created["create_wall_ms"])
                if created.get("create_wall_ms") is not None
                else (time.perf_counter() - t0) * 1000.0
            )
            sb = cls(
                api=api,
                room_id=room_id,
                agent_id=aid,
                timeout=timeout,
                template_id=tpl,
                docker_image=image,
                create_wall_ms=create_wall_ms,
                dind=dind,
                max_stream_output_bytes=max_stream_output_bytes,
            )
            if dind and image:
                await sb._prepare_dind(image)
            return sb
        except Exception:
            try:
                await asyncio.to_thread(api.destroy, room_id)
            except Exception:  # noqa: BLE001
                pass
            api.close()
            raise

    @property
    def sandbox_id(self) -> str:
        return self._room_id

    async def _host_exec(
        self,
        command: str,
        *,
        timeout: int = 60,
        cwd: str | None = None,
        stdin: bytes | None = None,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        if self._cleaned:
            raise SandboxTerminatedError("sandbox cleaned up")
        timeout_ms = min(max(timeout, 1) * 1000, _MAX_EXEC_TIMEOUT_MS)
        try:
            resp = await asyncio.to_thread(
                self._api.exec,
                self._room_id,
                self._agent_id,
                ["/bin/bash", "-lc", command],
                cwd=cwd,
                timeout_ms=timeout_ms,
                stdin=stdin,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if any(k in msg for k in ("not found", "destroyed", "terminated", "404")):
                raise SandboxTerminatedError(str(exc)) from exc
            return SandboxResult(stdout="", stderr=str(exc), exit_code=-1)
        stdout = _decode_b64(resp.get("stdout_b64"))
        stderr = _decode_b64(resp.get("stderr_b64"))
        cap = (
            max_output_bytes
            if max_output_bytes is not None
            else self._max_stream_output_bytes
        )
        if len(stdout.encode()) > cap:
            stdout = stdout.encode()[:cap].decode(errors="replace")
        if len(stderr.encode()) > cap:
            stderr = stderr.encode()[:cap].decode(errors="replace")
        code = resp.get("exit_code")
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=int(code) if code is not None else 1,
        )

    async def _prepare_dind(self, image: str) -> None:
        deadline = time.monotonic() + _DOCKER_READY_S
        while time.monotonic() < deadline:
            r = await self._host_exec(
                "docker info >/dev/null 2>&1 && echo ready", timeout=30
            )
            if r.exit_code == 0 and "ready" in r.stdout:
                break
            await asyncio.sleep(2.0)
        else:
            raise RuntimeError("dockerd not ready in guest")
        pull = await self._host_exec(
            f"docker pull {shlex.quote(image)}",
            timeout=int(_PULL_TIMEOUT_S),
        )
        if pull.exit_code != 0:
            raise RuntimeError(
                f"docker pull failed for {image!r}: {(pull.stderr or pull.stdout)[:800]}"
            )
        run = await self._host_exec(
            f"docker rm -f {shlex.quote(_CTR_NAME)} >/dev/null 2>&1 || true; "
            f"docker run -d --name {shlex.quote(_CTR_NAME)} "
            f"--entrypoint sleep {shlex.quote(image)} infinity",
            timeout=120,
        )
        if run.exit_code != 0:
            raise RuntimeError(
                f"docker run failed for {image!r}: {(run.stderr or run.stdout)[:800]}"
            )

    def _wrap_for_dind(self, command: str, workdir: str | None) -> str:
        work = f"-w {shlex.quote(workdir)} " if workdir else ""
        return (
            f"docker exec {work}{shlex.quote(_CTR_NAME)} "
            f"bash -lc {shlex.quote(command)}"
        )

    async def send_heartbeat(self, timeout: int = 30) -> None:
        result = await self._host_exec("true", timeout=timeout)
        if result.exit_code != 0:
            raise SandboxTerminatedError(
                f"heartbeat failed: {result.stderr or result.stdout}"
            )

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        if self._dind:
            wrapped = self._wrap_for_dind(command, workdir)
            return await self._host_exec(
                wrapped,
                timeout=timeout,
                max_output_bytes=max_output_bytes,
            )
        return await self._host_exec(
            command,
            timeout=timeout,
            cwd=workdir,
            max_output_bytes=max_output_bytes,
        )

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        if max_bytes is not None:
            cmd = f"head -c {int(max_bytes)} {shlex.quote(path)}"
        else:
            cmd = f"cat {shlex.quote(path)}"
        return await self.run_command(cmd, timeout=timeout)

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        data = content.encode() if isinstance(content, str) else content
        parent = str(PurePosixPath(path).parent)
        inner = f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(path)}"
        if executable:
            inner += f" && chmod +x {shlex.quote(path)}"
        if self._dind:
            script = (
                f"docker exec -i -u root {shlex.quote(_CTR_NAME)} bash -lc "
                f"{shlex.quote(inner)}"
            )
        else:
            script = inner
        return await self._host_exec(script, timeout=timeout, stdin=data)

    async def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        try:
            await asyncio.to_thread(self._api.destroy, self._room_id)
        except Exception:  # noqa: BLE001
            pass
        finally:
            if self._on_cleanup is not None:
                try:
                    maybe = self._on_cleanup(self)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                except Exception:  # noqa: BLE001
                    pass
            if self._owns_api:
                self._api.close()


async def fystash_sandbox_factory(env_dir: Path, timeout: int) -> FystashSandbox:
    """SandboxFactory for Harbor RL — env_dir accepted but Dockerfile not built.

    When ``FYSTASH_POOL_SIZE>0``, acquire from the capacity-backed pool.
    """
    _ = env_dir
    try:
        pool_size = int(os.environ.get("FYSTASH_POOL_SIZE", "0"))
    except ValueError:
        pool_size = 0
    if pool_size > 0:
        from tinker_cookbook.sandbox.fystash_pool import get_or_start_pool

        pool = await get_or_start_pool()
        return await pool.acquire(timeout=timeout)
    return await FystashSandbox.create(timeout=timeout)
