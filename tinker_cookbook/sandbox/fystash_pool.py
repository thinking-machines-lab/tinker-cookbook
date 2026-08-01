"""FystashSandboxPool — capacity-backed warm pool for Tinker (Layer 3).

Peer of tinker_cookbook ModalSandboxPool:

  reserve capacity → episode-batch replenish → acquire SandboxInterface →
  cleanup destroys room → replenish under the same reservation.

Env:
  FYSTASH_POOL_SIZE          (default 8; 0 disables pooling at factory layer)
  FYSTASH_POOL_CREATE_RATE   (default 4 concurrent replenish)
  FYSTASH_CAPACITY_TTL_S     (default 3600)
  FYSTASH_TEMPLATE_ID / FYSTASH_DOCKER_IMAGE / FYSTASH_API_KEY / FYSTASH_API
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time
import uuid
from typing import Any

from tinker_cookbook.sandbox.fystash_sandbox import (
    FystashSandbox,
    _RoomApi,
    __memory_for_template,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

logger = logging.getLogger(__name__)


class FystashSandboxPool:
    """Warm pool of Fystash rooms for concurrent Harbor RL / code_rl rollouts."""

    def __init__(
        self,
        *,
        pool_size: int | None = None,
        create_rate: int | None = None,
        capacity_ttl_s: int | None = None,
        sandbox_timeout_secs: int = 1200,
        api_url: str | None = None,
        api_key: str | None = None,
        template_id: str | None = None,
        docker_image: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("FYSTASH_API_KEY") or ""
        if not key:
            raise RuntimeError(
                "FYSTASH_API_KEY is required for FystashSandboxPool. "
                "Signup: https://fystash.ai/signup"
            )
        self._pool_size = (
            pool_size
            if pool_size is not None
            else int(os.environ.get("FYSTASH_POOL_SIZE", "8"))
        )
        self._create_rate = (
            create_rate
            if create_rate is not None
            else int(os.environ.get("FYSTASH_POOL_CREATE_RATE", "4"))
        )
        self._capacity_ttl_s = (
            capacity_ttl_s
            if capacity_ttl_s is not None
            else int(os.environ.get("FYSTASH_CAPACITY_TTL_S", "3600"))
        )
        self._sandbox_timeout_secs = sandbox_timeout_secs
        image = docker_image if docker_image is not None else (
            os.environ.get("FYSTASH_DOCKER_IMAGE") or None
        )
        if image:
            self._template_id = "docker"
            self._docker_image = image
            self._dind = True
        else:
            self._template_id = (
                template_id or os.environ.get("FYSTASH_TEMPLATE_ID") or "default"
            )
            self._docker_image = None
            self._dind = False
        self._agent_id = agent_id or os.environ.get("FYSTASH_AGENT_ID") or "tinker"
        base = (
            api_url or os.environ.get("FYSTASH_API") or "https://api.fystash.ai"
        ).rstrip("/")
        self._api = _RoomApi(base, key)
        self._warm_pool: asyncio.Queue[FystashSandbox] = asyncio.Queue()
        self._active_count = 0
        self._terminated = False
        self._maintain_task: asyncio.Task[None] | None = None
        self._reservation_id: str | None = None
        self._batch_ids: list[str] = []
        self._lock = asyncio.Lock()
        self._started = False
        self.start_wall_ms: float | None = None

    @property
    def reservation_id(self) -> str | None:
        return self._reservation_id

    @property
    def pool_size(self) -> int:
        return self._pool_size

    async def start(self) -> None:
        """Reserve capacity and kick the replenish loop."""
        if self._started:
            return
        if self._pool_size <= 0:
            raise RuntimeError("FYSTASH_POOL_SIZE must be > 0 to start a pool")
        t0 = time.perf_counter()
        labels = {"purpose": "tinker-pool", "agent": self._agent_id}
        reserved = await asyncio.to_thread(
            self._api.reserve_capacity,
            count=self._pool_size,
            template_id=self._template_id,
            ttl_s=self._capacity_ttl_s,
            kind="hard_l1_fence",
            labels=labels,
        )
        self._reservation_id = str(
            reserved.get("reservation_id")
            or reserved.get("id")
            or reserved.get("reservationId")
            or ""
        ) or None
        self._started = True
        self.start_wall_ms = (time.perf_counter() - t0) * 1000.0
        self._maintain_task = asyncio.create_task(self._maintain_pool())
        # Seed immediately so first acquire isn't starved.
        await self._maintain_pool_step()

    async def _notify_spent(self, _sb: FystashSandbox) -> None:
        self._active_count = max(0, self._active_count - 1)

    async def _replenish(self, need: int) -> None:
        if need <= 0 or self._terminated:
            return
        memory = _memory_for_template(self._template_id)
        labels = {"purpose": "tinker-pool-replenish"}
        try:
            batch = await asyncio.to_thread(
                self._api.create_episode_batch,
                count=need,
                template_id=self._template_id,
                agent_id=self._agent_id,
                room_id_prefix="tkp",
                memory_mib=memory,
                strategy="wave",
                attach_fabric=True,
                labels=labels,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("episode batch replenish failed: %s", exc)
            return
        batch_id = str(batch.get("batch_id") or "")
        if batch_id:
            self._batch_ids.append(batch_id)
        episodes = batch.get("episodes") or []
        for ep in episodes:
            room_id = str(ep.get("room_id") or "")
            if not room_id:
                continue
            agent_id = str(ep.get("agent_id") or self._agent_id)
            sb = FystashSandbox.from_existing(
                self._api,
                room_id=room_id,
                agent_id=agent_id,
                timeout=self._sandbox_timeout_secs,
                template_id=self._template_id,
                docker_image=self._docker_image,
                create_wall_ms=(
                    float(ep["create_wall_ms"])
                    if ep.get("create_wall_ms") is not None
                    else None
                ),
                dind=self._dind,
                on_cleanup=self._notify_spent,
            )
            if self._dind and self._docker_image:
                try:
                    await sb._prepare_dind(self._docker_image)
                except Exception as exc:  # noqa: BLE001
                    logger.error("DinD prepare failed for %s: %s", room_id, exc)
                    try:
                        await sb.cleanup()
                    except Exception:  # noqa: BLE001
                        pass
                    continue
            await self._warm_pool.put(sb)

    async def _maintain_pool(self) -> None:
        while not self._terminated:
            try:
                await self._maintain_pool_step()
            except Exception as exc:  # noqa: BLE001
                logger.error("Error maintaining FystashSandboxPool: %s", exc)
            await asyncio.sleep(1.0)

    async def _maintain_pool_step(self) -> None:
        async with self._lock:
            if self._terminated:
                return
            total = self._warm_pool.qsize() + self._active_count
            need = min(self._create_rate, self._pool_size - total)
            if need > 0:
                await self._replenish(need)

    async def acquire(self, *, timeout: int | None = None) -> FystashSandbox:
        """Borrow one warm sandbox. Blocks until available."""
        if self._terminated:
            raise RuntimeError("FystashSandboxPool has been terminated")
        if not self._started:
            await self.start()
        sb = await self._warm_pool.get()
        self._active_count += 1
        if timeout is not None:
            sb._timeout = timeout
        return sb

    async def run_in_workdir(
        self,
        files: dict[str, str],
        command: list[str],
        timeout: int | None = None,
    ) -> SandboxResult:
        """code_rl-shaped helper: borrow → write files → run → cleanup."""
        if self._terminated:
            raise RuntimeError("FystashSandboxPool has been terminated")
        sandbox = await self.acquire(timeout=timeout or self._sandbox_timeout_secs)
        try:
            workdir = f"/tmp/fystash-pool-{uuid.uuid4().hex[:12]}"
            mkdir = await sandbox.run_command(
                f"mkdir -p {shlex.quote(workdir)}", timeout=timeout or 60
            )
            if mkdir.exit_code != 0:
                return SandboxResult(
                    stdout="",
                    stderr=f"Failed to create workdir: {workdir}",
                    exit_code=mkdir.exit_code,
                )
            if files:
                await asyncio.gather(
                    *(
                        sandbox.write_file(f"{workdir}/{filename}", content)
                        for filename, content in files.items()
                    )
                )
            return await sandbox.run_command(
                shlex.join(command),
                workdir=workdir,
                timeout=timeout or self._sandbox_timeout_secs,
            )
        finally:
            await sandbox.cleanup()

    async def terminate(self) -> None:
        """Stop replenish, destroy warm rooms, release capacity."""
        self._terminated = True
        if self._maintain_task is not None:
            self._maintain_task.cancel()
            try:
                await self._maintain_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._maintain_task = None

        while self._active_count > 0:
            await asyncio.sleep(0.2)

        leftover: list[FystashSandbox] = []
        while not self._warm_pool.empty():
            try:
                leftover.append(self._warm_pool.get_nowait())
            except asyncio.QueueEmpty:
                break
        await asyncio.gather(
            *(sb.cleanup() for sb in leftover),
            return_exceptions=True,
        )

        for batch_id in list(self._batch_ids):
            try:
                await asyncio.to_thread(self._api.destroy_episode_batch, batch_id)
            except Exception:  # noqa: BLE001
                pass
        self._batch_ids.clear()

        if self._reservation_id:
            try:
                await asyncio.to_thread(
                    self._api.release_capacity, self._reservation_id
                )
            except Exception:  # noqa: BLE001
                pass
            self._reservation_id = None

        self._api.close()
        self._started = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "fystash-sandbox-pool",
            "pool_size": self._pool_size,
            "create_rate": self._create_rate,
            "capacity_ttl_s": self._capacity_ttl_s,
            "template_id": self._template_id,
            "dind": self._dind,
            "reservation_id": self._reservation_id,
            "warm_qsize": self._warm_pool.qsize(),
            "active_count": self._active_count,
            "start_wall_ms": self.start_wall_ms,
            "terminated": self._terminated,
        }


_POOL: FystashSandboxPool | None = None
_POOL_LOCK = asyncio.Lock()


async def get_or_start_pool(**kwargs: Any) -> FystashSandboxPool:
    """Process-level singleton pool (lazy start)."""
    global _POOL
    async with _POOL_LOCK:
        if _POOL is None or _POOL._terminated:
            _POOL = FystashSandboxPool(**kwargs)
            await _POOL.start()
        return _POOL


def reset_pool_singleton() -> None:
    """Test helper — drop the process singleton without awaiting terminate."""
    global _POOL
    _POOL = None
