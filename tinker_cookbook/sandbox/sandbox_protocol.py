"""Sandbox Protocol for pluggable code execution backends."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Sandbox(Protocol):
    async def exec(
        self,
        *args: str,
        workdir: str = "/workspace",
        timeout: int | None = None,
    ) -> tuple[int, str, str]: ...
    async def write_file(self, path: str, content: str) -> None: ...
    async def terminate(self) -> None: ...
