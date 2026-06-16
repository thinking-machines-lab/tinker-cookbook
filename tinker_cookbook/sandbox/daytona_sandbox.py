"""
Thin wrapper around Daytona Sandbox API.

Daytona provides cloud-based sandboxed execution environments.
Requires Daytona authentication: ``export DAYTONA_API_KEY=...``
(or ``DAYTONA_JWT_TOKEN`` + ``DAYTONA_ORGANIZATION_ID``).

Supports two usage modes:

- Stateful: a persistent sandbox where shell state (cwd, env, variables)
  carries across calls. Used for multi-turn agentic workloads.
- Stateless: an ephemeral sandbox per call, for one-shot code grading.

Configuration via environment variables:
    DAYTONA_API_URL: API endpoint (default: https://app.daytona.io/api)
    DAYTONA_TARGET: Target region for sandbox placement
    DAYTONA_SNAPSHOT: Pre-created snapshot name. Not required —
        image builds are content-hashed and cached across sandboxes
        automatically. Use this only to pin a specific pre-built snapshot.

See: https://www.daytona.io
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
from pathlib import Path
from typing import Any

try:
    from daytona import (
        AsyncDaytona,
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        DaytonaConfig,
        DaytonaNotFoundError,
        FileUpload,
        Image,
        SessionExecuteRequest,
    )
except ImportError:
    raise ImportError(
        "daytona is required for DaytonaSandbox. "
        "Install it with: uv pip install 'tinker-cookbook[daytona] @ "
        "git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'"
    ) from None

from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxInterface,
    SandboxResult,
    SandboxTerminatedError,
)

logger = logging.getLogger(__name__)


def _cap_output(text: str, max_bytes: int) -> str:
    """Cap a string to *max_bytes* bytes of UTF-8, decoding safely at the boundary."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="replace")


# 15-minute auto-stop matches the Daytona SDK default; 30-minute auto-delete
# ensures a crashed rollout or forgotten ``cleanup()`` call does not leak the
# sandbox indefinitely. Both are overridable via ``DaytonaSandbox.create``.
_DEFAULT_AUTO_STOP_MINUTES = 15
_DEFAULT_AUTO_DELETE_MINUTES = 30
_DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024
_SESSION_ID = "tinker-cookbook-shell"


class DaytonaSandbox(SandboxInterface):
    """
    Persistent Daytona sandbox that implements :class:`SandboxInterface`.

    State persists across ``run_command`` calls via a long-running background
    session (``sandbox.process.create_session`` + ``execute_session_command``).

    For stateless per-call grading, prefer :func:`run_code_in_daytona`.

    Usage:
        sandbox = await DaytonaSandbox.create()
        await sandbox.write_file("/workspace/code.py", "print('hello')")
        result = await sandbox.run_command("python /workspace/code.py")
        print(result.stdout)
        await sandbox.cleanup()
    """

    def __init__(
        self,
        *,
        client: AsyncDaytona,
        sandbox: Any,  # daytona.AsyncSandbox — Any to avoid a hard dep at module import
        max_stream_output_bytes: int,
        owns_client: bool,
    ) -> None:
        self._client = client
        self._sandbox = sandbox
        self._max_stream_output_bytes = max_stream_output_bytes
        self._owns_client = owns_client
        self._session_ready = False
        self._session_lock = asyncio.Lock()
        self._cleaned_up = False

    @classmethod
    async def create(
        cls,
        *,
        image: Image | str | None = None,
        snapshot: str | None = None,
        timeout: int = 600,
        auto_stop_minutes: int | None = None,
        auto_delete_minutes: int | None = None,
        labels: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
        target: str | None = None,
        max_stream_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        client: AsyncDaytona | None = None,
    ) -> DaytonaSandbox:
        """Create a new Daytona sandbox.

        Args:
            image: Image to use. Mutually exclusive with *snapshot*. If
                both are ``None`` (and ``DAYTONA_SNAPSHOT`` is unset),
                defaults to ``Image.debian_slim()``. Image builds are
                cached by Daytona across sandboxes, so passing the same
                image repeatedly does not re-build.
            snapshot: Name of a pre-created Daytona snapshot. Mutually
                exclusive with *image*. Falls back to ``DAYTONA_SNAPSHOT``
                env var if unset.
            timeout: Max wait time in seconds for sandbox creation.
            auto_stop_minutes: Minutes of inactivity before auto-stop.
                ``0`` disables. Defaults to 15.
            auto_delete_minutes: Minutes after stopping before auto-delete.
                ``0`` means delete immediately, negative disables. Defaults
                to 30. Leak protection if ``cleanup()`` is skipped.
            labels: Custom labels attached to the sandbox.
            env_vars: Environment variables set inside the sandbox.
            target: Target region for sandbox placement.
            max_stream_output_bytes: Cap per-stream output at this many
                bytes. Defaults to 128 KB.
            client: Existing ``AsyncDaytona`` client to reuse. If ``None``,
                a new client is created and owned by this sandbox.
        """
        if image is not None and snapshot is not None:
            raise ValueError("Provide either image or snapshot, not both.")

        resolved_auto_stop = (
            auto_stop_minutes if auto_stop_minutes is not None else _DEFAULT_AUTO_STOP_MINUTES
        )
        resolved_auto_delete = (
            auto_delete_minutes if auto_delete_minutes is not None else _DEFAULT_AUTO_DELETE_MINUTES
        )

        owns_client = client is None
        if owns_client:
            config = DaytonaConfig(target=target) if target is not None else DaytonaConfig()
            client = AsyncDaytona(config)
        assert client is not None  # for type checker

        try:
            resolved_snapshot = snapshot if snapshot is not None else os.getenv("DAYTONA_SNAPSHOT")
            if resolved_snapshot:
                params: CreateSandboxFromImageParams | CreateSandboxFromSnapshotParams = (
                    CreateSandboxFromSnapshotParams(
                        snapshot=resolved_snapshot,
                        auto_stop_interval=resolved_auto_stop,
                        auto_delete_interval=resolved_auto_delete,
                        labels=labels,
                        env_vars=env_vars,
                    )
                )
            else:
                resolved_image: Image | str = image if image is not None else Image.debian_slim()
                params = CreateSandboxFromImageParams(
                    image=resolved_image,
                    auto_stop_interval=resolved_auto_stop,
                    auto_delete_interval=resolved_auto_delete,
                    labels=labels,
                    env_vars=env_vars,
                )
            sandbox = await client.create(params, timeout=float(timeout))
        except Exception:
            if owns_client:
                # Best-effort close of the HTTP session we just opened.
                with contextlib.suppress(Exception):
                    await client.close()
            raise

        return cls(
            client=client,
            sandbox=sandbox,
            max_stream_output_bytes=max_stream_output_bytes,
            owns_client=owns_client,
        )

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.id

    async def _ensure_session(self) -> None:
        """Lazily create the shared background session used by ``run_command``."""
        if self._session_ready:
            return
        async with self._session_lock:
            if self._session_ready:
                return
            try:
                await self._sandbox.process.create_session(_SESSION_ID)
                self._session_ready = True
            except DaytonaNotFoundError as e:
                raise SandboxTerminatedError(str(e)) from e

    async def send_heartbeat(self, timeout: int = 30) -> None:
        try:
            await asyncio.wait_for(self._sandbox.refresh_activity(), timeout=timeout)
        except DaytonaNotFoundError as e:
            raise SandboxTerminatedError(str(e)) from e

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox.

        Uses a persistent session so state (cwd, env, shell variables)
        carries across calls. A ``workdir`` argument is prepended as
        ``cd <workdir> &&`` for per-call scoping without mutating
        long-term session state.
        """
        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes

        await self._ensure_session()

        full_command = f"cd {shlex.quote(workdir)} && {command}" if workdir else command
        try:
            response = await self._sandbox.process.execute_session_command(
                _SESSION_ID,
                SessionExecuteRequest(command=full_command),
                timeout=timeout,
            )
        except DaytonaNotFoundError as e:
            raise SandboxTerminatedError(str(e)) from e
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

        exit_code = response.exit_code if response.exit_code is not None else -1
        stdout = response.stdout or ""
        stderr = response.stderr or ""
        return SandboxResult(
            stdout=_cap_output(stdout, cap),
            stderr=_cap_output(stderr, cap),
            exit_code=exit_code,
        )

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox via the Daytona filesystem API."""
        try:
            content = await asyncio.wait_for(self._sandbox.fs.download_file(path), timeout=timeout)
        except DaytonaNotFoundError as e:
            raise SandboxTerminatedError(str(e)) from e
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

        if max_bytes is not None and len(content) > max_bytes:
            content = content[:max_bytes]
        return SandboxResult(
            stdout=content.decode("utf-8", errors="replace"),
            stderr="",
            exit_code=0,
        )

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Write content to a file in the sandbox."""
        if isinstance(content, str):
            content = content.encode()

        try:
            await asyncio.wait_for(
                self._sandbox.fs.upload_files([FileUpload(source=content, destination=path)]),
                timeout=timeout,
            )
        except DaytonaNotFoundError as e:
            raise SandboxTerminatedError(str(e)) from e
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

        if executable:
            chmod = await self.run_command(f"chmod +x {shlex.quote(path)}", timeout=timeout)
            if chmod.exit_code != 0:
                return chmod
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def cleanup(self) -> None:
        """Delete the sandbox and close the owned Daytona client, idempotently."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        # Both steps are best-effort so a double-cleanup or a sandbox that
        # Daytona already auto-reaped does not surface as an exception.
        with contextlib.suppress(Exception):
            await self._client.delete(self._sandbox)

        if self._owns_client:
            with contextlib.suppress(Exception):
                await self._client.close()


# ---------------------------------------------------------------------------
# Harbor RL factory
# ---------------------------------------------------------------------------


async def daytona_sandbox_factory(env_dir: Path, timeout: int) -> DaytonaSandbox:
    """Create a Daytona sandbox from a Harbor task environment directory.

    Signature matches ``harbor_env.SandboxFactory``.

    Args:
        env_dir: Path to the task's ``environment/`` directory (must
            contain a ``Dockerfile``).
        timeout: Max wait time in seconds for sandbox creation.
    """
    dockerfile_path = env_dir / "Dockerfile"
    image = Image.from_dockerfile(dockerfile_path)
    return await DaytonaSandbox.create(image=image, timeout=timeout)


# ---------------------------------------------------------------------------
# Stateless code grading helper (for code_rl)
# ---------------------------------------------------------------------------


async def run_code_in_daytona(
    code: str,
    files: dict[str, str],
    timeout: float,
    language: str = "python",
    snapshot: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Execute code in an ephemeral Daytona sandbox for stateless grading.

    Return shape matches :meth:`SandboxFusionClient.run` so this slots
    into ``code_rl.code_grading`` as a peer backend without re-shaping
    the caller.

    Args:
        code: Main code to execute (entry point — written as ``run.py``).
        files: Additional files to include ``{filename: content}``.
        timeout: Execution timeout in seconds.
        language: Programming language (default: ``python``).
        snapshot: Optional pre-created Daytona snapshot name for fast cold
            start. Falls back to ``DAYTONA_SNAPSHOT`` env var, then to
            ``Image.debian_slim()``.

    Returns:
        Tuple of ``(success: bool, response: dict)``. ``success`` is True
        only when the process exited with code 0. ``response`` contains
        ``exit_code``, ``stdout``, ``stderr``.
    """
    if language != "python":
        return False, {"error": f"Unsupported language for Daytona grading: {language!r}"}

    sandbox: DaytonaSandbox | None = None
    try:
        # Allow a generous creation budget on top of the execution timeout
        # so a slow first-time image build does not kill the sandbox before
        # the user's code starts.
        #
        # The default image installs numpy because the code_rl test harness
        # (`testing_util.py`) imports it. Users passing a custom `snapshot=`
        # are responsible for providing the runtime deps they need.
        image: Image | None = None
        if snapshot is None and not os.getenv("DAYTONA_SNAPSHOT"):
            image = Image.debian_slim().pip_install("numpy")
        sandbox = await DaytonaSandbox.create(
            image=image, snapshot=snapshot, timeout=int(timeout) + 60
        )

        all_files = [
            ("/workspace/run.py", code),
            *[(f"/workspace/{name}", content) for name, content in files.items()],
        ]
        await asyncio.gather(*(sandbox.write_file(path, content) for path, content in all_files))

        result = await sandbox.run_command(
            "python run.py",
            workdir="/workspace",
            timeout=int(timeout),
        )
    except Exception as e:
        return False, {"error": str(e)}
    finally:
        if sandbox is not None:
            await sandbox.cleanup()

    success = result.exit_code == 0
    return success, {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


__all__ = [
    "DaytonaSandbox",
    "daytona_sandbox_factory",
    "run_code_in_daytona",
]
