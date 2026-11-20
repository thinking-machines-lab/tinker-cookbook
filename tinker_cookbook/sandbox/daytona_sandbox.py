"""
Thin wrapper around Daytona Sandbox API.

Daytona provides cloud-based sandboxed execution environments.
Requires Daytona authentication: ``export DAYTONA_API_KEY=...``
(or ``DAYTONA_JWT_TOKEN`` + ``DAYTONA_ORGANIZATION_ID``).

Supports two usage modes:

- Persistent: a sandbox whose filesystem and background processes last for
  its lifetime, for multi-turn agentic workloads (``DaytonaSandbox``).
- Ephemeral: a sandbox per call, for one-shot code grading
  (``run_code_in_daytona``).

Configuration via environment variables:
    DAYTONA_API_URL: API endpoint (default: https://app.daytona.io/api)
    DAYTONA_TARGET: Target region for sandbox placement
    DAYTONA_SNAPSHOT: Pre-created snapshot to boot when neither an image nor a
        snapshot is passed (e.g. for code grading). Not required: Daytona caches
        image builds by content hash, so it builds each image once.
    DAYTONA_GRADING_CONCURRENCY: Max grading sandboxes alive at once in
        ``run_code_in_daytona`` (default: 32)

See: https://www.daytona.io/docs
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
import shlex
import time
import tomllib
import uuid
import weakref
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar, cast

try:
    from daytona import (
        AsyncDaytona,
        AsyncSandbox,
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        DaytonaAuthenticationError,
        DaytonaAuthorizationError,
        DaytonaBadRequestError,
        DaytonaConflictError,
        DaytonaFileNotFoundError,
        DaytonaForbiddenError,
        DaytonaNotFoundError,
        DaytonaRateLimitError,
        DaytonaTimeoutError,
        DaytonaUnprocessableEntityError,
        DaytonaValidationError,
        Image,
        Resources,
        SandboxState,
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

_DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024
# Python of the default image. Pinned rather than following the local interpreter, which
# Daytona's images may not support, so every machine builds (and reuses) the same image.
_DEFAULT_PYTHON = "3.12"
_CREATE_ATTEMPTS = 5  # while Daytona rate-limits sandbox creation
_DELETE_ATTEMPTS = 5  # after a create that did not complete: about 5 seconds in all
_REJECTED = (
    DaytonaAuthenticationError,
    DaytonaAuthorizationError,
    DaytonaBadRequestError,
    DaytonaForbiddenError,
    DaytonaUnprocessableEntityError,
    DaytonaValidationError,
)
# How long past a command's own timeout the client waits for its result.
_CLIENT_TIMEOUT_SLACK_SECONDS = 30
# Bound on each HTTP request run_command makes. Commands themselves run asynchronously,
# as Daytona drops responses to requests held open for several minutes.
_REQUEST_TIMEOUT_SECONDS = 30
_MAX_POLL_INTERVAL_SECONDS = 1.0

# What each run_command executes, via ``bash -c``, in a Daytona session of its own
# ($1: timeout in seconds, $2: the command, $3: max bytes per stream, $4: _CAPTURE_SCRIPT):
# - a fresh login shell per call, as ModalSandbox runs ``bash -lc``, so the working
#   directory, variables and ``exit`` never carry over between calls. Its stdin is
#   empty, so a command that prompts for input fails instead of waiting for it;
# - ``timeout`` stops the command after $1 seconds and runs it in a process group of
#   its own, so background processes it starts outlive the session. On a timeout
#   the whole group is killed, including children that outlive SIGTERM;
# - the call ends when the command exits, not when its output closes: a background
#   process started by the command can hold the output open for as long as it runs
#   (``cd /app && server > log &`` runs ``cd`` and the server in a background bash
#   whose output is the command's). If the output is still open once the command has
#   exited, the script waits up to a second for output on its way, then returns.
_RUN_SCRIPT = (
    'd=$(mktemp -d "${TMPDIR:-/tmp}/.run_command.XXXXXX") || exit 1\n'
    "start=$SECONDS\n"
    'timeout -k 5 "$1" bash -c "$4" capture "$d" "$3" "$2" < /dev/null > /dev/null 2>&1 &\n'
    "pid=$!\n"
    'wait "$pid"; code=$?\n'
    'if [ $((SECONDS - start)) -ge "$1" ] && { [ "$code" -eq 124 ] || [ "$code" -eq 137 ]; }; then\n'
    '  kill -KILL -- "-$pid" 2> /dev/null\n'
    "else\n"
    "  for _ in 1 2 3 4 5 6 7 8 9 10; do\n"
    '    [ -e "$d/out.done" ] && [ -e "$d/err.done" ] && break\n'
    "    sleep 0.1\n"
    "  done\n"
    "fi\n"
    'cat "$d/out" 2> /dev/null\n'
    'cat "$d/err" >&2 2> /dev/null\n'
    'rm -rf "$d"\n'
    'exit "$code"'
)

# Runs the command under ``timeout``, in the command's process group ($1: directory for
# the captured output, $2: max bytes per stream, $3: the command). Each stream goes to a
# capture process that writes its first $2 bytes to a file as they arrive (``dd bs=1``,
# so nothing waits in a buffer), marks the file done when the stream ends, and drains
# the rest. Being in the command's group, the capture processes keep draining output of
# background processes after the call has returned, and die with the group on a timeout.
_CAPTURE_SCRIPT = (
    'exec {o}> >(dd bs=1 count="$2" of="$1/out" 2> /dev/null; : > "$1/out.done"; cat > /dev/null)\n'
    'exec {e}> >(exec {o}>&-; dd bs=1 count="$2" of="$1/err" 2> /dev/null; : > "$1/err.done"; cat > /dev/null)\n'
    'bash -lc "$3" < /dev/null >&"$o" 2>&"$e" {o}>&- {e}>&-'
)

_BUILDING_STATES = frozenset({SandboxState.PENDING_BUILD, SandboxState.BUILDING_SNAPSHOT})

# A sandbox in any of these states runs no more commands (ephemeral ones are deleted on stop).
_GONE_STATES = frozenset(
    {
        SandboxState.STOPPING,
        SandboxState.STOPPED,
        SandboxState.DESTROYING,
        SandboxState.DESTROYED,
        SandboxState.ARCHIVING,
        SandboxState.ARCHIVED,
        SandboxState.ERROR,
        SandboxState.BUILD_FAILED,
    }
)


def _cap_output(text: str, max_bytes: int) -> str:
    """Cap a string to *max_bytes* bytes of UTF-8, decoding safely at the boundary."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="replace")


_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])


def _cancellation_as_terminated(fn: _F) -> _F:
    """Turn a cancellation raised inside a Daytona call into ``SandboxTerminatedError``.

    As in ``modal_sandbox``: an ``asyncio.CancelledError`` is a ``BaseException``, so it
    would escape the ``except Exception`` handlers of rollout error tolerance and cancel
    the whole training step. Cancellation of the calling task itself still propagates.
    """

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            raise SandboxTerminatedError(
                f"Daytona sandbox call {fn.__name__} was cancelled"
            ) from None

    return cast(_F, wrapped)


async def _delete_named(client: AsyncDaytona, name: str) -> None:
    """Delete the sandbox called *name*, if a create that did not complete made one.

    Retried for a few seconds: a create cancelled while its request was in flight can
    still make the sandbox a moment later, and Daytona refuses the delete while another
    state change, such as the sandbox's start, is in progress.

    A sandbox still building its image is not deleted: that cancels the build, and later
    creates from the same image can then fail for a while. It finishes the build instead,
    then stops after a minute of idling and, being ephemeral, is deleted.
    """
    error: Exception | None = None
    for attempt in range(_DELETE_ATTEMPTS):
        if attempt:
            await asyncio.sleep(attempt / 2)
        try:
            sandbox = await client.get(name)
            if getattr(sandbox.desired_state, "value", None) == "destroyed":
                return
            if sandbox.state in _BUILDING_STATES:
                await sandbox.set_autostop_interval(1)
                return
            await client.delete(sandbox)
            return
        except DaytonaNotFoundError:
            error = None
        except DaytonaConflictError as e:
            error = e
        except Exception as e:
            error = e
            break
    if error is not None:
        logger.warning(
            "Could not delete Daytona sandbox %s after its create did not complete; "
            "it runs until its time-to-live: %s",
            name,
            error,
        )


async def _create_sandbox(
    client: AsyncDaytona,
    params: CreateSandboxFromImageParams | CreateSandboxFromSnapshotParams,
    create_timeout: int,
) -> AsyncSandbox:
    """``client.create(params)``, retried while rate limited.

    A create that fails or is cancelled after Daytona made the sandbox (a failed image
    build, a start that timed out, Ctrl-C during a training step) removes it by its
    unique name with ``_delete_named``, rather than leaving it until its time-to-live.
    That is shielded, so it also completes while ``asyncio.run`` cancels the remaining
    tasks on shutdown.
    """
    assert params.name is not None
    for attempt in range(_CREATE_ATTEMPTS):
        try:
            return await client.create(params, timeout=create_timeout)
        except DaytonaRateLimitError:
            if attempt + 1 == _CREATE_ATTEMPTS:
                raise
            await asyncio.sleep(2**attempt)
        except _REJECTED:
            raise  # the API refused the request, so it made no sandbox
        except BaseException:
            await asyncio.shield(_delete_named(client, params.name))
            raise
    raise AssertionError("unreachable")


class DaytonaSandbox(SandboxInterface):
    """
    Daytona sandbox that implements :class:`SandboxInterface`.

    The filesystem and background processes persist for the sandbox's lifetime, while
    each ``run_command`` runs in a fresh ``bash -lc``, as with ``ModalSandbox``: shell
    state such as the working directory or exported variables does not carry over
    between calls. Images need ``bash`` and coreutils (``timeout``, ``dd``, ``mktemp``).

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
        sandbox: AsyncSandbox,
        max_stream_output_bytes: int,
        owns_client: bool,
    ) -> None:
        self._client = client
        self._sandbox = sandbox
        self._max_stream_output_bytes = max_stream_output_bytes
        self._owns_client = owns_client
        self._cleaned_up = False

    @classmethod
    @_cancellation_as_terminated
    async def create(
        cls,
        *,
        image: Image | str | None = None,
        snapshot: str | None = None,
        timeout: int = 600,
        create_timeout: int = 600,
        resources: Resources | None = None,
        labels: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
        target: str | None = None,
        max_stream_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        client: AsyncDaytona | None = None,
    ) -> DaytonaSandbox:
        """Create a new Daytona sandbox.

        The sandbox is ephemeral and lives at most ``timeout`` seconds: it never stops
        for being idle, and Daytona deletes it when its lifetime is up if ``cleanup()``
        has not already.

        Args:
            image: Image to build the sandbox from. Daytona caches builds by content
                hash, so many sandboxes from the same image build it once.
            snapshot: Name of a pre-created Daytona snapshot. Mutually exclusive with
                *image*. With neither, ``DAYTONA_SNAPSHOT`` is booted if set, else
                ``Image.debian_slim()``.
            timeout: Sandbox lifetime in seconds (rounded up to whole minutes), as in
                ``ModalSandbox.create``.
            create_timeout: Max seconds to wait for the image build and sandbox start.
            resources: CPU, memory (GiB) and disk (GiB) of a sandbox built from an
                image; Daytona's defaults if unset. A snapshot fixes its own size.
            labels: Custom labels attached to the sandbox, besides
                ``created-by=tinker-cookbook``.
            env_vars: Environment variables set inside the sandbox.
            target: Target region for sandbox placement.
            max_stream_output_bytes: Cap per-stream output at this many
                bytes. Defaults to 128 KB.
            client: Existing ``AsyncDaytona`` client to reuse. If ``None``,
                a new client is created and owned by this sandbox.
        """
        if image is not None and snapshot is not None:
            raise ValueError("Provide either image or snapshot, not both.")
        if image is None and snapshot is None:
            snapshot = os.getenv("DAYTONA_SNAPSHOT") or None
        if snapshot is not None and resources is not None:
            raise ValueError("A snapshot fixes the sandbox size; resources need an image.")

        name = f"tinker-cookbook-{uuid.uuid4().hex}"
        all_labels = {"created-by": "tinker-cookbook", **(labels or {})}
        ttl_minutes = max(1, math.ceil(timeout / 60))
        params: CreateSandboxFromImageParams | CreateSandboxFromSnapshotParams
        if snapshot is not None:
            params = CreateSandboxFromSnapshotParams(
                name=name,
                snapshot=snapshot,
                labels=all_labels,
                env_vars=env_vars,
                target=target,
                ephemeral=True,
                auto_stop_interval=0,
                ttl_minutes=ttl_minutes,
            )
        else:
            params = CreateSandboxFromImageParams(
                name=name,
                image=image if image is not None else Image.debian_slim(_DEFAULT_PYTHON),
                resources=resources,
                labels=all_labels,
                env_vars=env_vars,
                target=target,
                ephemeral=True,
                auto_stop_interval=0,
                ttl_minutes=ttl_minutes,
            )

        owns_client = client is None
        daytona = client if client is not None else AsyncDaytona()
        try:
            sandbox = await _create_sandbox(daytona, params, create_timeout)
        except BaseException:
            if owns_client:
                await daytona.close()
            raise

        return cls(
            client=daytona,
            sandbox=sandbox,
            max_stream_output_bytes=max_stream_output_bytes,
            owns_client=owns_client,
        )

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.id

    def _raise_if_cleaned_up(self) -> None:
        if self._cleaned_up:
            raise SandboxTerminatedError(f"Daytona sandbox {self.sandbox_id} was cleaned up")

    async def _is_gone(self) -> bool:
        """Whether the sandbox is deleted, or stopped or failed and so about to be."""
        try:
            await self._sandbox.refresh_data()
        except DaytonaNotFoundError:
            return True
        except Exception:
            return False  # state unknown
        return self._sandbox.state in _GONE_STATES

    async def _raise_if_gone(self, error: Exception) -> None:
        """Raise ``SandboxTerminatedError`` if *error* came from a sandbox that is gone.

        A sandbox that is gone fails differently depending on the call and the moment: a
        404 from commands once it is deleted, a plain ``DaytonaError`` from file transfers,
        a 502 while it is still shutting down. So the sandbox's state settles it.
        """
        if await self._is_gone():
            raise SandboxTerminatedError(
                f"Daytona sandbox {self.sandbox_id} is no longer running: {error}"
            ) from error

    @_cancellation_as_terminated
    async def send_heartbeat(self, timeout: int = 30) -> None:
        """Check that the sandbox is still running.

        Daytona sandboxes created here never stop for being idle, so the heartbeat only
        has to detect a sandbox that is gone.
        """
        self._raise_if_cleaned_up()
        try:
            await asyncio.wait_for(self._sandbox.refresh_data(), timeout=timeout)
        except DaytonaNotFoundError as e:
            raise SandboxTerminatedError(str(e)) from e
        if self._sandbox.state in _GONE_STATES:
            raise SandboxTerminatedError(
                f"Daytona sandbox {self.sandbox_id} is {self._sandbox.state}"
            )

    @_cancellation_as_terminated
    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox.

        Each call runs in a fresh login shell with an empty stdin, in *workdir* or else the
        image's WORKDIR. A command still running after *timeout* seconds is killed, along
        with what it started in its process group, and exits with 124. The call returns
        once the command exits: background processes it started keep running, and what
        they print after that is dropped.
        """
        self._raise_if_cleaned_up()
        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes
        script = shlex.join(
            [
                "bash",
                "-c",
                _RUN_SCRIPT,
                "run_command",
                str(timeout),
                command,
                str(cap),
                _CAPTURE_SCRIPT,
            ]
        )
        if workdir is not None:
            script = f"cd {shlex.quote(workdir)} && {script}"

        process = self._sandbox.process
        session_id = f"tinker-cookbook-{uuid.uuid4().hex}"
        deadline = time.monotonic() + timeout + _CLIENT_TIMEOUT_SLACK_SECONDS
        try:
            await process.create_session(session_id, request_timeout=_REQUEST_TIMEOUT_SECONDS)
            try:
                started = await process.execute_session_command(
                    session_id,
                    SessionExecuteRequest(command=script, run_async=True),
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
                assert started.cmd_id is not None
                exit_code = await self._wait_for_exit(session_id, started.cmd_id, deadline)
                if exit_code is None:
                    return SandboxResult(
                        stdout="", stderr=f"Command timed out after {timeout}s", exit_code=124
                    )
                logs = await process.get_session_command_logs(
                    session_id, started.cmd_id, request_timeout=_REQUEST_TIMEOUT_SECONDS
                )
            finally:
                # Ends whatever is still attached to the session, e.g. a command still
                # running past the deadline. Background processes are in their own group.
                try:
                    await process.delete_session(
                        session_id, request_timeout=_REQUEST_TIMEOUT_SECONDS
                    )
                except Exception as e:
                    logger.debug("Could not delete Daytona session %s: %s", session_id, e)
        except Exception as e:
            await self._raise_if_gone(e)
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

        return SandboxResult(
            stdout=_cap_output(logs.stdout or "", cap),
            stderr=_cap_output(logs.stderr or "", cap),
            exit_code=exit_code,
        )

    async def _wait_for_exit(self, session_id: str, cmd_id: str, deadline: float) -> int | None:
        """Poll an asynchronous session command until it exits, or ``None`` at *deadline*."""
        interval = 0.1
        while True:
            try:
                command = await self._sandbox.process.get_session_command(
                    session_id, cmd_id, request_timeout=_REQUEST_TIMEOUT_SECONDS
                )
                if command.exit_code is not None:
                    return command.exit_code
            except DaytonaTimeoutError:
                pass  # a slow poll; the next one may get through
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(interval, remaining))
            interval = min(interval * 2, _MAX_POLL_INTERVAL_SECONDS)

    @_cancellation_as_terminated
    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox via the Daytona filesystem API."""
        self._raise_if_cleaned_up()
        try:
            content = await self._sandbox.fs.download_file(path, timeout)
        except DaytonaFileNotFoundError as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        except Exception as e:
            await self._raise_if_gone(e)
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

        if max_bytes is not None:
            content = content[:max_bytes]
        return SandboxResult(
            stdout=content.decode("utf-8", errors="replace"),
            stderr="",
            exit_code=0,
        )

    @_cancellation_as_terminated
    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Write content to a file in the sandbox, creating parent directories."""
        self._raise_if_cleaned_up()
        if isinstance(content, str):
            content = content.encode()

        try:
            await self._sandbox.fs.upload_file(content, path, timeout)
            if executable:
                await self._sandbox.fs.set_file_permissions(path, mode="755")
        except Exception as e:
            await self._raise_if_gone(e)
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    @_cancellation_as_terminated
    async def cleanup(self) -> None:
        """Delete the sandbox and close the owned Daytona client, idempotently."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        try:
            await self._client.delete(self._sandbox)
        except DaytonaNotFoundError:
            pass  # already deleted, e.g. at the end of its lifetime
        except Exception as e:
            if not await self._is_gone():
                logger.warning(
                    "Could not delete Daytona sandbox %s; it runs until its time-to-live: %s",
                    self.sandbox_id,
                    e,
                )
        finally:
            if self._owns_client:
                await self._client.close()


# ---------------------------------------------------------------------------
# Harbor RL factory
# ---------------------------------------------------------------------------


def _size_mb(size: str) -> float:
    """A Harbor size such as ``"2G"`` or ``"512M"`` in megabytes."""
    units = {"K": 1 / 1024, "M": 1, "G": 1024}
    value = size.strip().upper()
    if value[-1:] not in units:
        raise ValueError(f"Invalid size {size!r}: expected a value like '2G' or '512M'")
    return float(value[:-1]) * units[value[-1]]


def _task_resources(task_toml: Path) -> Resources | None:
    """The CPUs, memory and storage a Harbor task asks for in its ``task.toml``."""
    if not task_toml.is_file():
        return None
    env = tomllib.loads(task_toml.read_text()).get("environment", {})
    cpus = env.get("cpus")
    memory_mb = env.get("memory_mb") or (_size_mb(env["memory"]) if "memory" in env else None)
    storage_mb = env.get("storage_mb") or (_size_mb(env["storage"]) if "storage" in env else None)
    if not (cpus or memory_mb or storage_mb):
        return None
    return Resources(
        cpu=math.ceil(cpus) if cpus else None,
        memory=math.ceil(memory_mb / 1024) if memory_mb else None,
        disk=math.ceil(storage_mb / 1024) if storage_mb else None,
    )


async def daytona_sandbox_factory(env_dir: Path, timeout: int) -> DaytonaSandbox:
    """Create a Daytona sandbox from a Harbor task environment directory.

    Signature matches ``harbor_env.SandboxFactory``. Daytona builds (and caches) the
    task's Dockerfile with ``env_dir`` as the build context, and the sandbox gets the
    ``cpus``, ``memory`` and ``storage`` set in the ``[environment]`` table of the
    task's ``task.toml``, which Terminal-Bench tasks set above Daytona's defaults.

    Args:
        env_dir: Path to the task's ``environment/`` directory (must
            contain a ``Dockerfile``).
        timeout: Sandbox lifetime in seconds.
    """
    image = Image.from_dockerfile(env_dir / "Dockerfile")
    resources = _task_resources(env_dir.parent / "task.toml")
    return await DaytonaSandbox.create(image=image, timeout=timeout, resources=resources)


# ---------------------------------------------------------------------------
# Stateless code grading helper (for code_rl)
# ---------------------------------------------------------------------------


_grading_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _grading_slots_for_loop() -> asyncio.Semaphore:
    """Bounds the grading sandboxes alive at once, as ``MODAL_POOL_SIZE`` does for Modal."""
    loop = asyncio.get_running_loop()
    if loop not in _grading_slots:
        limit = int(os.getenv("DAYTONA_GRADING_CONCURRENCY", "32"))
        _grading_slots[loop] = asyncio.Semaphore(limit)
    return _grading_slots[loop]


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
    the caller. At most ``DAYTONA_GRADING_CONCURRENCY`` (default 32) calls
    hold a sandbox at once; the rest wait for a free slot.

    Args:
        code: Main code to execute (entry point — written as ``run.py``).
        files: Additional files to include ``{filename: content}``.
        timeout: Execution timeout in seconds.
        language: Programming language (default: ``python``).
        snapshot: Optional pre-created Daytona snapshot name for fast cold
            start. Falls back to ``DAYTONA_SNAPSHOT`` env var, then to
            ``Image.debian_slim()`` with numpy.

    Returns:
        Tuple of ``(success: bool, response: dict)``. ``success`` is True
        only when the process exited with code 0. ``response`` contains
        ``exit_code``, ``stdout``, ``stderr``.
    """
    if language != "python":
        return False, {"error": f"Unsupported language for Daytona grading: {language!r}"}

    # The default image installs numpy because the code_rl test harness
    # (`testing_util.py`) imports it. Users passing a custom `snapshot=`
    # are responsible for providing the runtime deps they need.
    image: Image | None = None
    if snapshot is None and not os.getenv("DAYTONA_SNAPSHOT"):
        image = Image.debian_slim(_DEFAULT_PYTHON).pip_install("numpy")

    async with _grading_slots_for_loop():
        sandbox: DaytonaSandbox | None = None
        try:
            # The lifetime only bounds a sandbox whose cleanup never ran.
            sandbox = await DaytonaSandbox.create(
                image=image, snapshot=snapshot, timeout=int(timeout) + 600
            )
            all_files = {"run.py": code, **files}
            await asyncio.gather(
                *(
                    sandbox.write_file(f"/workspace/{name}", content)
                    for name, content in all_files.items()
                )
            )
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
