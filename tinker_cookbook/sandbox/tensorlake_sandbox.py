"""
Thin wrapper around the Tensorlake Sandbox API.

Tensorlake provides cloud microVM sandboxes that start in well under a second,
support snapshot/restore, and can expose ports inside the sandbox to the client.
Requires a Tensorlake API key: set ``TENSORLAKE_API_KEY``.

Configuration via environment variables:
    TENSORLAKE_API_KEY: API key (required)
    TENSORLAKE_API_URL: API endpoint (default: https://api.tensorlake.ai)
    TENSORLAKE_MAX_CONCURRENCY: Max concurrent sandboxes in TensorlakeSandboxPool (default: 32)

See: https://docs.tensorlake.ai/sandboxes/introduction
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import logging
import os
import re
import shlex
import shutil
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

try:
    from tensorlake.sandbox import (
        AsyncSandbox,
        AsyncSandboxClient,
        AsyncTcpTunnel,
        RemoteAPIError,
        SandboxNotFoundError,
    )
    from tensorlake.sandbox.models import CheckpointType, SnapshotWaitCondition
except ImportError:
    raise ImportError(
        "tensorlake is required for TensorlakeSandbox. "
        "Install it with: uv pip install 'tinker-cookbook[tensorlake] @ "
        "git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'"
    ) from None

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxInterface,
    SandboxResult,
    SandboxTerminatedError,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024
_UNCAPPED_BYTES = 2**40
_TMP_DOCKERFILE_RE = re.compile(r"\.tinker-[0-9a-f]{8}\.Dockerfile$")
_TERMINATED_RE = re.compile(
    r"sandbox '[^']*' (not found|not running|terminated)|sandbox (has been )?terminated", re.I
)


def _is_sandbox_terminated(e: BaseException) -> bool:
    """Check if an exception indicates the sandbox has died."""
    if isinstance(e, SandboxNotFoundError):
        return True
    # The SDK parses lifecycle responses (status "failed" or "terminated") and
    # sets ``sandbox_id`` only for them.
    if isinstance(e, RemoteAPIError) and e.sandbox_id is not None:
        return True
    # A dead sandbox gives a 404 "Sandbox '<id>' not found or not running".
    # A missing file also gives a 404, so match on the message.
    return _TERMINATED_RE.search(str(e)) is not None


# AsyncSandbox.run() buffers all output on the client before it returns, so cap
# stdout/stderr inside the sandbox. Each stream goes through ``head -c N``, then
# ``cat`` discards the rest so the program does not get SIGPIPE. ``stdbuf -o0``
# stops ``head`` from buffering, so output before a timeout kill is kept.
# $1 is the command and $2 is the cap. The command does not inherit fds 5 and 6,
# so a background process that redirects stdout/stderr does not keep the pipes
# open. ``wait`` on process-substitution PIDs needs bash 4.4+.
_CAPPED_RUN_SCRIPT = """\
n=$2
cap() {
  if command -v stdbuf >/dev/null 2>&1; then stdbuf -o0 head -c "$n"; else head -c "$n"; fi
  cat >/dev/null
}
exec 5> >(cap); p1=$!
exec 6> >(cap >&2); p2=$!
bash -lc "$1" >&5 2>&6 5>&- 6>&-; rc=$?
exec 5>&- 6>&-
wait "$p1" "$p2"
exit "$rc"
"""


def _cap(text: str, max_bytes: int) -> str:
    """Truncate *text* to at most *max_bytes* UTF-8 bytes."""
    data = text.encode()
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")


class TensorlakeSandbox(SandboxInterface):
    """
    Persistent Tensorlake sandbox for code execution. Implements SandboxInterface.

    Usage:
        sandbox = await TensorlakeSandbox.create()

        await sandbox.write_file("/workspace/code.py", "print('hello')")
        result = await sandbox.run_command("python3 /workspace/code.py")
        print(result.stdout)

        await sandbox.cleanup()

    Beyond SandboxInterface, this class supports long-running services
    (``start_process`` + ``open_port``) and snapshots (``checkpoint`` +
    ``create(snapshot_id=...)``), which agent environments such as MCP
    gateways need.
    """

    def __init__(
        self,
        sandbox: AsyncSandbox,
        max_stream_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        user: str | None = "root",
    ) -> None:
        self._sandbox = sandbox
        self._user = user
        self._max_stream_output_bytes = max_stream_output_bytes
        self._tunnels: dict[int, AsyncTcpTunnel] = {}
        self._tunnel_lock = asyncio.Lock()
        self._cleaned_up = False
        self._terminate_attempted = False

    @classmethod
    async def create(
        cls,
        image: str | None = None,
        timeout: int = 600,
        cpus: float | None = None,
        memory_mb: int | None = None,
        snapshot_id: str | None = None,
        allow_internet_access: bool = True,
        allow_out: list[str] | None = None,
        max_stream_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        user: str | None = "root",
    ) -> TensorlakeSandbox:
        """Create a new Tensorlake sandbox.

        Args:
            image: Registered sandbox image name. None uses the Tensorlake default
                image (Ubuntu with Python 3). See ``build_image_from_dockerfile``.
            timeout: Lifetime of the sandbox in seconds.
            cpus: CPUs for the sandbox. None uses the server default.
            memory_mb: Memory for the sandbox in MB. None uses the server default.
            snapshot_id: Restore from this snapshot (see ``checkpoint``) instead
                of booting a fresh image.
            allow_internet_access: If False, block all outbound network traffic.
            allow_out: If set, allow outbound traffic only to these destinations.
            max_stream_output_bytes: Default cap for stdout/stderr per command.
            user: User that runs commands. Defaults to root, like Modal. None
                uses the image's ``USER`` (the default image uses a non-root
                user). File reads and writes that the image's user cannot do
                fall back to this user.
        """
        sandbox = await AsyncSandbox.create(
            image=image,
            timeout_secs=timeout,
            cpus=cpus,
            memory_mb=memory_mb,
            snapshot_id=snapshot_id,
            allow_internet_access=allow_internet_access,
            allow_out=allow_out,
            # The SDK's HTTP timeout (300 s by default) also bounds each
            # command, so let it cover the sandbox lifetime.
            request_timeout=timeout,
        )
        return cls(sandbox=sandbox, max_stream_output_bytes=max_stream_output_bytes, user=user)

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.sandbox_id

    def _check_alive(self) -> None:
        # After ``cleanup`` the server may still accept a command for a moment,
        # so do not rely on it to reject the call.
        if self._cleaned_up:
            raise SandboxTerminatedError(f"Sandbox {self.sandbox_id} has been cleaned up")

    async def send_heartbeat(self, timeout: int = 30) -> None:
        self._check_alive()
        try:
            await asyncio.wait_for(self._sandbox.run("true"), timeout=timeout)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            raise

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox.

        On timeout, the process is killed and ``exit_code`` is -9.
        """
        self._check_alive()
        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes
        try:
            result = await self._sandbox.run(
                "bash",
                ["-c", _CAPPED_RUN_SCRIPT, "bash", command, str(cap)],
                working_dir=workdir,
                timeout=timeout,
                user=self._user,
            )
            return SandboxResult(
                stdout=_cap(result.stdout, cap),
                stderr=_cap(result.stderr, cap),
                exit_code=result.exit_code,
            )
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox."""
        self._check_alive()
        if max_bytes is not None:
            # The file API downloads the full file, so read only the prefix in the sandbox.
            return await self._read_file_as_user(f"head -c {max_bytes}", path, timeout)
        try:
            data = await asyncio.wait_for(self._sandbox.read_file(path), timeout=timeout)
        except TimeoutError:
            return SandboxResult(
                stdout="", stderr=f"read_file timed out after {timeout}s", exit_code=-1
            )
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            # The file API runs as the image's default user, which may not be
            # able to read the file. Read it as ``self._user`` instead.
            return await self._read_file_as_user("cat", path, timeout)
        content: bytes = data.value
        return SandboxResult(
            stdout=content.decode("utf-8", errors="replace"), stderr="", exit_code=0
        )

    async def _read_file_as_user(self, reader: str, path: str, timeout: int) -> SandboxResult:
        """Read *path* with *reader* (``cat`` or ``head -c N``) as ``self._user``.

        The SDK joins output lines with a newline and drops the trailing one,
        so send the content as base64. The reader limits the size, so do not
        cap the output.
        """
        result = await self.run_command(
            f"set -o pipefail; {reader} {shlex.quote(path)} | base64",
            timeout=timeout,
            max_output_bytes=_UNCAPPED_BYTES,
        )
        if result.exit_code != 0:
            return SandboxResult(stdout="", stderr=result.stderr, exit_code=result.exit_code)
        content = base64.b64decode(result.stdout)
        return SandboxResult(
            stdout=content.decode("utf-8", errors="replace"), stderr="", exit_code=0
        )

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Write content to a file in the sandbox. Parent directories are created."""
        self._check_alive()
        if isinstance(content, str):
            content = content.encode()
        try:
            await asyncio.wait_for(self._sandbox.write_file(path, content), timeout=timeout)
        except TimeoutError:
            return SandboxResult(
                stdout="", stderr=f"write_file timed out after {timeout}s", exit_code=-1
            )
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            # The file API runs as the image's default user, which may not be
            # able to write to *path*. Stage the file in /tmp and move it
            # into place as ``self._user``.
            staged = f"/tmp/.tinker-upload-{uuid.uuid4().hex}"
            try:
                await asyncio.wait_for(self._sandbox.write_file(staged, content), timeout=timeout)
            except Exception as e2:
                if _is_sandbox_terminated(e2):
                    raise SandboxTerminatedError(str(e2)) from e2
                return SandboxResult(stdout="", stderr=str(e2), exit_code=-1)
            quoted = shlex.quote(path)
            result = await self.run_command(
                f"mkdir -p {shlex.quote(os.path.dirname(path) or '/')} && mv {staged} {quoted}",
                timeout=timeout,
            )
            if result.exit_code != 0:
                return result
        if executable:
            return await self.run_command(f"chmod +x {shlex.quote(path)}", timeout=timeout)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def start_process(
        self,
        command: str,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> int:
        """Start a long-running shell command in the background. Returns its PID.

        Use this for services inside the sandbox (for example an MCP gateway),
        then reach them with ``open_port``.
        """
        self._check_alive()
        try:
            proc = await self._sandbox.start_process(
                "bash", ["-lc", command], env=env, working_dir=workdir, user=self._user
            )
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            raise SandboxError(f"Failed to start process: {e}") from e
        return proc.pid

    async def open_port(self, port: int) -> str:
        """Return a local base URL (``http://127.0.0.1:<n>``) that reaches *port* in the sandbox.

        The connection goes through an authenticated tunnel, so the port is not
        exposed to the internet. Tunnels are closed by ``cleanup``.
        """
        async with self._tunnel_lock:
            self._check_alive()
            tunnel = self._tunnels.get(port)
            if tunnel is None or tunnel.closed:
                tunnel = await self._sandbox.create_tunnel(port, local_port=0)
                self._tunnels[port] = tunnel
        return f"http://{tunnel.local_host}:{tunnel.local_port}"

    async def checkpoint(self, timeout: float = 300, include_memory: bool = True) -> str:
        """Snapshot the sandbox state and return the snapshot ID.

        Pass the ID to ``TensorlakeSandbox.create(snapshot_id=...)`` to start
        new sandboxes from this state (for example, one per rollout in a group).

        Args:
            timeout: Max seconds to wait for the snapshot.
            include_memory: If True, save VM memory and running processes as
                well as the filesystem. A restore then resumes where the
                sandbox left off. If False, save only the filesystem. A
                restore then cold-boots, like an image. Filesystem restores
                start several times faster on hosts that have not seen the
                snapshot before, so use ``False`` when only installed files
                matter.
        """
        self._check_alive()
        checkpoint_type = CheckpointType.MEMORY if include_memory else CheckpointType.FILESYSTEM
        # The SDK default returns once the snapshot is ready on the host that
        # took it. Restores on other hosts are much slower until the upload
        # completes, so wait for that here.
        snapshot = await self._sandbox.checkpoint(
            timeout=timeout,
            checkpoint_type=checkpoint_type,
            wait_until=SnapshotWaitCondition.COMPLETED,
        )
        if snapshot is None:
            raise SandboxError("Tensorlake checkpoint returned no snapshot")
        return snapshot.snapshot_id

    async def cleanup(self) -> None:
        """Close tunnels and terminate the sandbox. Safe to call multiple times."""
        if self._cleaned_up:
            return
        async with self._tunnel_lock:
            for tunnel in self._tunnels.values():
                with contextlib.suppress(Exception):
                    await tunnel.close()
            self._tunnels.clear()
        try:
            if self._terminate_attempted:
                # ``AsyncSandbox.terminate`` drops its client before it sends
                # the delete, so a second call does nothing. Delete directly.
                async with AsyncSandboxClient(_internal=True) as client:
                    await client.delete(self.sandbox_id)
            else:
                self._terminate_attempted = True
                await self._sandbox.terminate()
        except Exception as e:
            if not _is_sandbox_terminated(e):
                raise
        self._cleaned_up = True


class TensorlakeSandboxPool:
    """
    Concurrency-limited executor for one-shot runs (for example, grading code).

    Tensorlake sandboxes start in under a second, so the pool does not keep
    warm sandboxes. Each call creates a fresh sandbox, runs the command, and
    terminates it. At most ``max_concurrency`` sandboxes run at once.

    Has the same ``run_in_workdir`` / ``terminate`` API as ModalSandboxPool.

    Configuration via environment variables:
        TENSORLAKE_MAX_CONCURRENCY: Max concurrent sandboxes (default: 32)
    """

    def __init__(
        self,
        *,
        max_concurrency: int | None = None,
        sandbox_timeout_secs: int = 1200,
        image: str | None = None,
        setup_command: str | None = None,
    ):
        """
        Args:
            max_concurrency: Max sandboxes that run at the same time.
            sandbox_timeout_secs: Lifetime of each sandbox in seconds.
            image: Registered sandbox image name. None uses the default image.
            setup_command: Optional shell command that runs once in a template
                sandbox (for example ``pip install numpy``). The result is
                snapshotted and every run starts from that snapshot.
        """
        self._max_concurrency = max_concurrency or int(
            os.getenv("TENSORLAKE_MAX_CONCURRENCY", "32")
        )
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._sandbox_timeout_secs = sandbox_timeout_secs
        self._image = image
        self._setup_command = setup_command
        self._snapshot_id: str | None = None
        # Concurrent callers share one setup task, so setup runs at most once.
        # After a failure, the next call tries again.
        self._setup_task: asyncio.Future[str] | None = None
        self._active: set[TensorlakeSandbox] = set()
        # Setup, creations, and releases in progress. ``terminate`` waits for
        # them, so no sandbox or snapshot is left behind after it returns.
        self._pending: set[asyncio.Future[Any]] = set()
        self._terminated = False

    def _check_terminated(self) -> None:
        if self._terminated:
            raise SandboxError("TensorlakeSandboxPool has been terminated.")

    def _track(self, coro: Coroutine[object, object, _T]) -> asyncio.Future[_T]:
        """Start *coro* as a task that ``terminate`` waits for.

        Callers await the task through ``asyncio.shield``, so a cancelled
        caller does not cancel the task. The pool keeps ownership of the
        task until it is done.
        """
        task = asyncio.ensure_future(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def _reclaim(self, creation: asyncio.Future[TensorlakeSandbox]) -> None:
        """Release the sandbox of a creation whose caller was cancelled."""
        try:
            sandbox = await creation
        except Exception:
            return  # No sandbox, or the creation already cleaned it up.
        await self._release(sandbox)

    async def _create_sandbox(self, snapshot_id: str | None = None) -> TensorlakeSandbox:
        """Create a sandbox and add it to ``_active``."""
        self._check_terminated()

        async def create() -> TensorlakeSandbox:
            sandbox = await TensorlakeSandbox.create(
                image=self._image, timeout=self._sandbox_timeout_secs, snapshot_id=snapshot_id
            )
            if self._terminated:
                await sandbox.cleanup()
                self._check_terminated()
            self._active.add(sandbox)
            return sandbox

        # The SDK can allocate the remote sandbox before ``create`` returns,
        # so a cancelled caller must not cancel the creation.
        creation = self._track(create())
        try:
            return await asyncio.shield(creation)
        except asyncio.CancelledError:
            self._track(self._reclaim(creation))
            raise

    async def _release(self, sandbox: TensorlakeSandbox) -> None:
        self._active.discard(sandbox)

        async def cleanup() -> None:
            try:
                await sandbox.cleanup()
            except Exception as e:
                logger.warning(f"Tensorlake sandbox cleanup failed: {e}")
                # ``terminate`` tries again.
                self._active.add(sandbox)

        await asyncio.shield(self._track(cleanup()))

    async def _setup(self, setup_command: str) -> str:
        """Run *setup_command* in a template sandbox and snapshot the result."""
        template = await self._create_sandbox()
        try:
            result = await template.run_command(setup_command, timeout=600)
            self._check_terminated()
            if result.exit_code != 0:
                raise SandboxError(
                    f"Tensorlake pool setup failed ({result.exit_code}): {result.stderr}"
                )
            # Only the installed files matter here, and a filesystem snapshot
            # restores as fast as an image.
            # Set ``_snapshot_id`` inside this tracked task, so ``terminate``
            # sees it after it waits for ``_pending``.
            self._snapshot_id = await template.checkpoint(include_memory=False)
            return self._snapshot_id
        finally:
            await self._release(template)

    async def _get_snapshot_id(self) -> str | None:
        """Run ``setup_command`` once and return the snapshot ID."""
        if self._setup_command is None:
            return None
        task = self._setup_task
        if task is None:
            task = self._setup_task = self._track(self._setup(self._setup_command))
        try:
            return await asyncio.shield(task)
        except Exception as e:
            if self._setup_task is task:
                logger.warning(f"Tensorlake pool setup failed; the next call tries again: {e}")
                self._setup_task = None
            raise

    async def run_in_workdir(
        self,
        files: dict[str, str],
        command: list[str],
        timeout: int | None = None,
    ) -> SandboxResult:
        """
        Execute command with files in a fresh sandbox.
        If ``max_concurrency`` sandboxes are busy, waits until one finishes.

        Args:
            files: Files to write {filename: content}
            command: Command and arguments (e.g., ["python", "run.py"])
            timeout: Execution timeout in seconds
        """
        self._check_terminated()
        snapshot_id = await self._get_snapshot_id()
        async with self._semaphore:
            sandbox = await self._create_sandbox(snapshot_id)
            try:
                # The file API runs as the image's default user, which may not
                # be root. /tmp is writable for any user, so each upload works
                # on the first try instead of the staged fallback in write_file.
                workdir = f"/tmp/{uuid.uuid4().hex[:12]}"
                if files:
                    results = await asyncio.gather(
                        *(
                            sandbox.write_file(f"{workdir}/{filename}", content)
                            for filename, content in files.items()
                        )
                    )
                    for r in results:
                        if r.exit_code != 0:
                            return r
                else:
                    await sandbox.run_command(f"mkdir -p {shlex.quote(workdir)}")
                return await sandbox.run_command(
                    shlex.join(command),
                    workdir=workdir,
                    timeout=timeout or self._sandbox_timeout_secs,
                )
            finally:
                await self._release(sandbox)

    async def terminate(self) -> None:
        """Stop accepting work and terminate all sandboxes, including ones being created."""
        self._terminated = True
        # Each creation cleans up its own sandbox when it sees ``_terminated``.
        # Pending tasks can start new ones (a reclaim or a release), so loop
        # until none are left.
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)
        active, self._active = list(self._active), set()
        await asyncio.gather(*(sb.cleanup() for sb in active), return_exceptions=True)
        snapshot_id, self._snapshot_id = self._snapshot_id, None
        if snapshot_id is not None:
            try:
                async with AsyncSandboxClient(_internal=True) as client:
                    await client.delete_snapshot(snapshot_id)
            except Exception as e:
                logger.warning(f"Failed to delete Tensorlake snapshot {snapshot_id}: {e}")


def build_image_from_dockerfile(
    dockerfile_path: str | Path,
    context_dir: str | Path | None = None,
    name: str | None = None,
    rebuild: bool = False,
) -> str:
    """Build a Tensorlake sandbox image from a Dockerfile and return its registered name.

    The image is cached by name. The default name comes from a hash of the
    Dockerfile and the files in the context directory, so a second call with
    the same inputs does not rebuild.

    This call blocks while the image builds. In async code, run it with
    ``asyncio.to_thread``.

    Args:
        dockerfile_path: Path to the Dockerfile.
        context_dir: Build context directory. Defaults to the Dockerfile's directory.
        name: Registered image name. Defaults to ``tinker-<hash>``.
        rebuild: Build even if an image with this name exists.
    """
    from tensorlake.image.sandbox_builder import build_sandbox_image, find_sandbox_image_by_name

    dockerfile = Path(dockerfile_path).resolve()
    context = Path(context_dir).resolve() if context_dir is not None else dockerfile.parent
    if name is None:
        h = hashlib.sha256(dockerfile.read_bytes())
        # Skip the temporary Dockerfiles of other builds (see below).
        files = (
            p for p in context.rglob("*") if p.is_file() and not _TMP_DOCKERFILE_RE.match(p.name)
        )
        for f in sorted(files):
            h.update(str(f.relative_to(context)).encode() + b"\0" + f.read_bytes())
        name = f"tinker-{h.hexdigest()[:16]}"

    if not rebuild and find_sandbox_image_by_name(name) is not None:
        return name

    # Tensorlake uses the Dockerfile's directory as the build context, so put
    # a copy of the Dockerfile in the context directory when they differ.
    if context == dockerfile.parent:
        build_sandbox_image(str(dockerfile), registered_name=name)
    else:
        tmp_dockerfile = context / f".tinker-{uuid.uuid4().hex[:8]}.Dockerfile"
        shutil.copyfile(dockerfile, tmp_dockerfile)
        try:
            build_sandbox_image(str(tmp_dockerfile), registered_name=name)
        finally:
            tmp_dockerfile.unlink(missing_ok=True)
    return name


def registry_image_name(image_ref: str) -> str:
    """Registered Tensorlake image name for a container registry reference.

    The name is a sanitized copy of the reference plus a short hash of the
    exact reference, for example ``org/app:1.0`` becomes
    ``org-app-1-0-<hash>``. Harbor derives the same name, so images that
    Harbor imported or published (including the public Terminal-Bench
    images) resolve here without a rebuild.
    """
    ref = image_ref.strip()
    if not ref:
        raise ValueError("image_ref must be a non-empty string")
    sanitized = re.sub(r"[^a-z0-9]+", "-", ref.lower()).strip("-")
    digest = hashlib.sha256(ref.encode()).hexdigest()[:8]
    return f"{sanitized}-{digest}" if sanitized else digest


def resolve_registry_image(image_ref: str, import_if_missing: bool = True) -> str:
    """Return the registered Tensorlake image for a container registry reference.

    Looks the image up under :func:`registry_image_name`. The lookup covers
    images in your project and public images, so a Harbor-published image
    is found without a build. On a miss, imports the image from the registry
    and registers it under that name, unless ``import_if_missing`` is False,
    in which case a miss raises ``LookupError``.

    This call blocks while the image imports. In async code, run it with
    ``asyncio.to_thread``.
    """
    from tensorlake.image.sandbox_builder import find_sandbox_image_by_name, import_sandbox_image

    name = registry_image_name(image_ref)
    if find_sandbox_image_by_name(name) is not None:
        return name
    if not import_if_missing:
        raise LookupError(f"No registered Tensorlake image {name!r} for {image_ref!r}")
    logger.info("Importing %s into Tensorlake as %s", image_ref, name)
    import_sandbox_image(image_ref, registered_name=name)
    return name
