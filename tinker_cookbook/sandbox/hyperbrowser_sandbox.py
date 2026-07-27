"""
Thin wrapper around the Hyperbrowser Sandbox API.

Hyperbrowser provides cloud-based sandboxed execution environments.
Requires an API key: export HYPERBROWSER_API_KEY=...

Unlike Modal, Hyperbrowser has no server-side image builder: custom images are
built with a local ``docker buildx`` and uploaded as a root filesystem. To keep
that off the hot path, :class:`HyperbrowserImage` names images by the hash of
their contents and reuses any matching image already uploaded to the account, so
a given Dockerfile is built at most once ever rather than once per rollout. Specs
that only layer packages onto a base image skip Docker entirely by running their
layers as startup commands.

Configuration via environment variables:
    HYPERBROWSER_API_KEY: API key (required)
    HYPERBROWSER_POOL_SIZE: Number of sandboxes in the pool (default: 32)
    HYPERBROWSER_CREATION_RATE_LIMIT: Max sandboxes created per second (default: 4)
    HYPERBROWSER_REGION: Sandbox region, e.g. "us-east" (default: account default)
    HYPERBROWSER_SANDBOX_IMAGE: Default base image (default: "python")
    HYPERBROWSER_RUN_AS: User to run commands as (default: "root")
    HYPERBROWSER_IMAGE_CACHE_DIR: Where built-image markers are recorded
        (default: ~/.cache/tinker-cookbook/hyperbrowser)

See: https://hyperbrowser.ai/docs/sandboxes/introduction
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import tempfile
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TypeVar, cast

try:
    from hyperbrowser import AsyncHyperbrowser
    from hyperbrowser.client.managers.async_manager.sandbox import SandboxHandle
    from hyperbrowser.exceptions import HyperbrowserError
    from hyperbrowser.models import (
        CreateSandboxParams,
        SandboxExecParams,
        SandboxImageListParams,
        SandboxProcessResult,
        SandboxRegion,
    )
except ImportError:
    raise ImportError(
        "hyperbrowser is required for HyperbrowserSandbox. "
        "Install it with: uv pip install 'tinker-cookbook[hyperbrowser]'"
    ) from None

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_BASE_IMAGE = "python"
"""Hyperbrowser base image used when a spec doesn't name one.

Available base images: python, node, node-chromium, claude-code, codex, openclaw.
All are Ubuntu 24.04 with a passwordless-sudo `ubuntu` user.
"""

IMAGE_PLATFORM = "linux/amd64"
_IMAGE_NAME_PLATFORM_SUFFIX = "linux-amd64"
_IMAGE_NAME_PREFIX = "tinker"
_MAX_IMAGE_NAME_LEN = 64
_DIGEST_LEN = 12
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # Chunk larger file writes.
_DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024
_RUNTIME_ATTEMPTS = 3
_RUNTIME_BACKOFF_SECS = 1.0
_IMAGE_LIST_PAGE_SIZE = 100

# Exit codes synthesized when Hyperbrowser reports a terminal status without one.
_EXIT_CODE_TIMEOUT = 124  # Matches coreutils `timeout`.
_EXIT_CODE_KILLED = 137  # 128 + SIGKILL.
_EXIT_CODE_FAILED = -1  # Matches ModalSandbox's failure convention.


# ---------------------------------------------------------------------------
# Shared client
# ---------------------------------------------------------------------------

_client: AsyncHyperbrowser | None = None
_client_loop: asyncio.AbstractEventLoop | None = None
_client_guard = threading.Lock()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


async def get_client() -> AsyncHyperbrowser:
    """Return the shared Hyperbrowser client, creating it on first use.

    One client (and therefore one HTTP connection pool) is shared by every
    sandbox, so a 32-sandbox pool doesn't open 32 connection pools.

    The client is keyed to the event loop that created it: the underlying httpx
    pool binds to a loop, so a cached client would raise "Event loop is closed"
    for anything that calls ``asyncio.run`` more than once in a process.
    """
    global _client, _client_loop
    loop = _running_loop()
    with _client_guard:
        if _client is not None and _client_loop is loop and not (loop and loop.is_closed()):
            return _client

        api_key = os.getenv("HYPERBROWSER_API_KEY")
        if not api_key:
            raise SandboxError(
                "HYPERBROWSER_API_KEY is not set. Create a key at "
                "https://app.hyperbrowser.ai and export it before using "
                "HyperbrowserSandbox."
            )
        # Any previous client belongs to a loop we can no longer await on; drop it.
        _client = AsyncHyperbrowser(api_key=api_key)
        _client_loop = loop
        return _client


async def close_client() -> None:
    """Close the shared client. Safe to call when no client was created."""
    global _client, _client_loop
    with _client_guard:
        client, _client, _client_loop = _client, None, None
    if client is not None:
        with contextlib.suppress(Exception):
            await client.close()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _is_sandbox_terminated(e: BaseException) -> bool:
    """Check if an exception indicates the sandbox is gone.

    Deliberately does not treat a bare 404 as termination: the runtime returns
    404 for a missing *file* too, and callers expect that to be an ordinary
    nonzero-exit result rather than a dead sandbox.
    """
    if isinstance(e, SandboxTerminatedError):
        return True
    if isinstance(e, HyperbrowserError):
        return e.code == "sandbox_not_running" or e.status_code in (409, 410)
    msg = str(e).lower()
    return any(keyword in msg for keyword in ("sandbox_not_running", "not running", "terminated"))


def _is_retryable(e: BaseException) -> bool:
    """Check if an exception is a transient runtime failure worth retrying."""
    if _is_sandbox_terminated(e):
        return False
    if isinstance(e, HyperbrowserError):
        if e.retryable:
            return True
        return e.status_code is not None and e.status_code >= 500
    return isinstance(e, (asyncio.TimeoutError, ConnectionError, OSError))


def _raise_if_terminated(e: BaseException) -> None:
    """Re-raise as SandboxTerminatedError when the sandbox is gone.

    Callers wrap this around broad ``except Exception`` blocks, so an already
    converted error must propagate rather than be re-wrapped or swallowed.
    """
    if isinstance(e, SandboxTerminatedError):
        raise e
    if _is_sandbox_terminated(e):
        raise SandboxTerminatedError(str(e)) from e


# ---------------------------------------------------------------------------
# Image specs
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Reduce a name to characters that are safe in a Hyperbrowser image name."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-_.")
    return slug or "image"


def _hash_context_dir(context_dir: Path) -> str:
    """Hash every file in a build context, so the digest tracks its contents.

    Skips ``.git`` and honors a top-level ``.dockerignore``'s simple patterns.
    """
    ignored = _dockerignore_patterns(context_dir)
    hasher = hashlib.sha256()
    for path in sorted(p for p in context_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(context_dir).as_posix()
        if rel.startswith(".git/") or _is_dockerignored(rel, ignored):
            continue
        hasher.update(rel.encode())
        hasher.update(f":{path.stat().st_mode & 0o777:o}:".encode())
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
    return hasher.hexdigest()


def _dockerignore_patterns(context_dir: Path) -> list[str]:
    dockerignore = context_dir / ".dockerignore"
    if not dockerignore.is_file():
        return []
    lines = dockerignore.read_text(errors="replace").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _parse_dockerfile_workdir(dockerfile: Path) -> str | None:
    """Return the effective WORKDIR of a Dockerfile, or None if it sets none.

    Successive relative WORKDIRs stack, as Docker specifies. Build-arg and env
    interpolation is not resolved — a WORKDIR containing ``$`` is treated as
    unknown rather than guessed at.
    """
    try:
        text = dockerfile.read_text(errors="replace")
    except OSError:
        return None

    # Join line continuations so `WORKDIR \\\n /app` is seen as one instruction.
    workdir: PurePosixPath | None = None
    for line in text.replace("\\\n", " ").splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("workdir "):
            continue
        value = stripped[len("workdir ") :].strip().strip("\"'")
        if not value or "$" in value:
            return None
        workdir = (
            PurePosixPath(value)
            if value.startswith("/")
            else (workdir or PurePosixPath("/")) / value
        )
    return str(workdir) if workdir is not None else None


def _is_dockerignored(rel_path: str, patterns: list[str]) -> bool:
    return any(
        Path(rel_path).match(pattern) or rel_path.startswith(pattern.rstrip("/") + "/")
        for pattern in patterns
        if not pattern.startswith("!")
    )


@dataclass(frozen=True)
class HyperbrowserImage:
    """A Hyperbrowser sandbox image spec, mirroring the ``modal.Image`` surface.

    Specs are declarative and content-addressed; nothing is built until
    :meth:`resolve` runs. Three kinds:

    - ``base``: a Hyperbrowser base image (``python``, ``node``, ...).
    - ``registry``: a Docker registry reference (needs a local Docker to import).
    - ``dockerfile``: a local Dockerfile plus build context (needs local Docker).
    - ``prebuilt``: an image already uploaded to the account; never built.

    ``base`` and ``registry`` specs may carry layers (``apt_install``,
    ``pip_install``, ``run_commands``). Layers on a ``base`` spec are applied as
    sandbox startup commands when Docker isn't available, so the common case
    needs no local Docker at all.

    Usage:
        image = HyperbrowserImage.base("python").pip_install("numpy")
        sandbox = await HyperbrowserSandbox.create(image=image)
    """

    kind: str = "base"
    reference: str = DEFAULT_BASE_IMAGE
    """Base image name, registry reference, or uploaded image name."""
    image_id: str | None = None
    """Specific uploaded image revision. Only meaningful for ``prebuilt``."""
    dockerfile: str | None = None
    """Dockerfile name, relative to the context dir. Only for ``dockerfile``."""
    context_dir: str | None = None
    """Build context directory. Only for ``dockerfile``."""
    layers: tuple[str, ...] = ()
    """Shell commands layered on top of ``reference``."""

    # -- constructors --------------------------------------------------------

    @classmethod
    def base(cls, name: str = DEFAULT_BASE_IMAGE) -> HyperbrowserImage:
        """A Hyperbrowser base image. Analogous to ``modal.Image.debian_slim()``."""
        return cls(kind="base", reference=name)

    @classmethod
    def from_registry(cls, reference: str) -> HyperbrowserImage:
        """A Docker registry image. Analogous to ``modal.Image.from_registry()``.

        Requires a local Docker daemon the first time the image is used, since
        Hyperbrowser imports images by uploading an exported root filesystem.
        """
        return cls(kind="registry", reference=reference)

    @classmethod
    def from_dockerfile(
        cls, path: str | Path, context_dir: str | Path | None = None
    ) -> HyperbrowserImage:
        """Build from a local Dockerfile. Analogous to ``modal.Image.from_dockerfile()``.

        Requires a local Docker daemon the first time the resulting image is
        used; afterwards every run reuses the uploaded image.
        """
        dockerfile_path = Path(path)
        context = Path(context_dir) if context_dir is not None else dockerfile_path.parent
        return cls(
            kind="dockerfile",
            reference=context.name,
            dockerfile=dockerfile_path.name,
            context_dir=str(context.resolve()),
        )

    @classmethod
    def from_hyperbrowser(cls, image_name: str, image_id: str | None = None) -> HyperbrowserImage:
        """Launch an image already uploaded to the Hyperbrowser account.

        The escape hatch for machines without Docker: prebuild elsewhere, then
        reference the image by name here.
        """
        return cls(kind="prebuilt", reference=image_name, image_id=image_id)

    # -- layers --------------------------------------------------------------

    def apt_install(self, *packages: str) -> HyperbrowserImage:
        if not packages:
            return self
        joined = " ".join(shlex.quote(p) for p in packages)
        return self._add_layer(
            "apt-get update && DEBIAN_FRONTEND=noninteractive "
            f"apt-get install -y --no-install-recommends {joined}"
        )

    def pip_install(self, *packages: str) -> HyperbrowserImage:
        if not packages:
            return self
        joined = " ".join(shlex.quote(p) for p in packages)
        # `python3 -m pip` rather than bare `pip`: the base images ship pip as a
        # module but not on PATH. --break-system-packages is required by PEP 668
        # on the Ubuntu 24.04 base images.
        return self._add_layer(
            f"python3 -m pip install --no-cache-dir --break-system-packages {joined}"
        )

    def run_commands(self, *commands: str) -> HyperbrowserImage:
        image = self
        for command in commands:
            image = image._add_layer(command)
        return image

    def _add_layer(self, command: str) -> HyperbrowserImage:
        if self.kind == "prebuilt":
            raise ValueError(
                "Cannot layer commands onto a prebuilt Hyperbrowser image. "
                "Rebuild the image with the extra layers, or start from "
                "HyperbrowserImage.base()/from_registry()."
            )
        if self.kind == "dockerfile":
            raise ValueError(
                "Cannot layer commands onto a Dockerfile image. Add the steps to "
                "the Dockerfile instead so they are captured by its digest."
            )
        return replace(self, layers=(*self.layers, command))

    # -- identity ------------------------------------------------------------

    def digest(self) -> str:
        """Content hash of this spec. Identical specs produce identical digests."""
        spec: dict[str, object] = {
            "kind": self.kind,
            "reference": self.reference,
            "layers": list(self.layers),
            "platform": IMAGE_PLATFORM,
        }
        if self.kind == "dockerfile":
            assert self.context_dir is not None and self.dockerfile is not None
            context = Path(self.context_dir)
            spec["dockerfile"] = self.dockerfile
            spec["dockerfile_sha256"] = hashlib.sha256(
                (context / self.dockerfile).read_bytes()
            ).hexdigest()
            spec["context_sha256"] = _hash_context_dir(context)
        canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def image_name(self) -> str:
        """The image name this spec launches, at most 64 characters.

        For specs that need building, the name embeds a content digest, so two
        machines building the same spec agree on the name and whichever uploads
        first satisfies everyone else. Specs that launch an existing image
        (``prebuilt``, ``base``) return that image's name unchanged.
        """
        if not self.needs_build():
            return self.reference
        digest = self.digest()[:_DIGEST_LEN]
        fixed_len = len(f"{_IMAGE_NAME_PREFIX}____{digest}__{_IMAGE_NAME_PLATFORM_SUFFIX}")
        slug = _slugify(self.reference)[: _MAX_IMAGE_NAME_LEN - fixed_len].strip("-_.") or "image"
        return f"{_IMAGE_NAME_PREFIX}__{slug}__{digest}__{_IMAGE_NAME_PLATFORM_SUFFIX}"

    def needs_build(self) -> bool:
        """Whether launching this spec requires a local Docker build and upload.

        ``base`` specs never do — their layers run at sandbox startup instead.
        """
        return self.kind not in ("prebuilt", "base")

    def startup_commands(self) -> tuple[str, ...]:
        """Layers to run after sandbox creation instead of baking them in.

        Only used for ``base`` specs (and as a Docker-less fallback for
        ``registry`` specs), where re-running cheap package installs per sandbox
        is preferable to requiring a local Docker daemon.
        """
        return self.layers

    def default_workdir(self) -> str | None:
        """The image's ``WORKDIR``, or None if it can't be determined.

        Hyperbrowser's image format carries only env/command/args, so a built
        image loses its ``WORKDIR`` and every exec would otherwise start in the
        runtime's own default directory. :class:`SandboxInterface` specifies that
        ``workdir=None`` runs in the image's WORKDIR, so recover it from the
        Dockerfile. Registry and base images have no local Dockerfile to read,
        and keep the runtime default.
        """
        if self.kind != "dockerfile":
            return None
        assert self.context_dir is not None and self.dockerfile is not None
        return _parse_dockerfile_workdir(Path(self.context_dir) / self.dockerfile)

    # -- resolution ----------------------------------------------------------

    async def resolve(
        self, client: AsyncHyperbrowser, *, build_timeout: float | None = None
    ) -> ResolvedImage:
        """Resolve this spec to a launchable Hyperbrowser image.

        Short-circuits, in order: prebuilt names, layer-free base images, the
        local build marker, the account's uploaded images, and only then a build.
        """
        if self.kind == "prebuilt":
            return ResolvedImage(self.reference, self.image_id, ())
        if self.kind == "base" and not self.layers:
            return ResolvedImage(self.reference, None, ())
        if self.kind == "base":
            # Cheap layers on a base image: run them at startup rather than
            # requiring Docker to bake an image for them.
            return ResolvedImage(self.reference, None, self.layers)

        image_name = self.image_name()
        workdir = self.default_workdir()
        if _marker_path(image_name).is_file():
            return ResolvedImage(image_name, None, (), workdir)

        async with _build_lock(image_name):
            # Double-checked: another task in this process may have built it
            # while we waited, and another machine may have uploaded it.
            if _marker_path(image_name).is_file():
                return ResolvedImage(image_name, None, (), workdir)
            if await _image_uploaded(client, image_name):
                logger.info("Reusing uploaded Hyperbrowser image %s", image_name)
                _write_marker(image_name, self.digest())
                return ResolvedImage(image_name, None, (), workdir)
            return await self._build(client, image_name, build_timeout=build_timeout)

    async def _build(
        self, client: AsyncHyperbrowser, image_name: str, *, build_timeout: float | None
    ) -> ResolvedImage:
        if not _docker_available():
            if self.kind == "registry" and self.layers:
                logger.warning(
                    "Docker is unavailable; running %s layers as startup commands "
                    "instead of baking an image.",
                    len(self.layers),
                )
                return ResolvedImage(self.reference, None, self.layers)
            raise SandboxError(_missing_docker_message(self, image_name))

        logger.info("Building Hyperbrowser image %s (this happens once per spec)", image_name)
        if self.kind == "dockerfile":
            assert self.context_dir is not None and self.dockerfile is not None
            await client.sandboxes.build_image_from_dockerfile(
                context_path=self.context_dir,
                dockerfile=self.dockerfile,
                image_name=image_name,
                platform=IMAGE_PLATFORM,
                wait=True,
                wait_timeout=build_timeout,
                upload_timeout=build_timeout,
            )
        else:
            # A registry reference plus layers: express it as a tiny Dockerfile so
            # the same local-build-and-upload path handles both kinds.
            with tempfile.TemporaryDirectory(prefix="tinker-hyperbrowser-") as context:
                Path(context, "Dockerfile").write_text(self._synthesized_dockerfile())
                await client.sandboxes.build_image_from_dockerfile(
                    context_path=context,
                    dockerfile="Dockerfile",
                    image_name=image_name,
                    platform=IMAGE_PLATFORM,
                    wait=True,
                    wait_timeout=build_timeout,
                    upload_timeout=build_timeout,
                )
        _write_marker(image_name, self.digest())
        return ResolvedImage(image_name, None, (), self.default_workdir())

    def _synthesized_dockerfile(self) -> str:
        lines = [f"FROM {self.reference}"]
        lines += [f"RUN {command}" for command in self.layers]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ResolvedImage:
    """A launchable image plus any layers deferred to sandbox startup."""

    image_name: str
    image_id: str | None
    startup_commands: tuple[str, ...]
    default_workdir: str | None = None
    """The image's WORKDIR, recovered from its Dockerfile. See
    :meth:`HyperbrowserImage.default_workdir`."""


# ---------------------------------------------------------------------------
# Build coordination and caching
# ---------------------------------------------------------------------------

_build_locks: dict[str, asyncio.Lock] = {}
_build_locks_guard = asyncio.Lock()


@contextlib.asynccontextmanager
async def _build_lock(image_name: str):
    """Serialize builds of the same image within this process.

    An RL step creates ``group_size`` sandboxes at once; without this they would
    each shell out to ``docker buildx`` for the same image.
    """
    async with _build_locks_guard:
        lock = _build_locks.setdefault(image_name, asyncio.Lock())
    async with lock:
        yield


def _cache_dir() -> Path:
    override = os.getenv("HYPERBROWSER_IMAGE_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "tinker-cookbook" / "hyperbrowser"


def _marker_path(image_name: str) -> Path:
    return _cache_dir() / "images" / f"{image_name}.json"


def _write_marker(image_name: str, digest: str) -> None:
    """Record that an image is available, to skip the list_images round-trip."""
    path = _marker_path(image_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"image_name": image_name, "digest": digest})
    tmp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)


async def _image_uploaded(client: AsyncHyperbrowser, image_name: str) -> bool:
    """Check whether a fully-uploaded image with this exact name already exists."""
    response = await client.sandboxes.list_images(
        SandboxImageListParams(search=image_name, limit=_IMAGE_LIST_PAGE_SIZE)
    )
    return any(image.image_name == image_name and image.uploaded for image in response.images)


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _missing_docker_message(image: HyperbrowserImage, image_name: str) -> str:
    """Explain the failure and give the exact command that unblocks it."""
    if image.kind == "dockerfile":
        source = f"HyperbrowserImage.from_dockerfile({image.dockerfile!r}, {image.context_dir!r})"
    else:
        source = f"HyperbrowserImage.from_registry({image.reference!r})"
    prebuild = (
        'python -c "import asyncio\n'
        "from tinker_cookbook.sandbox.hyperbrowser_sandbox import (\n"
        "    HyperbrowserImage, get_client)\n"
        f"image = {source}\n"
        'asyncio.run(image.resolve(asyncio.run(get_client())))"'
    )
    return (
        f"Building Hyperbrowser image {image_name!r} needs a local Docker daemon, and "
        "`docker` was not found on PATH. Hyperbrowser imports images by uploading an "
        "exported root filesystem, so there is no server-side builder.\n"
        "Either:\n"
        "  1. Install Docker and re-run — the build happens once, and every later run "
        "(on any machine) reuses the uploaded image.\n"
        "  2. Prebuild on a Docker-capable machine, then pass "
        f"HyperbrowserImage.from_hyperbrowser({image_name!r}) here:\n"
        f"{prebuild}\n"
        "  3. Use HyperbrowserImage.base(...).apt_install(...)/.pip_install(...) instead, "
        "whose layers run at sandbox startup and need no Docker."
    )


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class HyperbrowserSandbox:
    """
    Persistent Hyperbrowser sandbox for code execution. Conforms to SandboxInterface.

    Usage:
        sandbox = await HyperbrowserSandbox.create()

        await sandbox.write_file("/workspace/code.py", "print('hello')")
        result = await sandbox.run_command("python /workspace/code.py")
        print(result.stdout)

        await sandbox.cleanup()
    """

    def __init__(
        self,
        sandbox: SandboxHandle,
        *,
        run_as: str | None,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        default_workdir: str | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._run_as = run_as
        self._max_output_bytes = max_output_bytes
        self._default_workdir = default_workdir
        self._files = sandbox.files.with_run_as(run_as)

    @classmethod
    async def create(
        cls,
        image: HyperbrowserImage | str | None = None,
        timeout: int = 600,
        *,
        region: str | None = None,
        cpu: int | None = None,
        memory_mib: int | None = None,
        disk_mib: int | None = None,
        allow_internet_access: bool | None = None,
        allow_out: list[str] | None = None,
        deny_out: list[str] | None = None,
        run_as: str | None = None,
        workdir: str | None = None,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        build_timeout: float | None = None,
    ) -> HyperbrowserSandbox:
        """Create a new Hyperbrowser sandbox.

        Args:
            image: Image spec, a base image name, or None for the default.
            timeout: Sandbox lifetime in seconds. Hyperbrowser's lifetime is
                minute-granular and fixed at creation — there is no extend API,
                so size this for the whole episode.
            region: Sandbox region (default: ``HYPERBROWSER_REGION`` or account default).
            cpu / memory_mib / disk_mib: Resource requests.
            allow_internet_access / allow_out / deny_out: Network policy.
            run_as: User to run commands and file operations as. Defaults to
                ``HYPERBROWSER_RUN_AS`` or ``root``; base images otherwise
                default to the unprivileged ``ubuntu`` user.
            workdir: Directory for commands that don't name one. Defaults to the
                image's WORKDIR when it can be recovered (see
                :meth:`HyperbrowserImage.default_workdir`).
            max_output_bytes: Default cap on captured stdout/stderr.
            build_timeout: Seconds to allow for a one-time image build.
        """
        if image is None:
            image = HyperbrowserImage.base(
                os.getenv("HYPERBROWSER_SANDBOX_IMAGE", DEFAULT_BASE_IMAGE)
            )
        elif isinstance(image, str):
            image = HyperbrowserImage.base(image)

        client = await get_client()
        resolved = await image.resolve(client, build_timeout=build_timeout)

        params = CreateSandboxParams(
            image_name=resolved.image_name,
            image_id=resolved.image_id,
            # Hyperbrowser lifetimes are in whole minutes; never round down to 0.
            timeout_minutes=max(1, math.ceil(timeout / 60)),
            region=cast(SandboxRegion | None, region or os.getenv("HYPERBROWSER_REGION") or None),
            cpu=cpu,
            memory_mib=memory_mib,
            disk_mib=disk_mib,
            allow_internet_access=allow_internet_access,
            allow_out=allow_out,
            deny_out=deny_out,
        )
        handle = await client.sandboxes.create(params)
        if run_as is None:
            run_as = os.getenv("HYPERBROWSER_RUN_AS", "root")
        sandbox = cls(
            handle,
            run_as=run_as,
            max_output_bytes=max_output_bytes,
            default_workdir=workdir or resolved.default_workdir,
        )

        for command in resolved.startup_commands:
            result = await sandbox.run_command(command, timeout=600)
            if result.exit_code != 0:
                await sandbox.cleanup()
                raise SandboxError(
                    f"Sandbox startup command failed ({result.exit_code}): "
                    f"{command}\n{result.stderr[:2000]}"
                )
        return sandbox

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.id

    async def send_heartbeat(self, timeout: int = 30) -> None:
        try:
            await asyncio.wait_for(self._exec("true", None, timeout), timeout=timeout)
        except Exception as e:
            _raise_if_terminated(e)
            raise

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox."""
        cap = max_output_bytes if max_output_bytes is not None else self._max_output_bytes
        try:
            result = await self._exec(command, workdir or self._default_workdir, timeout)
            return _to_sandbox_result(result, cap)
        except Exception as e:
            _raise_if_terminated(e)
            return SandboxResult(stdout="", stderr=str(e), exit_code=_EXIT_CODE_FAILED)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox.

        Content is returned in ``stdout``, matching ModalSandbox (which shells
        out to ``cat``) so callers can treat the backends interchangeably.
        """
        try:
            content = await self._with_retries(
                functools.partial(self._download, path, max_bytes, timeout)
            )
            return SandboxResult(
                stdout=content.decode("utf-8", errors="replace"), stderr="", exit_code=0
            )
        except Exception as e:
            if isinstance(e, HyperbrowserError) and e.status_code == 404:
                return SandboxResult(stdout="", stderr=f"{path}: No such file", exit_code=1)
            _raise_if_terminated(e)
            return SandboxResult(stdout="", stderr=str(e), exit_code=_EXIT_CODE_FAILED)

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Write content to a file in the sandbox, creating parent directories."""
        if isinstance(content, str):
            content = content.encode()

        try:
            parent = os.path.dirname(path)
            if parent not in ("", "/"):
                # `mkdir -p` semantics: an existing directory is not an error, and
                # a genuinely missing one surfaces when the upload below fails.
                with contextlib.suppress(HyperbrowserError):
                    await asyncio.wait_for(
                        self._files.make_dir(parent, parents=True), timeout=timeout
                    )
            await self._upload(path, content, timeout)
            if executable:
                await self._with_retries(functools.partial(self._chmod, path, "755", timeout))
            return SandboxResult(stdout="", stderr="", exit_code=0)
        except Exception as e:
            _raise_if_terminated(e)
            return SandboxResult(stdout="", stderr=str(e), exit_code=_EXIT_CODE_FAILED)

    async def cleanup(self) -> None:
        """Stop the Hyperbrowser sandbox. Idempotent."""
        try:
            await self._sandbox.stop()
        except HyperbrowserError as e:
            # Already stopped or expired — nothing left to clean up.
            if e.status_code not in (404, 409, 410) and not _is_sandbox_terminated(e):
                raise
        except SandboxTerminatedError:
            pass

    # -- internals -----------------------------------------------------------

    async def _exec(self, command: str, workdir: str | None, timeout: int) -> SandboxProcessResult:
        """Run ``command`` under bash, matching ModalSandbox's `bash -lc`."""
        params = SandboxExecParams(
            command="bash",
            args=["-lc", command],
            cwd=workdir,
            timeout_sec=timeout,
            run_as=self._run_as,
        )
        return await self._with_retries(lambda: self._sandbox.exec(params))

    async def _download(self, path: str, max_bytes: int | None, timeout: int) -> bytes:
        """Fetch file bytes; download() streams raw, read_bytes() base64-encodes."""
        if max_bytes is None:
            return await asyncio.wait_for(self._files.download(path), timeout=timeout)
        return await asyncio.wait_for(
            self._files.read_bytes(path, length=max_bytes), timeout=timeout
        )

    async def _chmod(self, path: str, mode: str, timeout: int) -> None:
        await asyncio.wait_for(self._files.chmod(path=path, mode=mode), timeout=timeout)

    async def _upload_chunk(self, path: str, chunk: bytes, append: bool, timeout: int) -> None:
        if append:
            await asyncio.wait_for(
                self._files.write_bytes(path, chunk, append=True), timeout=timeout
            )
        else:
            # upload() PUTs raw bytes; write_bytes() base64-encodes into JSON.
            await asyncio.wait_for(self._files.upload(path, chunk), timeout=timeout)

    async def _upload(self, path: str, content: bytes, timeout: int) -> None:
        """Upload file content, chunking payloads too large for a single request."""
        for offset in range(0, max(len(content), 1), _MAX_UPLOAD_BYTES):
            chunk = content[offset : offset + _MAX_UPLOAD_BYTES]
            await self._with_retries(
                functools.partial(self._upload_chunk, path, chunk, offset > 0, timeout)
            )

    async def _with_retries(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Retry transient runtime failures; surface terminations immediately."""
        for attempt in range(_RUNTIME_ATTEMPTS):
            try:
                return await operation()
            except Exception as e:
                _raise_if_terminated(e)
                if attempt == _RUNTIME_ATTEMPTS - 1 or not _is_retryable(e):
                    raise
                logger.debug(
                    "Retrying Hyperbrowser request after %s (attempt %d/%d)",
                    e,
                    attempt + 1,
                    _RUNTIME_ATTEMPTS,
                )
                await asyncio.sleep(_RUNTIME_BACKOFF_SECS * (attempt + 1))
        raise AssertionError("unreachable")


def _to_sandbox_result(result: SandboxProcessResult, cap: int) -> SandboxResult:
    """Convert a Hyperbrowser process result into a SandboxResult.

    Hyperbrowser reports a status alongside an optional exit code; synthesize
    conventional codes for terminal statuses that have none, so callers can rely
    on ``exit_code`` the way they do with Modal.
    """
    stderr = result.stderr
    if result.error:
        stderr = f"{stderr}\n{result.error}" if stderr else result.error

    exit_code = result.exit_code
    if exit_code is None:
        if result.status == "timed_out":
            exit_code = _EXIT_CODE_TIMEOUT
        elif result.status == "killed":
            exit_code = _EXIT_CODE_KILLED
        elif result.status == "exited":
            exit_code = 0
        else:
            exit_code = _EXIT_CODE_FAILED

    return SandboxResult(
        stdout=result.stdout[:cap],
        stderr=stderr[:cap],
        exit_code=exit_code,
    )


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class HyperbrowserSandboxPool:
    """
    Pool of Hyperbrowser sandboxes for concurrent execution.

    Each sandbox handles one request at a time. The pool manages
    borrowing and returning sandboxes automatically.

    Configuration via environment variables:
        HYPERBROWSER_POOL_SIZE: Number of sandboxes in the pool (default: 32)
        HYPERBROWSER_CREATION_RATE_LIMIT: Max sandboxes created per second (default: 4)
    """

    def __init__(
        self,
        *,
        pool_size: int | None = None,  # Number of warm sandboxes to maintain during the job run.
        sandbox_timeout_secs: int = 1200,  # Time after which a sandbox is terminated.
        image: HyperbrowserImage | None = None,
    ):
        self._pool_size = pool_size or int(os.getenv("HYPERBROWSER_POOL_SIZE", "32"))
        self._creation_rate_limit = int(os.getenv("HYPERBROWSER_CREATION_RATE_LIMIT", "4"))
        self._sandbox_timeout_secs = sandbox_timeout_secs
        self._image = image
        self._terminated = False

        self._warm_pool: asyncio.Queue[HyperbrowserSandbox] = asyncio.Queue()  # Warm sandboxes.
        self._to_terminate: list[HyperbrowserSandbox] = []  # Sandboxes pending termination.
        self._active_count = 0  # Number of in-use sandboxes.
        self._maintenance_stopped = asyncio.Event()

        self._maintenance = asyncio.create_task(self._maintain_pool())

    async def _create(self) -> HyperbrowserSandbox:
        return await HyperbrowserSandbox.create(
            image=self._image, timeout=self._sandbox_timeout_secs
        )

    async def _maintain_pool(self) -> None:
        """Background task to handle all sandbox creation and termination."""
        try:
            while not self._terminated:
                try:
                    await self._maintain_pool_step()
                except Exception as e:
                    logger.error(f"Error maintaining HyperbrowserSandboxPool: {e}")
                await asyncio.sleep(1.0)
        finally:
            # Lets terminate() wait for any in-flight creation to land in the
            # warm pool, so it can't drain the queue and then be handed a
            # freshly created sandbox that nobody ever stops.
            self._maintenance_stopped.set()

    async def _maintain_pool_step(self) -> None:
        """Single iteration of pool maintenance: terminate used sandboxes, create new ones."""
        # Batch terminate used sandboxes
        if self._to_terminate:
            to_terminate, self._to_terminate = self._to_terminate, []
            await asyncio.gather(*(sb.cleanup() for sb in to_terminate), return_exceptions=True)

        # Create new sandboxes in parallel (respecting rate limit)
        total = self._warm_pool.qsize() + self._active_count
        need = min(self._creation_rate_limit, self._pool_size - total)
        if need > 0 and not self._terminated:
            new_sandboxes = await asyncio.gather(
                *(self._create() for _ in range(need)),
                return_exceptions=True,
            )
            for sb in new_sandboxes:
                if isinstance(sb, BaseException):
                    logger.error(f"Error creating Hyperbrowser sandbox: {sb}")
                else:
                    await self._warm_pool.put(sb)

    async def run_in_workdir(
        self,
        files: dict[str, str],
        command: list[str],
        timeout: int | None = None,
    ) -> SandboxResult:
        """
        Execute command with files using an available sandbox from the pool.
        If all sandboxes are busy, waits until one becomes available.

        Creates an isolated workdir, writes files, and runs the command.

        Args:
            files: Files to write {filename: content}
            command: Command and arguments (e.g., ["python", "run.py"])
            timeout: Execution timeout in seconds
        """
        if self._terminated:
            raise SandboxError("HyperbrowserSandboxPool has been terminated.")

        sandbox = await self._warm_pool.get()
        self._active_count += 1

        try:
            workdir = f"/workspace/{uuid.uuid4().hex[:12]}"
            result = await sandbox.run_command(
                f"mkdir -p {shlex.quote(workdir)}", timeout=timeout or 60
            )
            if result.exit_code != 0:
                return SandboxResult(
                    stdout="",
                    stderr=f"Failed to create workdir: {workdir}",
                    exit_code=result.exit_code,
                )

            if files:
                await asyncio.gather(
                    *(
                        sandbox.write_file(f"{workdir}/{filename}", content)
                        for filename, content in files.items()
                    )
                )
            return await sandbox.run_command(
                shlex.join(command), workdir=workdir, timeout=timeout or self._sandbox_timeout_secs
            )
        finally:
            self._active_count -= 1
            self._to_terminate.append(sandbox)

    async def terminate(self) -> None:
        """Exit the pool and terminate all sandboxes."""
        self._terminated = True

        # Wait for active sandboxes to finish and be added to _to_terminate
        while self._active_count > 0:
            await asyncio.sleep(0.5)

        # Let the maintenance loop finish first. Draining the queue while it is
        # mid-creation would leak those sandboxes: they'd be enqueued after the
        # drain and left running until their lifetime expired.
        await self._maintenance_stopped.wait()

        # Collect and terminate all sandboxes
        all_sandboxes = list(self._to_terminate)
        while not self._warm_pool.empty():
            try:
                all_sandboxes.append(self._warm_pool.get_nowait())
            except asyncio.QueueEmpty:
                break
        await asyncio.gather(*(sb.cleanup() for sb in all_sandboxes), return_exceptions=True)
