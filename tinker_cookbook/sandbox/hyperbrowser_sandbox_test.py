"""Unit tests for HyperbrowserImage and HyperbrowserSandbox.

Offline: the Hyperbrowser SDK is replaced with in-memory fakes, so these run in
CI without an API key, network access, or Docker.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from hyperbrowser import AsyncHyperbrowser
from hyperbrowser.client.managers.async_manager.sandbox import SandboxHandle
from hyperbrowser.exceptions import HyperbrowserError
from hyperbrowser.models import SandboxExecParams, SandboxProcessResult, SandboxProcessStatus

from tinker_cookbook.sandbox import SandboxBackend, resolve_backend
from tinker_cookbook.sandbox.hyperbrowser_sandbox import (
    HyperbrowserImage,
    HyperbrowserSandbox,
    HyperbrowserSandboxPool,
    ResolvedImage,
    _to_sandbox_result,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxInterface, SandboxTerminatedError


def _process_result(
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = 0,
    status: SandboxProcessStatus = "exited",
    error: str | None = None,
) -> SandboxProcessResult:
    return SandboxProcessResult(
        id="proc-1",
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        started_at=0,
        completed_at=1,
        error=error,
    )


class FakeFiles:
    """In-memory stand-in for SandboxFilesApi."""

    def __init__(self) -> None:
        self.contents: dict[str, bytes] = {}
        self.dirs: list[str] = []
        self.modes: dict[str, str] = {}
        self.run_as: str | None = None
        self.appends: list[tuple[str, bool]] = []

    def with_run_as(self, run_as: str | None) -> "FakeFiles":
        self.run_as = run_as
        return self

    async def download(self, path: str) -> bytes:
        if path not in self.contents:
            raise HyperbrowserError(f"{path} not present", status_code=404)
        return self.contents[path]

    async def read_bytes(self, path: str, *, length: int | None = None) -> bytes:
        return (await self.download(path))[:length]

    async def upload(self, path: str, data: bytes) -> None:
        self.contents[path] = bytes(data)
        self.appends.append((path, False))

    async def write_bytes(self, path: str, data: bytes, *, append: bool = False) -> None:
        self.contents[path] = self.contents.get(path, b"") + bytes(data) if append else bytes(data)
        self.appends.append((path, append))

    async def make_dir(self, path: str, *, parents: bool = False) -> None:
        self.dirs.append(path)

    async def chmod(self, *, path: str, mode: str) -> None:
        self.modes[path] = mode


class FakeSandboxHandle:
    """In-memory stand-in for the SDK's SandboxHandle."""

    def __init__(self, results: Sequence[SandboxProcessResult | Exception] | None = None) -> None:
        self.id = "sbx-123"
        self.files = FakeFiles()
        self.execs: list[SandboxExecParams] = []
        self.stopped = False
        self._results = list(results or [])

    async def exec(self, params: SandboxExecParams) -> SandboxProcessResult:
        self.execs.append(params)
        if not self._results:
            return _process_result()
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def stop(self) -> None:
        self.stopped = True


class FakeSandboxesApi:
    def __init__(self, uploaded: list[str] | None = None) -> None:
        self.uploaded = list(uploaded or [])
        self.builds: list[str] = []
        self.list_calls = 0

    async def list_images(self, params):
        self.list_calls += 1
        images = [
            SimpleNamespace(image_name=name, uploaded=True)
            for name in self.uploaded
            if params.search in name
        ]
        return SimpleNamespace(images=images)

    async def build_image_from_dockerfile(self, *, image_name: str, **kwargs):
        self.builds.append(image_name)
        self.uploaded.append(image_name)
        return SimpleNamespace(id="build-1", status="completed")


class FakeClient:
    def __init__(self, uploaded: list[str] | None = None) -> None:
        self.sandboxes = FakeSandboxesApi(uploaded)


def _sandbox(handle: FakeSandboxHandle, run_as: str = "root") -> HyperbrowserSandbox:
    """Build a sandbox over a fake handle (structurally compatible, not a subclass)."""
    return HyperbrowserSandbox(cast(SandboxHandle, handle), run_as=run_as)


def _resolve(image: HyperbrowserImage, client: FakeClient) -> ResolvedImage:
    return asyncio.run(image.resolve(cast(AsyncHyperbrowser, client)))


@pytest.fixture(autouse=True)
def isolated_image_cache(tmp_path, monkeypatch):
    """Keep build markers out of the developer's real cache directory."""
    monkeypatch.setenv("HYPERBROWSER_IMAGE_CACHE_DIR", str(tmp_path / "hb-cache"))


@pytest.fixture
def dockerfile_context(tmp_path: Path) -> Path:
    context = tmp_path / "environment"
    (context / "sub").mkdir(parents=True)
    (context / "Dockerfile").write_text("FROM python:3.12-slim\nRUN echo hi\n")
    (context / "sub" / "data.txt").write_text("payload")
    return context


# ---------------------------------------------------------------------------
# Image specs
# ---------------------------------------------------------------------------


def test_sandbox_conforms_to_interface():
    assert isinstance(_sandbox(FakeSandboxHandle()), SandboxInterface)


def test_base_image_launches_reference_without_building():
    image = HyperbrowserImage.base("python")
    assert image.image_name() == "python"
    assert not image.needs_build()
    assert image.startup_commands() == ()


def test_base_layers_become_startup_commands():
    image = HyperbrowserImage.base("python").apt_install("git").pip_install("numpy")
    assert not image.needs_build()
    commands = image.startup_commands()
    assert len(commands) == 2
    assert "apt-get install -y --no-install-recommends git" in commands[0]
    # Bare `pip` is not on PATH in the base images.
    assert commands[1].startswith("python3 -m pip install") and "numpy" in commands[1]


def test_digest_is_stable_and_layer_sensitive():
    a = HyperbrowserImage.from_registry("python:3.12-slim").pip_install("numpy")
    b = HyperbrowserImage.from_registry("python:3.12-slim").pip_install("numpy")
    c = HyperbrowserImage.from_registry("python:3.12-slim").pip_install("scipy")
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_image_name_fits_hyperbrowser_limit():
    image = HyperbrowserImage.from_registry(
        "registry.example.com/some/really/long/org/name/and-image:v1.2.3-rc4"
    )
    name = image.image_name()
    assert len(name) <= 64
    assert name.startswith("tinker__")
    assert name.endswith("__linux-amd64")


def test_dockerfile_digest_tracks_context_contents(dockerfile_context: Path):
    def digest() -> str:
        return HyperbrowserImage.from_dockerfile(dockerfile_context / "Dockerfile").digest()

    original = digest()

    (dockerfile_context / "sub" / "data.txt").write_text("changed")
    assert digest() != original, "context file changes must invalidate the image"

    (dockerfile_context / "sub" / "data.txt").write_text("payload")
    (dockerfile_context / "Dockerfile").write_text("FROM python:3.12-slim\nRUN echo bye\n")
    assert digest() != original, "Dockerfile changes must invalidate the image"


def test_dockerignored_files_do_not_affect_digest(dockerfile_context: Path):
    (dockerfile_context / ".dockerignore").write_text("sub/\n")
    original = HyperbrowserImage.from_dockerfile(dockerfile_context / "Dockerfile").digest()
    (dockerfile_context / "sub" / "data.txt").write_text("changed")
    assert HyperbrowserImage.from_dockerfile(dockerfile_context / "Dockerfile").digest() == original


def test_workdir_is_recovered_from_the_dockerfile(dockerfile_context: Path):
    """Hyperbrowser images drop WORKDIR, so we read it back off the Dockerfile."""
    dockerfile = dockerfile_context / "Dockerfile"

    dockerfile.write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    assert HyperbrowserImage.from_dockerfile(dockerfile).default_workdir() == "/app"

    # Relative WORKDIRs stack, as Docker specifies.
    dockerfile.write_text("FROM x\nWORKDIR /app\nWORKDIR src\n")
    assert HyperbrowserImage.from_dockerfile(dockerfile).default_workdir() == "/app/src"

    # Quoted and continued forms.
    dockerfile.write_text('FROM x\nWORKDIR \\\n  "/quoted/dir"\n')
    assert HyperbrowserImage.from_dockerfile(dockerfile).default_workdir() == "/quoted/dir"

    # Unresolvable or absent -> fall back to the runtime default.
    dockerfile.write_text("FROM x\nWORKDIR /app/$VERSION\n")
    assert HyperbrowserImage.from_dockerfile(dockerfile).default_workdir() is None
    dockerfile.write_text("FROM x\nRUN echo hi\n")
    assert HyperbrowserImage.from_dockerfile(dockerfile).default_workdir() is None


def test_only_dockerfile_specs_know_their_workdir():
    assert HyperbrowserImage.base("python").default_workdir() is None
    assert HyperbrowserImage.from_registry("python:3.12-slim").default_workdir() is None
    assert HyperbrowserImage.from_hyperbrowser("uploaded").default_workdir() is None


def test_default_workdir_is_used_when_no_workdir_is_given():
    handle = FakeSandboxHandle()
    sandbox = HyperbrowserSandbox(
        cast(SandboxHandle, handle), run_as="root", default_workdir="/app"
    )

    asyncio.run(sandbox.run_command("pwd"))
    assert handle.execs[-1].cwd == "/app"

    asyncio.run(sandbox.run_command("pwd", workdir="/other"))
    assert handle.execs[-1].cwd == "/other", "an explicit workdir must win"


def test_layers_rejected_on_unbuildable_kinds(dockerfile_context: Path):
    with pytest.raises(ValueError, match="prebuilt"):
        HyperbrowserImage.from_hyperbrowser("uploaded").pip_install("numpy")
    with pytest.raises(ValueError, match="Dockerfile"):
        HyperbrowserImage.from_dockerfile(dockerfile_context / "Dockerfile").pip_install("numpy")


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------


def test_resolve_prebuilt_makes_no_api_calls():
    client = FakeClient()
    resolved = _resolve(HyperbrowserImage.from_hyperbrowser("uploaded", "img-9"), client)
    assert (resolved.image_name, resolved.image_id) == ("uploaded", "img-9")
    assert client.sandboxes.list_calls == 0
    assert client.sandboxes.builds == []


def test_resolve_base_with_layers_defers_to_startup():
    client = FakeClient()
    image = HyperbrowserImage.base("python").pip_install("numpy")
    resolved = _resolve(image, client)
    assert resolved.image_name == "python"
    assert len(resolved.startup_commands) == 1
    assert client.sandboxes.builds == []


def test_resolve_reuses_uploaded_image(monkeypatch, dockerfile_context: Path):
    monkeypatch.setattr(
        "tinker_cookbook.sandbox.hyperbrowser_sandbox._docker_available", lambda: True
    )
    image = HyperbrowserImage.from_dockerfile(dockerfile_context / "Dockerfile")
    client = FakeClient(uploaded=[image.image_name()])

    resolved = _resolve(image, client)

    assert resolved.image_name == image.image_name()
    assert client.sandboxes.builds == [], "an already-uploaded image must not be rebuilt"


def test_resolve_builds_once_for_concurrent_callers(monkeypatch, dockerfile_context: Path):
    """group_size concurrent rollouts must not each shell out to docker buildx."""
    monkeypatch.setattr(
        "tinker_cookbook.sandbox.hyperbrowser_sandbox._docker_available", lambda: True
    )
    image = HyperbrowserImage.from_dockerfile(dockerfile_context / "Dockerfile")
    client = FakeClient()

    async def resolve_all() -> list[ResolvedImage]:
        typed_client = cast(AsyncHyperbrowser, client)
        return list(await asyncio.gather(*(image.resolve(typed_client) for _ in range(8))))

    resolved = asyncio.run(resolve_all())

    assert client.sandboxes.builds == [image.image_name()]
    assert {r.image_name for r in resolved} == {image.image_name()}


def test_resolve_without_docker_explains_the_alternatives(monkeypatch, dockerfile_context: Path):
    monkeypatch.setattr(
        "tinker_cookbook.sandbox.hyperbrowser_sandbox._docker_available", lambda: False
    )
    image = HyperbrowserImage.from_dockerfile(dockerfile_context / "Dockerfile")
    with pytest.raises(Exception, match="from_hyperbrowser"):
        _resolve(image, FakeClient())


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "exit_code", "expected"),
    [
        ("exited", 0, 0),
        ("exited", 3, 3),
        ("exited", None, 0),
        ("timed_out", None, 124),
        ("killed", None, 137),
        ("failed", None, -1),
    ],
)
def test_exit_code_synthesis(status: SandboxProcessStatus, exit_code: int | None, expected: int):
    result = _to_sandbox_result(_process_result(status=status, exit_code=exit_code), cap=1024)
    assert result.exit_code == expected


def test_output_is_capped_and_error_folded_into_stderr():
    result = _to_sandbox_result(
        _process_result(stdout="x" * 500, stderr="y" * 500, error="boom"), cap=10
    )
    assert result.stdout == "x" * 10
    assert len(result.stderr) == 10
    assert "boom" in _to_sandbox_result(_process_result(error="boom"), cap=1024).stderr


# ---------------------------------------------------------------------------
# Sandbox operations
# ---------------------------------------------------------------------------


def test_run_command_execs_under_bash_with_workdir_and_run_as():
    handle = FakeSandboxHandle([_process_result(stdout="hello\n")])
    sandbox = _sandbox(handle)

    result = asyncio.run(sandbox.run_command("echo hello", workdir="/tmp", timeout=30))

    assert result.stdout == "hello\n"
    assert result.exit_code == 0
    (params,) = handle.execs
    assert params.command == "bash"
    assert params.args == ["-lc", "echo hello"]
    assert params.cwd == "/tmp"
    assert params.timeout_sec == 30
    assert params.run_as == "root"


def test_run_command_respects_max_output_bytes():
    handle = FakeSandboxHandle([_process_result(stdout="z" * 1000)])
    sandbox = _sandbox(handle)
    result = asyncio.run(sandbox.run_command("cat big", max_output_bytes=16))
    assert result.stdout == "z" * 16


def test_read_file_returns_content_in_stdout():
    """ModalSandbox shells out to `cat`, so callers read file bytes from stdout."""
    handle = FakeSandboxHandle()
    handle.files.contents["/workspace/out.txt"] = b"file body"
    sandbox = _sandbox(handle)

    result = asyncio.run(sandbox.read_file("/workspace/out.txt"))
    assert (result.stdout, result.exit_code) == ("file body", 0)

    truncated = asyncio.run(sandbox.read_file("/workspace/out.txt", max_bytes=4))
    assert truncated.stdout == "file"


def test_read_file_missing_is_a_nonzero_exit_not_an_exception():
    sandbox = _sandbox(FakeSandboxHandle())
    result = asyncio.run(sandbox.read_file("/nope"))
    assert result.exit_code == 1
    assert "/nope" in result.stderr


def test_write_file_creates_parents_and_sets_mode():
    handle = FakeSandboxHandle()
    sandbox = _sandbox(handle)

    result = asyncio.run(sandbox.write_file("/tests/run.sh", "echo hi", executable=True))

    assert result.exit_code == 0
    assert handle.files.contents["/tests/run.sh"] == b"echo hi"
    assert handle.files.dirs == ["/tests"]
    assert handle.files.modes["/tests/run.sh"] == "755"


def test_write_file_at_root_skips_mkdir():
    handle = FakeSandboxHandle()
    sandbox = _sandbox(handle)
    asyncio.run(sandbox.write_file("/top.txt", "x"))
    assert handle.files.dirs == []


def test_write_file_chunks_large_payloads():
    handle = FakeSandboxHandle()
    sandbox = _sandbox(handle)
    payload = b"a" * (20 * 1024 * 1024)

    asyncio.run(sandbox.write_file("/big.bin", payload))

    assert handle.files.contents["/big.bin"] == payload
    appends = [append for _, append in handle.files.appends]
    assert appends == [False, True, True], "first chunk writes, the rest append"


def test_empty_write_still_creates_the_file():
    handle = FakeSandboxHandle()
    sandbox = _sandbox(handle)
    asyncio.run(sandbox.write_file("/empty.txt", ""))
    assert handle.files.contents["/empty.txt"] == b""


def test_files_api_is_scoped_to_run_as():
    handle = FakeSandboxHandle()
    _sandbox(handle, "ubuntu")
    assert handle.files.run_as == "ubuntu"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_dead_sandbox_raises_terminated_error():
    error = HyperbrowserError("gone", status_code=409, code="sandbox_not_running")
    sandbox = _sandbox(FakeSandboxHandle([error]), "root")
    with pytest.raises(SandboxTerminatedError):
        asyncio.run(sandbox.run_command("true"))


def test_heartbeat_raises_terminated_error():
    error = HyperbrowserError("gone", status_code=409, code="sandbox_not_running")
    sandbox = _sandbox(FakeSandboxHandle([error]), "root")
    with pytest.raises(SandboxTerminatedError):
        asyncio.run(sandbox.send_heartbeat())


def test_other_errors_become_failed_results_not_exceptions():
    error = HyperbrowserError("bad request", status_code=400)
    sandbox = _sandbox(FakeSandboxHandle([error]), "root")
    result = asyncio.run(sandbox.run_command("true"))
    assert result.exit_code == -1
    assert "bad request" in result.stderr


def test_transient_errors_are_retried():
    handle = FakeSandboxHandle(
        [
            HyperbrowserError("upstream hiccup", status_code=503, retryable=True),
            _process_result(stdout="recovered"),
        ]
    )
    sandbox = _sandbox(handle)

    result = asyncio.run(sandbox.run_command("true"))

    assert result.stdout == "recovered"
    assert len(handle.execs) == 2


def test_retries_are_bounded():
    errors = [HyperbrowserError("still down", status_code=503, retryable=True) for _ in range(5)]
    handle = FakeSandboxHandle(errors)
    sandbox = _sandbox(handle)

    result = asyncio.run(sandbox.run_command("true"))

    assert result.exit_code == -1
    assert len(handle.execs) == 3


def test_cleanup_tolerates_an_already_gone_sandbox():
    handle = FakeSandboxHandle()

    async def stop():
        raise HyperbrowserError("already stopped", status_code=404)

    handle.stop = stop
    asyncio.run(_sandbox(handle).cleanup())


def test_cleanup_stops_the_sandbox():
    handle = FakeSandboxHandle()
    asyncio.run(_sandbox(handle).cleanup())
    assert handle.stopped


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


def test_pool_terminate_leaves_no_running_sandboxes(monkeypatch):
    """terminate() must not race the maintenance loop into leaking sandboxes.

    Draining the warm queue while a creation batch is in flight would enqueue
    those sandboxes afterwards and leave them billing until they expired.
    """
    created: list[FakeSandboxHandle] = []

    async def fake_create(*args, **kwargs) -> HyperbrowserSandbox:
        await asyncio.sleep(0.3)
        handle = FakeSandboxHandle()
        created.append(handle)
        return _sandbox(handle)

    monkeypatch.setattr(HyperbrowserSandbox, "create", fake_create)

    async def run() -> None:
        pool = HyperbrowserSandboxPool(pool_size=4, sandbox_timeout_secs=60)
        # Terminate mid-creation: the first batch is 0.3s in flight, so calling
        # terminate at 0.05s is exactly the window that used to leak.
        await asyncio.sleep(0.05)
        await pool.terminate()
        # Give any creation that outlived terminate() time to land, so the
        # assertions below can see sandboxes the pool failed to account for.
        await asyncio.sleep(0.5)

    asyncio.run(run())

    assert created, "expected the pool to have created sandboxes"
    assert all(handle.stopped for handle in created), (
        f"{sum(not h.stopped for h in created)}/{len(created)} sandboxes leaked"
    )


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_resolve_backend_precedence(monkeypatch):
    monkeypatch.delenv("TINKER_SANDBOX_BACKEND", raising=False)
    assert resolve_backend() == SandboxBackend.MODAL

    monkeypatch.setenv("TINKER_SANDBOX_BACKEND", "hyperbrowser")
    assert resolve_backend() == SandboxBackend.HYPERBROWSER
    assert resolve_backend(SandboxBackend.MODAL) == SandboxBackend.MODAL

    monkeypatch.setenv("TINKER_SANDBOX_BACKEND", "not-a-backend")
    with pytest.raises(ValueError, match="TINKER_SANDBOX_BACKEND"):
        resolve_backend()
