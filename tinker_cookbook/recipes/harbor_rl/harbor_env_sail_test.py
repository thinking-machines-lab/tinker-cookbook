"""Unit tests for the Sail sandbox factory."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("sail")

import sail

from tinker_cookbook.recipes.harbor_rl.harbor_env import sail_sandbox_factory
from tinker_cookbook.sandbox.sail_sandbox import SailSandbox
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult


@pytest.fixture
def create(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace SailSandbox.create with a mock that records its arguments."""
    mock = AsyncMock(return_value=object())
    monkeypatch.setattr(SailSandbox, "create", mock)
    return mock


@pytest.mark.parametrize(
    "resources",
    [
        "cpus = 4\nmemory_mb = 65536\nstorage_mb = 307200\n",
        'cpus = 4\nmemory = "64G"\nstorage = "300G"\n',
    ],
    ids=["mb_fields", "deprecated_strings"],
)
def test_sail_sandbox_factory_sizes_to_task_resources(
    tmp_path: Path, create: AsyncMock, resources: str
) -> None:
    (tmp_path / "task.toml").write_text(
        f'[environment]\ndocker_image = "python:3.12-slim"\n{resources}'
    )
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    create.assert_awaited_once_with(
        image=sail.Image.from_registry("python:3.12-slim"),
        image_build_timeout=600,
        max_lifetime_seconds=900,
        size="m",
        memory_limit_gib=64,
        disk_limit_gib=300,
        egress_policy=None,
    )


def test_sail_sandbox_factory_forwards_build_timeout(tmp_path: Path, create: AsyncMock) -> None:
    task_toml = '[environment]\ndocker_image = "python:3.12-slim"\nbuild_timeout_sec = 3600\n'
    (tmp_path / "task.toml").write_text(task_toml)
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    assert create.call_args.kwargs["image_build_timeout"] == 3600


def test_sail_sandbox_factory_rejects_compose_tasks(tmp_path: Path, create: AsyncMock) -> None:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "docker-compose.yaml").write_text("services: {}\n")

    with pytest.raises(ValueError, match=r"docker-compose\.yaml"):
        asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    create.assert_not_awaited()


@pytest.mark.parametrize("os_value", ["windows", "Windows"])
def test_sail_sandbox_factory_rejects_windows_tasks(
    tmp_path: Path, create: AsyncMock, os_value: str
) -> None:
    task_toml = (
        '[environment]\ndocker_image = "mcr.microsoft.com/windows/servercore:ltsc2022"\n'
        f'os = "{os_value}"\n'
    )
    (tmp_path / "task.toml").write_text(task_toml)
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    with pytest.raises(ValueError, match="Windows"):
        asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    create.assert_not_awaited()


@pytest.mark.parametrize(
    "spec", ["gpus = 1\n", '[environment.tpu]\ntype = "v5e"\ntopology = "2x2"\n']
)
def test_sail_sandbox_factory_rejects_accelerator_tasks(
    tmp_path: Path, create: AsyncMock, spec: str
) -> None:
    task_toml = f'[environment]\ndocker_image = "python:3.12-slim"\n{spec}'
    (tmp_path / "task.toml").write_text(task_toml)
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    with pytest.raises(ValueError, match="accelerator"):
        asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    create.assert_not_awaited()


@pytest.mark.parametrize(
    ("spec", "egress_policy"),
    [
        ('network_mode = "no-network"\n', sail.EgressPolicy.no_network()),
        (
            'network_mode = "allowlist"\nallowed_hosts = ["pypi.org", "*.example.com"]\n',
            sail.EgressPolicy.allow_only("pypi.org", "*.example.com"),
        ),
        ('network_mode = "allowlist"\n', sail.EgressPolicy.no_egress()),
        ("allow_internet = false\n", sail.EgressPolicy.no_network()),
        ('allow_internet = false\nnetwork_mode = "public"\n', None),
        (
            'network_mode = "no-network"\n[agent]\nnetwork_mode = "no-network"\n',
            sail.EgressPolicy.no_network(),
        ),
        (
            'allow_internet = false\n[verifier]\nnetwork_mode = "no-network"\n',
            sail.EgressPolicy.no_network(),
        ),
        (
            'network_mode = "no-network"\n[verifier]\nenvironment_mode = "shared"\n',
            sail.EgressPolicy.no_network(),
        ),
    ],
)
def test_sail_sandbox_factory_applies_network_mode(
    tmp_path: Path, create: AsyncMock, spec: str, egress_policy: object
) -> None:
    task_toml = f'[environment]\ndocker_image = "python:3.12-slim"\n{spec}'
    (tmp_path / "task.toml").write_text(task_toml)
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    assert create.call_args.kwargs["egress_policy"] == egress_policy


@pytest.mark.parametrize(
    "spec",
    [
        '[agent]\nnetwork_mode = "no-network"\n',
        'network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n[verifier]\nnetwork_mode = "allowlist"\n',
    ],
)
def test_sail_sandbox_factory_rejects_phase_network_overrides(
    tmp_path: Path, create: AsyncMock, spec: str
) -> None:
    """Harbor rejects a differing phase override on a provider that cannot switch policies."""
    task_toml = f'[environment]\ndocker_image = "python:3.12-slim"\n{spec}'
    (tmp_path / "task.toml").write_text(task_toml)
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    with pytest.raises(ValueError, match="network policy"):
        asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    create.assert_not_awaited()


@pytest.mark.parametrize(
    "spec",
    [
        '[verifier]\nenvironment_mode = "separate"\n',
        '[verifier.environment]\ndocker_image = "python:3.12-slim"\nnetwork_mode = "no-network"\n',
    ],
)
def test_sail_sandbox_factory_rejects_separate_verifier_environments(
    tmp_path: Path, create: AsyncMock, spec: str
) -> None:
    """The cookbook grades in the agent's sandbox, so a verifier container of its own cannot run."""
    task_toml = f'[environment]\ndocker_image = "python:3.12-slim"\n{spec}'
    (tmp_path / "task.toml").write_text(task_toml)
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    with pytest.raises(ValueError, match="separate environment"):
        asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    create.assert_not_awaited()


def test_sail_sandbox_factory_rejects_healthchecks(tmp_path: Path, create: AsyncMock) -> None:
    """A Sailbox starts no service, so a readiness check has nothing to wait on."""
    (tmp_path / "task.toml").write_text(
        '[environment]\ndocker_image = "python:3.12-slim"\n'
        '[environment.healthcheck]\ncommand = "curl -sf http://localhost:8080"\n'
    )
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    with pytest.raises(ValueError, match="healthcheck"):
        asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    create.assert_not_awaited()


def test_sail_sandbox_factory_rejects_multi_step_tasks(tmp_path: Path, create: AsyncMock) -> None:
    """The cookbook runs one episode per task and has no step loop."""
    (tmp_path / "task.toml").write_text(
        '[environment]\ndocker_image = "python:3.12-slim"\n[[steps]]\nname = "first"\n'
    )
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    with pytest.raises(ValueError, match="multi-step"):
        asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    create.assert_not_awaited()


def test_sail_sandbox_factory_uploads_runtime_files_for_prebuilt_image(
    tmp_path: Path, create: AsyncMock
) -> None:
    (tmp_path / "task.toml").write_text(
        '[environment]\ndocker_image = "python:3.12-slim"\nworkdir = "/app"\n'
    )
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "data.txt").write_text("hello")
    sandbox = MagicMock()
    sandbox.sailbox.fs.mkdir.aio = AsyncMock()
    sandbox.sailbox.fs.upload_dir.aio = AsyncMock()
    create.return_value = sandbox

    assert asyncio.run(sail_sandbox_factory(env_dir, timeout=900)) is sandbox

    sandbox.sailbox.fs.mkdir.aio.assert_awaited_once_with("/app")
    sandbox.sailbox.fs.upload_dir.aio.assert_awaited_once_with(env_dir, "/app")


def test_sail_sandbox_factory_creates_declared_workdir(tmp_path: Path, create: AsyncMock) -> None:
    (tmp_path / "task.toml").write_text(
        '[environment]\ndocker_image = "python:3.12-slim"\nworkdir = "/app"\n'
    )
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    sandbox = MagicMock()
    sandbox.sailbox.fs.mkdir.aio = AsyncMock()
    create.return_value = sandbox

    asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    # The fixture's sandbox cannot run commands or upload, so either would fail the test.
    sandbox.sailbox.fs.mkdir.aio.assert_awaited_once_with("/app")


def test_sail_sandbox_factory_cleans_up_when_setup_fails(tmp_path: Path, create: AsyncMock) -> None:
    (tmp_path / "task.toml").write_text(
        '[environment]\ndocker_image = "python:3.12-slim"\nworkdir = "/app"\n'
    )
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    sandbox = MagicMock()
    sandbox.sailbox.fs.mkdir.aio = AsyncMock(side_effect=RuntimeError("mkdir failed"))
    sandbox.cleanup = AsyncMock()
    create.return_value = sandbox

    with pytest.raises(RuntimeError, match="mkdir failed"):
        asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    sandbox.cleanup.assert_awaited_once_with()


def test_sail_sandbox_factory_uploads_runtime_files_to_image_workdir(
    tmp_path: Path, create: AsyncMock
) -> None:
    (tmp_path / "task.toml").write_text('[environment]\ndocker_image = "python:3.12-slim"\n')
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "data.txt").write_text("hello")
    sandbox = MagicMock()
    sandbox.run_command = AsyncMock(
        return_value=SandboxResult(stdout="/root\n", stderr="", exit_code=0)
    )
    sandbox.sailbox.fs.upload_dir.aio = AsyncMock()
    create.return_value = sandbox

    asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    sandbox.run_command.assert_awaited_once_with("pwd")
    sandbox.sailbox.fs.upload_dir.aio.assert_awaited_once_with(env_dir, "/root")


def test_sail_sandbox_factory_keeps_build_context_out_of_prebuilt_image(
    tmp_path: Path, create: AsyncMock
) -> None:
    (tmp_path / "task.toml").write_text('[environment]\ndocker_image = "python:3.12-slim"\n')
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY data.txt /app/\n")
    (env_dir / "data.txt").write_text("hello")

    # The fixture's sandbox has no upload method, so an upload would fail the test.
    asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    create.assert_awaited_once_with(
        image=sail.Image.from_registry("python:3.12-slim"),
        image_build_timeout=600,
        max_lifetime_seconds=900,
        size="s",
        memory_limit_gib=None,
        disk_limit_gib=None,
        egress_policy=None,
    )


def test_sail_sandbox_factory_builds_dockerfile_without_task_toml(
    tmp_path: Path, create: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    image = sail.Image.from_registry("python:3.12-slim")
    from_dockerfile = AsyncMock(return_value=image)
    monkeypatch.setattr(sail.Image, "from_dockerfile", SimpleNamespace(aio=from_dockerfile))

    asyncio.run(sail_sandbox_factory(env_dir, timeout=900))

    from_dockerfile.assert_awaited_once_with(env_dir / "Dockerfile", context_dir=env_dir)
    create.assert_awaited_once_with(
        image=image,
        image_build_timeout=600,
        max_lifetime_seconds=900,
        size="s",
        memory_limit_gib=None,
        disk_limit_gib=None,
        egress_policy=None,
    )
