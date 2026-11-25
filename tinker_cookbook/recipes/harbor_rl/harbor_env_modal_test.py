"""Unit tests for the Modal sandbox factory."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("modal")

import modal

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.recipes.harbor_rl.harbor_env import default_sandbox_factory
from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult


@pytest.fixture
def create(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace the image build and ModalSandbox.create with mocks."""
    monkeypatch.setattr(modal.Image, "from_dockerfile", MagicMock())
    mock = AsyncMock(return_value=object())
    monkeypatch.setattr(ModalSandbox, "create", mock)
    return mock


def test_default_sandbox_factory_creates_declared_workdir(
    tmp_path: Path, create: AsyncMock
) -> None:
    (tmp_path / "task.toml").write_text('[environment]\nworkdir = "/app"\n')
    sandbox = MagicMock()
    sandbox.run_command = AsyncMock(return_value=SandboxResult(stdout="", stderr="", exit_code=0))
    create.return_value = sandbox

    assert asyncio.run(default_sandbox_factory(tmp_path / "environment", timeout=900)) is sandbox

    sandbox.run_command.assert_awaited_once_with("mkdir -p /app")


def test_default_sandbox_factory_leaves_image_workdir_alone(
    tmp_path: Path, create: AsyncMock
) -> None:
    # The fixture's sandbox cannot run commands, so a mkdir would fail the test.
    asyncio.run(default_sandbox_factory(tmp_path / "environment", timeout=900))

    create.assert_awaited_once()


def test_default_sandbox_factory_cleans_up_when_workdir_creation_fails(
    tmp_path: Path, create: AsyncMock
) -> None:
    (tmp_path / "task.toml").write_text('[environment]\nworkdir = "/app"\n')
    sandbox = MagicMock()
    sandbox.run_command = AsyncMock(
        return_value=SandboxResult(stdout="", stderr="Permission denied", exit_code=1)
    )
    sandbox.cleanup = AsyncMock()
    create.return_value = sandbox

    with pytest.raises(SandboxError, match="/app: Permission denied"):
        asyncio.run(default_sandbox_factory(tmp_path / "environment", timeout=900))

    sandbox.cleanup.assert_awaited_once_with()
