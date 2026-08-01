"""Unit tests for Fystash sandbox backend (no live network)."""

from __future__ import annotations

import asyncio
import os
from unittest import mock

import pytest

from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    default_sandbox_factory,
    resolve_sandbox_factory,
)
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.sandbox.fystash_sandbox import FystashSandbox, fystash_sandbox_factory


def test_sandbox_backend_enum_includes_fystash() -> None:
    assert SandboxBackend.FYSTASH == "fystash"


def test_resolve_default_is_modal() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TINKER_SANDBOX_BACKEND", None)
        factory = resolve_sandbox_factory()
    assert factory is default_sandbox_factory


def test_resolve_fystash_backend_string() -> None:
    factory = resolve_sandbox_factory(sandbox_backend="fystash")
    assert factory is fystash_sandbox_factory


def test_resolve_fystash_from_env() -> None:
    with mock.patch.dict(os.environ, {"TINKER_SANDBOX_BACKEND": "fystash"}):
        factory = resolve_sandbox_factory()
    assert factory is fystash_sandbox_factory


def test_resolve_explicit_factory_wins() -> None:
    sentinel = default_sandbox_factory
    factory = resolve_sandbox_factory(
        sandbox_backend="fystash",
        sandbox_factory=sentinel,
    )
    assert factory is sentinel


def test_resolve_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown sandbox_backend"):
        resolve_sandbox_factory(sandbox_backend="daytona")


def test_fystash_create_requires_api_key() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FYSTASH_API_KEY", None)

        async def _run() -> None:
            await FystashSandbox.create(timeout=60)

        with pytest.raises(RuntimeError, match="FYSTASH_API_KEY"):
            asyncio.run(_run())


def test_pool_start_requires_api_key() -> None:
    from tinker_cookbook.sandbox.fystash_pool import FystashSandboxPool

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FYSTASH_API_KEY", None)
        with pytest.raises(RuntimeError, match="FYSTASH_API_KEY"):
            FystashSandboxPool(pool_size=2)


def test_from_existing_does_not_own_api() -> None:
    from tinker_cookbook.sandbox.fystash_sandbox import _RoomApi

    api = _RoomApi("https://api.fystash.ai", "dummy")
    sb = FystashSandbox.from_existing(
        api, room_id="r1", agent_id="tinker", template_id="default"
    )
    assert sb._owns_api is False
    assert sb.sandbox_id == "r1"
