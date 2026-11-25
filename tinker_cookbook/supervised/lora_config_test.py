"""The supervised trainer forwards its LoRA target groups to the SDK for a fresh adapter."""

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder


class _Stop(Exception):
    pass


class _ServiceClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_ServiceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    async def create_lora_training_client(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        raise _Stop


def _create_call(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    _ServiceClient.calls = []
    config = train.Config(
        log_path=str(tmp_path),
        model_name="Qwen/Qwen3-8B",
        renderer_name="role_colon",
        recipe_name="lora_config_test",
        dataset_builder=cast(SupervisedDatasetBuilder, None),
        **overrides,
    )
    with patch.object(train.tinker, "ServiceClient", _ServiceClient):
        with pytest.raises(_Stop):
            asyncio.run(train.main(config))
    (call,) = _ServiceClient.calls
    return call


def test_defaults_keep_the_sdk_defaults(tmp_path: Path) -> None:
    call = _create_call(tmp_path)
    assert call["rank"] == 32
    assert (call["train_mlp"], call["train_attn"], call["train_unembed"]) == (True, True, True)


def test_target_groups_are_forwarded(tmp_path: Path) -> None:
    call = _create_call(
        tmp_path,
        lora_rank=8,
        train_mlp=False,
        train_attn=True,
        train_unembed=False,
    )
    assert call["rank"] == 8
    assert (call["train_mlp"], call["train_attn"], call["train_unembed"]) == (False, True, False)
