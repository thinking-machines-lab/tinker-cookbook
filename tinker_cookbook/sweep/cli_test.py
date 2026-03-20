"""Tests for sweep CLI utilities."""

from typing import Any
from unittest.mock import MagicMock

import chz
import pytest

from tinker_cookbook.sweep.cli import (
    _extract_recipe_from_argv,
    _get_batch_size,
    _make_sweep_config_cls,
    get_recipe,
)


class TestGetRecipe:
    def test_sft_alias_resolves(self):
        config_cls, main_fn = get_recipe("sft")
        assert config_cls is not None
        assert callable(main_fn)

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Could not import recipe"):
            get_recipe("nonexistent_recipe_xyz")

    def test_unknown_module_raises(self):
        with pytest.raises(ValueError, match="Could not import recipe"):
            get_recipe("tinker_cookbook.recipes.does_not_exist.train")

    def test_async_function_is_wrapped(self):
        """get_recipe wraps async cli_main into a sync function."""
        import types

        # Create a fake module with an async cli_main
        fake_module = types.ModuleType("fake_recipe")

        @chz.chz
        class FakeConfig:
            log_path: str | None = None
            learning_rate: float = 1e-4

        async def async_main(config: Any) -> None:
            pass

        fake_module.CLIConfig = FakeConfig  # type: ignore[attr-defined]
        fake_module.cli_main = async_main  # type: ignore[attr-defined]

        import sys

        sys.modules["fake_recipe"] = fake_module
        try:
            config_cls, main_fn = get_recipe("fake_recipe")
            assert config_cls is FakeConfig
            # The returned function should be synchronous (not a coroutine function)
            import inspect

            assert not inspect.iscoroutinefunction(main_fn)
        finally:
            del sys.modules["fake_recipe"]


class TestExtractRecipeFromArgv:
    def test_extracts_recipe(self):
        assert _extract_recipe_from_argv(["recipe=math_rl", "base.model=foo"]) == "math_rl"

    def test_defaults_to_sft(self):
        assert _extract_recipe_from_argv(["base.model=foo"]) == "sft"

    def test_empty_argv(self):
        assert _extract_recipe_from_argv([]) == "sft"

    def test_recipe_with_module_path(self):
        argv = ["recipe=tinker_cookbook.recipes.harbor_rl.train"]
        assert _extract_recipe_from_argv(argv) == "tinker_cookbook.recipes.harbor_rl.train"


class TestGetBatchSize:
    def test_batch_size_field(self):
        config = MagicMock(batch_size=64)
        assert _get_batch_size(config) == 64

    def test_groups_per_batch_fallback(self):
        config = MagicMock(spec=["groups_per_batch"])
        config.groups_per_batch = 32
        assert _get_batch_size(config) == 32

    def test_default(self):
        config = MagicMock(spec=[])
        assert _get_batch_size(config) == 128


class TestMakeSweepConfigCls:
    def test_creates_valid_chz_config(self):
        @chz.chz
        class FakeConfig:
            log_path: str | None = None
            learning_rate: float = 1e-4

        cls = _make_sweep_config_cls(FakeConfig)
        # Should be instantiable
        instance = cls()
        assert isinstance(instance.base, FakeConfig)
        assert isinstance(instance.learning_rates, list)
        assert isinstance(instance.lora_ranks, list)
        assert instance.max_parallel == 1
        assert instance.metric == "train_mean_nll"

    def test_default_learning_rates(self):
        @chz.chz
        class FakeConfig:
            log_path: str | None = None

        cls = _make_sweep_config_cls(FakeConfig)
        instance = cls()
        assert len(instance.learning_rates) == 6
        assert len(instance.lora_ranks) == 2
