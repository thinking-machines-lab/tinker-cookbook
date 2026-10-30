"""Tests that the on-policy distillation recipes forward ``temperature`` to the training config."""

import asyncio

import pytest

from tinker_cookbook.recipes.distillation import on_policy_distillation, on_policy_multi_teacher

RECIPES = [on_policy_distillation, on_policy_multi_teacher]


def _run_cli_main(recipe, monkeypatch, tmp_path, **cli_kwargs):
    captured = {}

    async def fake_resolve_renderer_name(**kwargs):
        return "qwen3"

    async def fake_train(config):
        captured["config"] = config

    monkeypatch.setattr(
        recipe.checkpoint_utils,
        "resolve_renderer_name_from_checkpoint_or_default_async",
        fake_resolve_renderer_name,
    )
    monkeypatch.setattr(recipe.cli_utils, "check_log_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr(recipe.train_on_policy, "main", fake_train)

    cli_config = recipe.CLIConfig(log_path=str(tmp_path), **cli_kwargs)
    asyncio.run(recipe.cli_main(cli_config))
    return captured["config"]


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda recipe: recipe.__name__.rsplit(".", 1)[-1])
@pytest.mark.parametrize("temperature", [0.0, 0.25])
def test_cli_main_forwards_temperature(recipe, temperature, monkeypatch, tmp_path):
    config = _run_cli_main(recipe, monkeypatch, tmp_path, temperature=temperature)
    assert config.temperature == temperature


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda recipe: recipe.__name__.rsplit(".", 1)[-1])
def test_cli_main_keeps_default_temperature(recipe, monkeypatch, tmp_path):
    config = _run_cli_main(recipe, monkeypatch, tmp_path)
    assert config.temperature == 1.0
