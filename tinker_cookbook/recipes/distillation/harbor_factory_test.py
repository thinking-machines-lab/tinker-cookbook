"""Recipe factory forwarding without loading tasks, models, or sandboxes."""

import asyncio
import pickle
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import chz
import pytest
import tinker

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.distillation.train_on_policy import Config as DistillationConfig
from tinker_cookbook.recipes.distillation import (
    on_policy_distillation_harbor_multi_turn as recipe,
)
from tinker_cookbook.recipes.distillation.harbor_multiturn import (
    HarborDistillationDatasetBuilder,
    zero_reward,
)
from tinker_cookbook.recipes.harbor_rl import harbor_env
from tinker_cookbook.sandbox import SandboxInterface

NON_CALLABLE_FACTORY = 17
TASKS = [harbor_env.HarborTask("synthetic", "complete the task", Path("synthetic-task"))]


async def custom_factory(env_dir: Path, timeout: int) -> SandboxInterface:
    raise AssertionError("Factory must not run while building configuration")


class _FalseyFactory:
    def __bool__(self) -> bool:
        return False

    async def __call__(self, env_dir: Path, timeout: int) -> SandboxInterface:
        raise AssertionError("Factory must not run while building configuration")


FALSEY_FACTORY = _FalseyFactory()


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("No task, model, service, or sandbox execution is allowed")


@dataclass
class _Boundary:
    train_main: AsyncMock
    renderer_resolver: AsyncMock
    log_dir_check: Mock
    task_loader: Mock


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch) -> _Boundary:
    mocked = _Boundary(
        train_main=AsyncMock(),
        renderer_resolver=AsyncMock(return_value="resolved-renderer"),
        log_dir_check=Mock(),
        task_loader=Mock(return_value=TASKS),
    )
    monkeypatch.setattr(recipe.train_on_policy, "main", mocked.train_main)
    monkeypatch.setattr(
        recipe.checkpoint_utils,
        "resolve_renderer_name_from_checkpoint_or_default_async",
        mocked.renderer_resolver,
    )
    monkeypatch.setattr(recipe.cli_utils, "check_log_dir", mocked.log_dir_check)
    monkeypatch.setattr(recipe, "load_harbor_tasks", mocked.task_loader)
    monkeypatch.setattr(harbor_env, "load_harbor_tasks", mocked.task_loader)
    monkeypatch.setattr(tinker, "ServiceClient", _forbidden)
    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", _forbidden)
    monkeypatch.setattr(harbor_env.ModalSandbox, "create", _forbidden)
    return mocked


def _captured_config(boundary: _Boundary) -> DistillationConfig:
    boundary.train_main.assert_awaited_once()
    call = boundary.train_main.await_args
    assert call is not None
    config = call.args[0]
    assert isinstance(config, DistillationConfig)
    return config


def _captured_builder(boundary: _Boundary) -> HarborDistillationDatasetBuilder:
    builder = _captured_config(boundary).dataset_configs[0].dataset_builder
    assert isinstance(builder, HarborDistillationDatasetBuilder)
    assert builder.reward_fn is zero_reward
    assert builder.tasks is TASKS
    return builder


def _run_entrypoint(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    assert recipe.__file__ is not None
    monkeypatch.setattr(sys, "argv", [recipe.__file__, "log_path=synthetic-output", *args])
    runpy.run_path(recipe.__file__, run_name="__main__")


def test_programmatic_factory_reaches_runtime_builder(boundary: _Boundary) -> None:
    config = recipe.CLIConfig(log_path="synthetic-output", sandbox_factory=custom_factory)
    asyncio.run(recipe.cli_main(config, TASKS))
    builder = _captured_builder(boundary)
    assert builder.sandbox_factory is custom_factory
    assert builder._make_env_group_builders(config.group_size)[0].sandbox_factory is custom_factory
    boundary.task_loader.assert_not_called()
    boundary.log_dir_check.assert_called_once_with("synthetic-output", behavior_if_exists="ask")


@pytest.mark.parametrize("from_cli", [False, True])
def test_falsey_factory_reaches_runtime_builder(
    boundary: _Boundary, monkeypatch: pytest.MonkeyPatch, from_cli: bool
) -> None:
    if from_cli:
        _run_entrypoint(monkeypatch, f"sandbox_factory={__name__}:FALSEY_FACTORY")
    else:
        config = recipe.CLIConfig(log_path="synthetic-output", sandbox_factory=FALSEY_FACTORY)
        asyncio.run(recipe.cli_main(config, TASKS))
    builder = _captured_builder(boundary)
    assert builder.sandbox_factory is FALSEY_FACTORY
    assert builder._make_env_group_builders(builder.group_size)[0].sandbox_factory is FALSEY_FACTORY


def test_existing_two_argument_call_keeps_default_backend(boundary: _Boundary) -> None:
    config = recipe.CLIConfig(log_path="synthetic-output")
    asyncio.run(recipe.cli_main(config, TASKS))
    builder = _captured_builder(boundary)
    runtime_builder = builder._make_env_group_builders(config.group_size)[0]
    assert runtime_builder.sandbox_factory is harbor_env.default_sandbox_factory


def test_real_entrypoint_accepts_an_annotated_async_factory(
    boundary: _Boundary, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_entrypoint(monkeypatch, f"sandbox_factory={__name__}:custom_factory")
    builder = _captured_builder(boundary)
    assert builder.sandbox_factory is custom_factory
    assert builder._make_env_group_builders(builder.group_size)[0].sandbox_factory is custom_factory
    boundary.task_loader.assert_called_once_with("terminal-bench-2.0")
    boundary.log_dir_check.assert_called_once_with("synthetic-output", behavior_if_exists="ask")


@pytest.mark.parametrize("explicit_none", [False, True])
def test_real_entrypoint_preserves_default_backend(
    boundary: _Boundary, monkeypatch: pytest.MonkeyPatch, explicit_none: bool
) -> None:
    _run_entrypoint(monkeypatch, *(["sandbox_factory=None"] if explicit_none else []))
    builder = _captured_builder(boundary)
    runtime_builder = builder._make_env_group_builders(builder.group_size)[0]
    assert runtime_builder.sandbox_factory is harbor_env.default_sandbox_factory
    boundary.task_loader.assert_called_once_with("terminal-bench-2.0")


def test_real_entrypoint_accepts_the_default_factory_by_name(
    boundary: _Boundary, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_entrypoint(monkeypatch, f"sandbox_factory={harbor_env.__name__}:default_sandbox_factory")
    builder = _captured_builder(boundary)
    assert builder.sandbox_factory is harbor_env.default_sandbox_factory


def test_callable_config_survives_chz_and_pickle(boundary: _Boundary) -> None:
    config = chz.entrypoint(
        recipe.CLIConfig,
        argv=["log_path=synthetic-output", f"sandbox_factory={__name__}:custom_factory"],
    )
    assert config.sandbox_factory is custom_factory
    assert chz.asdict(config)["sandbox_factory"] is custom_factory
    restored = pickle.loads(pickle.dumps(config))
    assert isinstance(restored, recipe.CLIConfig)
    assert restored.sandbox_factory is custom_factory
    asyncio.run(recipe.cli_main(restored, TASKS))
    assert _captured_builder(boundary).sandbox_factory is custom_factory


@pytest.mark.parametrize("renderer_name", [None, "explicit-renderer"])
def test_factory_does_not_change_renderer_or_training_config(
    boundary: _Boundary, renderer_name: str | None
) -> None:
    config = recipe.CLIConfig(
        log_path="synthetic-output",
        sandbox_factory=custom_factory,
        model_name="student-model",
        teacher_model="teacher-model",
        renderer_name=renderer_name,
        load_checkpoint_path="tinker://synthetic/state/checkpoint",
        teacher_checkpoint="tinker://teacher/state/checkpoint",
        base_url="https://example.invalid",
        group_size=2,
        groups_per_batch=3,
        max_turns=7,
        sandbox_timeout=300,
        command_timeout=40,
        max_trajectory_tokens=4096,
    )
    asyncio.run(recipe.cli_main(config, TASKS))
    boundary.renderer_resolver.assert_awaited_once_with(
        model_name="student-model",
        explicit_renderer_name=renderer_name,
        load_checkpoint_path=config.load_checkpoint_path,
        base_url=config.base_url,
    )
    forwarded = _captured_config(boundary)
    builder = _captured_builder(boundary)
    assert builder.renderer_name == forwarded.renderer_name == "resolved-renderer"
    assert builder.model_name == forwarded.model_name == "student-model"
    assert builder.group_size == 2
    assert builder.batch_size == 3
    assert builder.max_turns == 7
    assert builder.sandbox_timeout == 300
    assert builder.command_timeout == 40
    assert builder.max_trajectory_tokens == 4096
    assert forwarded.dataset_configs[0].teacher_config.base_model == "teacher-model"
    assert (
        forwarded.dataset_configs[0].teacher_config.load_checkpoint_path
        == config.teacher_checkpoint
    )
    assert forwarded.load_checkpoint_path == config.load_checkpoint_path
    assert forwarded.base_url == config.base_url


def test_cli_help_exposes_the_factory_override(boundary: _Boundary) -> None:
    help_text = chz.Blueprint(recipe.CLIConfig).get_help(color=False)
    assert "sandbox_factory" in help_text
    assert "module:callable" in help_text
    boundary.task_loader.assert_not_called()
    boundary.train_main.assert_not_awaited()


def test_noncallable_cli_factory_fails_before_task_or_training_work(
    boundary: _Boundary, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="callable"):
        _run_entrypoint(monkeypatch, f"sandbox_factory={__name__}:NON_CALLABLE_FACTORY")
    boundary.task_loader.assert_not_called()
    boundary.renderer_resolver.assert_not_awaited()
    boundary.log_dir_check.assert_not_called()
    boundary.train_main.assert_not_awaited()
