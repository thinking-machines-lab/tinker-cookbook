"""Tests for picklability of RL EnvGroupBuilders and rollout executor infrastructure."""

import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pytest

from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.rollouts import (
    _RolloutTask,
    get_rollout_executor,
    set_rollout_executor,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


class TestProblemGroupBuilderPickle:
    def test_pickle_roundtrip(self) -> None:
        """ProblemGroupBuilder with a Renderer-bound env_thunk survives pickle.

        Uses the real MathEnv class, matching how recipes actually construct builders.
        """
        math_env_mod = pytest.importorskip(
            "tinker_cookbook.recipes.math_rl.math_env",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        MathEnv = math_env_mod.MathEnv

        tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
        renderer = get_renderer("llama3", tokenizer)

        builder = ProblemGroupBuilder(
            env_thunk=partial(MathEnv, "What is 2+2?", "4", renderer),
            num_envs=4,
            dataset_name="test_math",
        )

        restored = pickle.loads(pickle.dumps(builder))

        assert restored.num_envs == 4
        assert restored.dataset_name == "test_math"
        # Verify the renderer inside the partial survived
        assert restored.env_thunk.args[2]._renderer_name == "llama3"

    def test_pickle_with_convo_prefix(self) -> None:
        """ProblemGroupBuilder with convo_prefix in the partial survives pickle."""
        math_env_mod = pytest.importorskip(
            "tinker_cookbook.recipes.math_rl.math_env",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        MathEnv = math_env_mod.MathEnv

        tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
        renderer = get_renderer("llama3", tokenizer)
        convo_prefix: list[Message] = [{"role": "system", "content": "You are helpful."}]

        builder = ProblemGroupBuilder(
            env_thunk=partial(MathEnv, "What is 2+2?", "4", renderer, convo_prefix=convo_prefix),
            num_envs=2,
        )

        restored = pickle.loads(pickle.dumps(builder))
        assert restored.env_thunk.keywords["convo_prefix"] == convo_prefix


class TestRolloutTask:
    def test_pickle_roundtrip(self) -> None:
        """_RolloutTask survives pickle roundtrip with a real Renderer-bound builder."""
        math_env_mod = pytest.importorskip(
            "tinker_cookbook.recipes.math_rl.math_env",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        MathEnv = math_env_mod.MathEnv

        tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
        renderer = get_renderer("llama3", tokenizer)

        builder = ProblemGroupBuilder(
            env_thunk=partial(MathEnv, "What is 2+2?", "4", renderer),
            num_envs=2,
        )

        # SamplingClient can't be constructed without a server, so test with None
        # to verify the dataclass + builder pickle. Full integration requires a server.
        task = _RolloutTask(
            sampling_client=None,  # type: ignore[arg-type]
            env_group_builder=builder,
            max_tokens=256,
            temperature=1.0,
            remove_constant_reward_groups=False,
            enable_logging=False,
        )

        restored = pickle.loads(pickle.dumps(task))
        assert restored.max_tokens == 256
        assert restored.temperature == 1.0
        assert restored.remove_constant_reward_groups is False
        assert restored.env_group_builder.num_envs == 2
        assert restored.env_group_builder.env_thunk.args[2]._renderer_name == "llama3"


class TestRolloutExecutorContextVar:
    def test_default_is_none(self) -> None:
        """Default rollout executor is None (in-process async)."""
        assert get_rollout_executor() is None

    def test_set_and_get(self) -> None:
        """set_rollout_executor / get_rollout_executor roundtrip."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            set_rollout_executor(executor)
            assert get_rollout_executor() is executor
        finally:
            set_rollout_executor(None)
            executor.shutdown(wait=False)
        assert get_rollout_executor() is None
