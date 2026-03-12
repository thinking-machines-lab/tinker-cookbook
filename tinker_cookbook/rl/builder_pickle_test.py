"""Tests for picklability of RL EnvGroupBuilders."""

import pickle
from functools import partial

from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


class TestProblemGroupBuilderPickle:
    def test_pickle_roundtrip(self) -> None:
        """ProblemGroupBuilder with a Renderer-bound env_thunk survives pickle.

        Uses the real MathEnv class, matching how recipes actually construct builders.
        """
        from tinker_cookbook.recipes.math_rl.math_env import MathEnv
        from tinker_cookbook.rl.problem_env import ProblemGroupBuilder

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
        from tinker_cookbook.recipes.math_rl.math_env import MathEnv
        from tinker_cookbook.rl.problem_env import ProblemGroupBuilder

        tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
        renderer = get_renderer("llama3", tokenizer)
        convo_prefix: list[Message] = [{"role": "system", "content": "You are helpful."}]

        builder = ProblemGroupBuilder(
            env_thunk=partial(MathEnv, "What is 2+2?", "4", renderer, convo_prefix=convo_prefix),
            num_envs=2,
        )

        restored = pickle.loads(pickle.dumps(builder))
        assert restored.env_thunk.keywords["convo_prefix"] == convo_prefix
