"""Tests for picklability of VerifiersEnvGroupBuilder."""

import pickle

import pytest

try:
    import verifiers.v1 as _verifiers  # noqa: F401

    _has_verifiers = True
except ImportError:
    _has_verifiers = False


@pytest.mark.skipif(not _has_verifiers, reason="verifiers not installed")
class TestVerifiersEnvGroupBuilderPickle:
    def test_pickle_excludes_vf_env(self) -> None:
        """VerifiersEnvGroupBuilder excludes the live vf.Env from pickle state."""
        from unittest.mock import MagicMock

        import verifiers.v1 as vf

        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersEnvGroupBuilder

        task = vf.Task(vf.TaskData(idx=42, name="arithmetic", prompt="What is 2+2?"))
        builder = VerifiersEnvGroupBuilder(vf_env=MagicMock(), task=task)
        state = builder.__getstate__()
        assert state["vf_env"] is None
        assert state["task"].data.idx == 42
        assert state["task"].data.prompt == "What is 2+2?"

    def test_unpickle_recovers_env_from_context(self) -> None:
        from unittest.mock import MagicMock

        import verifiers.v1 as vf

        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
            VerifiersEnvGroupBuilder,
            set_vf_env,
        )

        task = vf.Task(vf.TaskData(idx=7, name="arithmetic"))
        builder = VerifiersEnvGroupBuilder(vf_env=MagicMock(), task=task)
        payload = pickle.dumps(builder)

        context_env = MagicMock()
        set_vf_env(context_env)
        try:
            restored = pickle.loads(payload)
        finally:
            set_vf_env(None)  # type: ignore[arg-type]
        assert restored.vf_env is context_env
        assert restored.task.data.idx == 7
        assert restored.logging_tags() == ["arithmetic"]
