"""Tests for picklability of VerifiersEnvGroupBuilder."""

import pytest

try:
    import verifiers as _verifiers  # noqa: F401

    _has_verifiers = True
except ImportError:
    _has_verifiers = False


@pytest.mark.skipif(not _has_verifiers, reason="verifiers not installed")
class TestVerifiersEnvGroupBuilderPickle:
    def test_pickle_excludes_vf_env(self) -> None:
        """VerifiersEnvGroupBuilder excludes vf_env from pickle state."""
        from unittest.mock import MagicMock

        from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersEnvGroupBuilder

        builder = VerifiersEnvGroupBuilder(
            vf_env=MagicMock(),
            prompt=[{"role": "user", "content": "What is 2+2?"}],
            example_id=42,
            task="arithmetic",
            answer="4",
        )
        state = builder.__getstate__()
        assert state["vf_env"] is None
        assert state["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
        assert state["example_id"] == 42
        assert state["task"] == "arithmetic"
        assert state["answer"] == "4"
