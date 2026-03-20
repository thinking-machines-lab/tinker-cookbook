"""Tests for EnvFromMessageEnv (tinker_cookbook/rl/message_env.py).

Verifies that EnvFromMessageEnv correctly bridges message-level environments
to the token-level Env interface, including:
- Threading: build_generation_prompt runs via asyncio.to_thread
- Parse success/failure handling
- Max trajectory token enforcement
- Stop condition propagation
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import tinker

from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_input(tokens: list[int]) -> tinker.ModelInput:
    return tinker.ModelInput.from_ints(tokens)


class StubMessageEnv(MessageEnv):
    """Minimal MessageEnv for testing."""

    def __init__(
        self,
        initial_messages: list[Message],
        step_result: MessageStepResult,
    ):
        self._initial_messages = initial_messages
        self._step_result = step_result
        self.step_calls: list[Message] = []

    async def initial_observation(self) -> list[Message]:
        return self._initial_messages

    async def step(self, message: Message) -> MessageStepResult:
        self.step_calls.append(message)
        return self._step_result


def _make_renderer(
    gen_prompt_tokens: list[int] | None = None,
    stop_sequences: list[str] | None = None,
    parse_message: Message | None = None,
    parse_success: bool = True,
) -> MagicMock:
    """Build a mock Renderer with the methods EnvFromMessageEnv calls."""
    renderer = MagicMock()

    prompt = _make_model_input(gen_prompt_tokens or [1, 2, 3])
    renderer.build_generation_prompt = MagicMock(return_value=prompt)
    renderer.get_stop_sequences = MagicMock(return_value=stop_sequences or ["<stop>"])
    renderer.parse_response = MagicMock(
        return_value=(
            parse_message or {"role": "assistant", "content": "hello"},
            parse_success,
        )
    )
    return renderer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitialObservation:
    def test_returns_rendered_prompt_and_stop_condition(self):
        """initial_observation should render messages and return base stop condition."""
        renderer = _make_renderer(gen_prompt_tokens=[10, 20, 30], stop_sequences=["<eos>"])
        initial_msgs: list[Message] = [{"role": "user", "content": "hi"}]
        msg_env = StubMessageEnv(
            initial_messages=initial_msgs,
            step_result=MessageStepResult(reward=0, episode_done=False, next_messages=[]),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        model_input, stop_cond = asyncio.run(env.initial_observation())

        assert model_input.to_ints() == [10, 20, 30]
        assert stop_cond == ["<eos>"]
        renderer.build_generation_prompt.assert_called_once_with(initial_msgs)

    def test_render_runs_in_thread(self):
        """build_generation_prompt should be dispatched via asyncio.to_thread."""
        renderer = _make_renderer()
        msg_env = StubMessageEnv(
            initial_messages=[{"role": "user", "content": "hi"}],
            step_result=MessageStepResult(reward=0, episode_done=False, next_messages=[]),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        with patch(
            "tinker_cookbook.rl.message_env.asyncio.to_thread", wraps=asyncio.to_thread
        ) as mock_to_thread:
            asyncio.run(env.initial_observation())
            mock_to_thread.assert_called_once()
            # First positional arg should be the renderer method
            assert mock_to_thread.call_args[0][0] is renderer.build_generation_prompt


class TestStepParseFailure:
    def test_parse_failure_returns_failed_reward(self):
        """When parse_response fails, step returns failed_parse_reward."""
        renderer = _make_renderer(parse_success=False)
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(reward=1.0, episode_done=False, next_messages=[]),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            failed_parse_reward=-2.0,
            terminate_on_parse_error=True,
        )

        result = asyncio.run(env.step([1, 2, 3]))

        assert result.reward == -2.0
        assert result.episode_done is True
        assert result.metrics == {"parse_error": 1.0}
        assert result.next_observation.length == 0
        # MessageEnv.step should NOT have been called
        assert len(msg_env.step_calls) == 0

    def test_parse_failure_no_terminate(self):
        """When terminate_on_parse_error=False, episode continues after parse failure."""
        renderer = _make_renderer(parse_success=False)
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(reward=1.0, episode_done=False, next_messages=[]),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            failed_parse_reward=-1.0,
            terminate_on_parse_error=False,
        )

        result = asyncio.run(env.step([1, 2, 3]))

        assert result.episode_done is False
        assert result.reward == -1.0


class TestStepSuccess:
    def test_delegates_to_message_env_and_renders(self):
        """On successful parse, step delegates to MessageEnv and renders next messages."""
        assistant_msg: Message = {"role": "assistant", "content": "answer"}
        next_msgs: list[Message] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "followup"},
        ]
        renderer = _make_renderer(
            gen_prompt_tokens=[10, 20, 30, 40],
            stop_sequences=["<stop>"],
            parse_message=assistant_msg,
        )
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.75,
                episode_done=False,
                next_messages=next_msgs,
                metrics={"custom": 1.0},
            ),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        result = asyncio.run(env.step([5, 6, 7]))

        # Should have delegated parsed message to MessageEnv
        assert len(msg_env.step_calls) == 1
        assert msg_env.step_calls[0] == assistant_msg

        assert result.reward == 0.75
        assert result.episode_done is False
        assert result.next_observation.to_ints() == [10, 20, 30, 40]
        assert result.metrics == {"custom": 1.0}
        assert result.next_stop_condition == ["<stop>"]

    def test_custom_stop_condition_from_message_env(self):
        """When MessageEnv returns a next_stop_condition, it overrides the base one."""
        renderer = _make_renderer(stop_sequences=["<base_stop>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.5,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
                next_stop_condition=["<custom_stop>"],
            ),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        result = asyncio.run(env.step([1]))

        assert result.next_stop_condition == ["<custom_stop>"]

    def test_none_stop_condition_falls_back_to_base(self):
        """When MessageEnv returns None for next_stop_condition, base is used."""
        renderer = _make_renderer(stop_sequences=["<base>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.5,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
                next_stop_condition=None,
            ),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        result = asyncio.run(env.step([1]))

        assert result.next_stop_condition == ["<base>"]


class TestMaxTrajectoryTokens:
    def test_context_overflow_terminates_episode(self):
        """When next_observation exceeds max_trajectory_tokens, episode ends."""
        # Renderer returns a 100-token observation
        renderer = _make_renderer(gen_prompt_tokens=list(range(100)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.9,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
                metrics={"turns": 5.0},
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=50,  # limit is 50, observation is 100
        )

        result = asyncio.run(env.step([1]))

        assert result.episode_done is True
        assert result.reward == 0.0
        assert result.next_observation.length == 0  # empty observation
        assert result.metrics["context_overflow"] == 1.0
        # Original metrics should be preserved
        assert result.metrics["turns"] == 5.0

    def test_within_limit_continues(self):
        """When next_observation is within max_trajectory_tokens, episode continues."""
        renderer = _make_renderer(gen_prompt_tokens=[1, 2, 3], stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.5,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=1000,  # plenty of room
        )

        result = asyncio.run(env.step([1]))

        assert result.episode_done is False
        assert result.reward == 0.5
        assert "context_overflow" not in result.metrics

    def test_no_limit_set(self):
        """When max_trajectory_tokens is None, no overflow check occurs."""
        renderer = _make_renderer(gen_prompt_tokens=list(range(10000)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=1.0,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
            ),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        result = asyncio.run(env.step([1]))

        assert result.episode_done is False
        assert "context_overflow" not in result.metrics


class TestMaxGenerationTokens:
    def test_generation_budget_causes_overflow(self):
        """When observation + max_generation_tokens > max_trajectory_tokens, episode ends.

        The observation (80 tokens) fits under the trajectory limit (100) by itself,
        but adding the generation budget (30) pushes it over: 80 + 30 = 110 > 100.
        """
        renderer = _make_renderer(gen_prompt_tokens=list(range(80)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.9,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
                metrics={"turns": 2.0},
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=100,
            max_generation_tokens=30,
        )

        result = asyncio.run(env.step([1]))

        assert result.episode_done is True
        assert result.reward == 0.0
        assert result.next_observation.length == 0
        assert result.metrics["context_overflow"] == 1.0
        assert result.metrics["turns"] == 2.0

    def test_generation_budget_within_limit_continues(self):
        """When observation + max_generation_tokens <= max_trajectory_tokens, episode continues."""
        renderer = _make_renderer(gen_prompt_tokens=list(range(50)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.5,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=100,
            max_generation_tokens=30,  # 50 + 30 = 80 <= 100
        )

        result = asyncio.run(env.step([1]))

        assert result.episode_done is False
        assert result.reward == 0.5
        assert "context_overflow" not in result.metrics

    def test_initial_observation_raises_on_overflow(self):
        """initial_observation raises ValueError when prompt + generation budget exceeds limit."""
        renderer = _make_renderer(gen_prompt_tokens=list(range(80)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[{"role": "user", "content": "hi"}],
            step_result=MessageStepResult(reward=0, episode_done=False, next_messages=[]),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=100,
            max_generation_tokens=30,  # 80 + 30 = 110 > 100
        )

        with pytest.raises(ValueError, match="too long for the model's context window"):
            asyncio.run(env.initial_observation())

    def test_initial_observation_ok_when_within_limit(self):
        """initial_observation succeeds when prompt + generation budget fits."""
        renderer = _make_renderer(gen_prompt_tokens=list(range(50)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[{"role": "user", "content": "hi"}],
            step_result=MessageStepResult(reward=0, episode_done=False, next_messages=[]),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=100,
            max_generation_tokens=30,  # 50 + 30 = 80 <= 100
        )

        model_input, stop_cond = asyncio.run(env.initial_observation())

        assert model_input.to_ints() == list(range(50))
        assert stop_cond == ["<s>"]


class TestStepThreading:
    def test_step_renders_in_thread(self):
        """On successful parse, the next observation rendering should use to_thread."""
        renderer = _make_renderer()
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.5,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
            ),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        with patch(
            "tinker_cookbook.rl.message_env.asyncio.to_thread", wraps=asyncio.to_thread
        ) as mock_to_thread:
            asyncio.run(env.step([1, 2]))
            mock_to_thread.assert_called_once()
            assert mock_to_thread.call_args[0][0] is renderer.build_generation_prompt


class TestLogsPassthrough:
    """MessageStepResult.logs should be forwarded to StepResult.logs."""

    def test_logs_forwarded_on_success(self):
        """Logs from MessageEnv are passed through on normal step."""
        renderer = _make_renderer(gen_prompt_tokens=[1, 2, 3])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.5,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
                logs={"assistant": "hello world", "tool_call_0": "name=search"},
            ),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        result = asyncio.run(env.step([1]))

        assert result.logs == {"assistant": "hello world", "tool_call_0": "name=search"}

    def test_logs_forwarded_on_context_overflow(self):
        """Logs from MessageEnv are preserved even when context overflows."""
        renderer = _make_renderer(gen_prompt_tokens=list(range(100)))
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.5,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
                logs={"assistant": "some response", "tool_result_0": "result data"},
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=50,
        )

        result = asyncio.run(env.step([1]))

        assert result.episode_done is True
        assert result.metrics["context_overflow"] == 1.0
        assert result.logs == {"assistant": "some response", "tool_result_0": "result data"}

    def test_no_logs_on_parse_error(self):
        """Parse errors bypass MessageEnv, so logs are empty."""
        renderer = _make_renderer(parse_success=False)
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=1.0,
                episode_done=False,
                next_messages=[],
                logs={"should_not": "appear"},
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer, message_env=msg_env, terminate_on_parse_error=True
        )

        result = asyncio.run(env.step([1]))

        assert result.logs == {}

    def test_empty_logs_by_default(self):
        """When MessageEnv doesn't set logs, StepResult.logs defaults to empty."""
        renderer = _make_renderer(gen_prompt_tokens=[1, 2])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.5,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
            ),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        result = asyncio.run(env.step([1]))

        assert result.logs == {}
