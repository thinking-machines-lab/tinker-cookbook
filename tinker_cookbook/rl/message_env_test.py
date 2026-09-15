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
from tinker.types import ImageChunk

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.renderers.base import Message, ParseTermination, RenderContext, RenderedMessage
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.rl import types
from tinker_cookbook.rl.data_processing import _flatten_chunks, _is_prefix, trajectory_to_data
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use.agent_tool_message_env import build_agent_tool_env
from tinker_cookbook.tool_use.tools import FunctionTool, simple_tool_result
from tinker_cookbook.tool_use.types import ToolResult

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
    termination: ParseTermination | None = None,
    parse_success: bool | None = None,
) -> MagicMock:
    """Build a mock Renderer with the methods EnvFromMessageEnv calls.

    Pass ``termination`` directly to drive any ParseTermination state
    (including ``EOS``, which can't be expressed via the bool shortcut).
    ``parse_success`` is a back-compat shortcut: True -> STOP_SEQUENCE,
    False -> MALFORMED.
    """
    if termination is None:
        if parse_success is None:
            parse_success = True
        termination = (
            ParseTermination.STOP_SEQUENCE if parse_success else ParseTermination.MALFORMED
        )

    renderer = MagicMock()

    prompt = _make_model_input(gen_prompt_tokens or [1, 2, 3])
    renderer.build_generation_prompt = MagicMock(return_value=prompt)
    renderer.get_stop_sequences = MagicMock(return_value=stop_sequences or ["<stop>"])
    renderer.parse_response = MagicMock(
        return_value=(
            parse_message or {"role": "assistant", "content": "hello"},
            termination,
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

        result = asyncio.run(env.initial_observation())

        assert isinstance(result, tuple)
        model_input, stop_cond = result
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
        assert result.metrics == {"parse_error": 1.0, "stop/parse_error": 1.0}
        assert result.next_observation.length == 0
        # MessageEnv.step should NOT have been called
        assert len(msg_env.step_calls) == 0

    def test_eos_termination_invokes_grader(self):
        """Regression test for issue #685.

        When parse_response returns ``ParseTermination.EOS`` (model emitted
        EOS instead of the renderer's stop sequence — the common base-model
        single-turn case), ``EnvFromMessageEnv.step`` must NOT short-circuit
        with ``failed_parse_reward``. It must call ``MessageEnv.step`` so the
        grader runs. Pre-#685, the renderer reported ``parse_success=False``
        on this shape and every base-model single-turn benchmark scored 0%.
        """

        renderer = _make_renderer(
            parse_message={"role": "assistant", "content": "graded answer"},
            termination=ParseTermination.EOS,
        )
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(reward=1.0, episode_done=True, next_messages=[]),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            failed_parse_reward=-99.0,  # would be the bug's reward if grader skipped
        )

        result = asyncio.run(env.step([1, 2, 3]))

        # Grader was invoked and its reward propagated — not the failed-parse reward.
        assert result.reward == 1.0
        assert "parse_error" not in result.metrics
        assert len(msg_env.step_calls) == 1
        assert msg_env.step_calls[0]["content"] == "graded answer"

    def test_malformed_termination_skips_grader(self):
        """``ParseTermination.MALFORMED`` short-circuits with failed_parse_reward."""

        renderer = _make_renderer(termination=ParseTermination.MALFORMED)
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(reward=1.0, episode_done=True, next_messages=[]),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env, failed_parse_reward=-2.0)

        result = asyncio.run(env.step([1, 2, 3]))

        assert result.reward == -2.0
        assert result.metrics == {"parse_error": 1.0, "stop/parse_error": 1.0}
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
        assert result.reward == -0.1  # default context_overflow_reward
        assert result.next_observation.length == 0  # empty observation
        assert result.metrics["context_overflow"] == 1.0
        # Original metrics should be preserved
        assert result.metrics["turns"] == 5.0

    def test_no_overflow_check_when_episode_done(self):
        """When episode is already done, context overflow check should NOT fire.

        The real reward from the env should be preserved even if the rendered
        conversation exceeds the limit — there is no next sampling call.
        """
        renderer = _make_renderer(gen_prompt_tokens=list(range(100)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.9,
                episode_done=True,  # episode is done (e.g., model gave final answer)
                next_messages=[{"role": "user", "content": "x"}],
                metrics={"accuracy": 1.0},
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=50,  # observation (100) exceeds limit (50)
            context_overflow_reward=-1.0,
        )

        result = asyncio.run(env.step([1]))

        # Real reward should be preserved, NOT replaced by context_overflow_reward
        assert result.reward == 0.9
        assert result.episode_done is True
        assert "context_overflow" not in result.metrics
        assert result.metrics["accuracy"] == 1.0

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
        assert result.reward == -0.1  # default context_overflow_reward
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

    def test_initial_observation_overflow_returns_graceful_stop(self):
        """initial_observation returns InitialObservationOverflow (instead of
        raising) when prompt + generation budget exceeds the limit, so the
        rollout ends gracefully with the flat context_overflow_reward."""
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

        result = asyncio.run(env.initial_observation())

        assert isinstance(result, types.InitialObservationOverflow)
        assert result.reward == -0.1  # default context_overflow_reward
        assert result.metrics["max_tokens_reached"] == 1.0
        assert result.metrics["stop/max_tokens"] == 1.0
        assert "too long for the model's context window" in str(
            result.logs["initial_observation_overflow"]
        )

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

        result = asyncio.run(env.initial_observation())

        assert isinstance(result, tuple)
        model_input, stop_cond = result
        assert model_input.to_ints() == list(range(50))
        assert stop_cond == ["<s>"]


class TestContextOverflowReward:
    def test_custom_overflow_reward(self):
        """context_overflow_reward is used when episode terminates due to overflow."""
        renderer = _make_renderer(gen_prompt_tokens=list(range(100)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.9,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=50,
            context_overflow_reward=-0.5,
        )

        result = asyncio.run(env.step([1]))

        assert result.episode_done is True
        assert result.reward == -0.5
        assert result.metrics["context_overflow"] == 1.0

    def test_custom_overflow_reward_with_generation_budget(self):
        """context_overflow_reward works with max_generation_tokens check."""
        renderer = _make_renderer(gen_prompt_tokens=list(range(80)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.9,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
                metrics={"turns": 3.0},
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=100,
            max_generation_tokens=30,  # 80 + 30 = 110 > 100
            context_overflow_reward=-1.0,
        )

        result = asyncio.run(env.step([1]))

        assert result.episode_done is True
        assert result.reward == -1.0
        assert result.metrics["context_overflow"] == 1.0
        assert result.metrics["turns"] == 3.0

    def test_default_overflow_reward(self):
        """Default context_overflow_reward is -0.1 (matches failed_parse_reward)."""
        renderer = _make_renderer(gen_prompt_tokens=list(range(100)), stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.9,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
            ),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            max_trajectory_tokens=50,
        )

        result = asyncio.run(env.step([1]))

        assert result.reward == -0.1


class TestMaxTokensReached:
    def test_stop_reason_length_terminates_episode(self):
        """When stop_reason='length', episode terminates with context_overflow_reward."""
        renderer = _make_renderer(gen_prompt_tokens=[1, 2, 3], stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.9,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
            ),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        result = asyncio.run(env.step([1, 2, 3], extra={"stop_reason": "length"}))

        assert result.episode_done is True
        assert result.reward == -0.1  # default context_overflow_reward
        assert result.next_observation.length == 0
        assert result.metrics["max_tokens_reached"] == 1.0
        # MessageEnv.step should NOT have been called (we short-circuit)
        assert len(msg_env.step_calls) == 0

    def test_stop_reason_length_uses_custom_overflow_reward(self):
        """stop_reason='length' uses the configured context_overflow_reward."""
        renderer = _make_renderer()
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(reward=0.9, episode_done=False, next_messages=[]),
        )
        env = EnvFromMessageEnv(
            renderer=renderer,
            message_env=msg_env,
            context_overflow_reward=-0.5,
        )

        result = asyncio.run(env.step([1], extra={"stop_reason": "length"}))

        assert result.reward == -0.5
        assert result.metrics["max_tokens_reached"] == 1.0

    def test_stop_reason_stop_continues_normally(self):
        """When stop_reason='stop' (default), normal processing occurs."""
        renderer = _make_renderer(gen_prompt_tokens=[10, 20], stop_sequences=["<s>"])
        msg_env = StubMessageEnv(
            initial_messages=[],
            step_result=MessageStepResult(
                reward=0.7,
                episode_done=False,
                next_messages=[{"role": "user", "content": "x"}],
            ),
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=msg_env)

        result = asyncio.run(env.step([1, 2]))

        assert result.reward == 0.7
        assert result.episode_done is False
        assert "max_tokens_reached" not in result.metrics
        assert len(msg_env.step_calls) == 1


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


# Token-preserving tool rollouts: real renderer through trajectory assembly.

ANALYSIS = "<|channel|>analysis<|message|>Search first.<|end|><|start|>assistant"
CANONICAL = (
    ' to=functions.search<|channel|>commentary <|constrain|>json<|message|>{"q": "x"}<|call|>'
)
REORDERED = (
    '<|channel|>commentary to=functions.search <|constrain|>json<|message|>{"q": "x"}<|call|>'
)


@pytest.fixture(scope="module")
def renderer() -> GptOssRenderer:
    return GptOssRenderer(get_tokenizer("openai/gpt-oss-20b"), use_system_prompt=False)


def search(q: str) -> ToolResult:
    """Search for a document."""
    return simple_tool_result(f"Found {q}")


async def reward(messages: list[Message]) -> tuple[float, dict[str, float]]:
    return 1.0, {}


def make_env(renderer: GptOssRenderer, preserve: bool = True) -> EnvFromMessageEnv:
    return build_agent_tool_env(
        renderer=renderer,
        tools=[FunctionTool(search)],
        initial_messages=[{"role": "user", "content": "Find it."}],
        reward_fn=reward,
        max_turns=10,
        preserve_sampled_tokens=preserve,
    )


@pytest.mark.parametrize(
    "response",
    [
        ANALYSIS + CANONICAL,
        ANALYSIS + REORDERED,
        CANONICAL,  # Tool-only parse fallback must not enter the token history.
        ANALYSIS + CANONICAL.replace('{"q": "x"}', ' {"q": "x"} '),
        ANALYSIS + CANONICAL.replace(" <|constrain|>json", ""),
    ],
)
def test_multiple_turns_pack_without_changing_actions(
    renderer: GptOssRenderer, response: str
) -> None:
    async def run() -> None:
        env = make_env(renderer)
        initial = await env.initial_observation()
        assert isinstance(initial, tuple)
        ob, _ = initial
        transitions = []
        action = renderer.tokenizer.encode(response, add_special_tokens=False)
        for _ in range(3):
            step = await env.step(action)
            assert not step.episode_done
            assert _is_prefix(ob.to_ints() + action, step.next_observation.to_ints())
            transitions.append(
                Transition(ob, TokensWithLogprobs(action, [-1.0] * len(action)), 0, False)
            )
            ob = step.next_observation
        final = renderer.tokenizer.encode(
            "<|channel|>final<|message|>Done.<|return|>", add_special_tokens=False
        )
        step = await env.step(final)
        assert step.episode_done
        assert _is_prefix(ob.to_ints() + final, step.next_observation.to_ints())
        assert (
            renderer.tokenizer.decode(step.next_observation.to_ints()[ob.length + len(final) :])
            == "<|start|>assistant"
        )
        transitions.append(Transition(ob, TokensWithLogprobs(final, [-1.0] * len(final)), 1, True))
        data = trajectory_to_data(Trajectory(transitions, step.next_observation), 1.0)
        assert len(data) == 1
        mask = data[0].loss_fn_inputs["mask"].to_torch().bool()
        targets = data[0].loss_fn_inputs["target_tokens"].to_torch()
        assert targets[mask].tolist() == action * 3 + final
        assert data[0].loss_fn_inputs["logprobs"].to_torch()[mask].tolist() == [-1.0] * int(
            mask.sum()
        )
        assert data[0].loss_fn_inputs["advantages"].to_torch()[mask].tolist() == [1.0] * int(
            mask.sum()
        )

    asyncio.run(run())


@pytest.mark.parametrize("preserve", [False, True])
def test_canonical_rendering_and_default_behavior(renderer: GptOssRenderer, preserve: bool) -> None:
    async def run() -> None:
        env = make_env(renderer, preserve)
        await env.initial_observation()
        action = renderer.tokenizer.encode(ANALYSIS + CANONICAL, add_special_tokens=False)
        step = await env.step(action)
        messages = await env.message_env.initial_observation()
        assert (
            step.next_observation.to_ints() == renderer.build_generation_prompt(messages).to_ints()
        )

    asyncio.run(run())


def test_disabled_flag_retains_retemplating(renderer: GptOssRenderer) -> None:
    async def run() -> None:
        env = make_env(renderer, False)
        initial = await env.initial_observation()
        assert isinstance(initial, tuple)
        action = renderer.tokenizer.encode(ANALYSIS + REORDERED, add_special_tokens=False)
        step = await env.step(action)
        assert not _is_prefix(initial[0].to_ints() + action, step.next_observation.to_ints())
        messages = await env.message_env.initial_observation()
        assert (
            step.next_observation.to_ints() == renderer.build_generation_prompt(messages).to_ints()
        )

    asyncio.run(run())


@pytest.mark.parametrize("fallback", ["injection", "truncation", "parse_retry"])
def test_fallback_resynchronizes_next_append(renderer: GptOssRenderer, fallback: str) -> None:
    async def run() -> None:
        env = make_env(renderer)
        await env.initial_observation()
        action = renderer.tokenizer.encode(ANALYSIS + REORDERED, add_special_tokens=False)
        await env.step(action)
        if fallback == "injection":
            ob, _ = await env.inject_messages([{"role": "user", "content": "Search again."}])
        elif fallback == "truncation":
            env.terminate_on_length = False
            truncated = renderer.tokenizer.encode(
                "<|channel|>analysis<|message|>Wait", add_special_tokens=False
            )
            step = await env.step(truncated, extra={"stop_reason": "length"})
            assert not step.episode_done
            ob = step.next_observation
        elif fallback == "parse_retry":
            env.set_parse_error_policy(ParseErrorPolicy(max_consecutive=2))
            invalid = renderer.tokenizer.encode(
                CANONICAL.replace('{"q": "x"}', "{bad"), add_special_tokens=False
            )
            step = await env.step(invalid)
            assert not step.episode_done
            ob = step.next_observation
        messages = await env.message_env.initial_observation()
        assert ob.to_ints() == renderer.build_generation_prompt(messages).to_ints()
        next_step = await env.step(action)
        assert _is_prefix(ob.to_ints() + action, next_step.next_observation.to_ints())

    asyncio.run(run())


class ReplacingEnv(MessageEnv):
    def __init__(self) -> None:
        self.history: list[Message] = [{"role": "user", "content": "Original question"}]
        self.replaced = False

    async def initial_observation(self) -> list[Message]:
        return self.history

    async def step(self, message: Message) -> MessageStepResult:
        if not self.replaced:
            self.replaced = True
            self.history = [{"role": "user", "content": "Replacement context"}]
            return MessageStepResult(0, False, self.history)
        self.history.append(message)
        return MessageStepResult(0, False, self.history, appended_messages=[])


def test_history_replacement_then_empty_append(renderer: GptOssRenderer) -> None:
    async def run() -> None:
        env = EnvFromMessageEnv(renderer, ReplacingEnv(), preserve_sampled_tokens=True)
        initial = await env.initial_observation()
        assert isinstance(initial, tuple)
        action = renderer.tokenizer.encode(ANALYSIS + REORDERED, add_special_tokens=False)
        replaced = await env.step(action)
        assert not _is_prefix(initial[0].to_ints() + action, replaced.next_observation.to_ints())
        appended = await env.step(action)
        assert _is_prefix(
            replaced.next_observation.to_ints() + action, appended.next_observation.to_ints()
        )

    asyncio.run(run())


def test_context_limit_uses_preserved_token_length(renderer: GptOssRenderer) -> None:
    async def run() -> None:
        env = make_env(renderer)
        await env.initial_observation()
        action = renderer.tokenizer.encode(ANALYSIS + REORDERED, add_special_tokens=False)
        first = await env.step(action)
        env.max_trajectory_tokens = first.next_observation.length + len(action)
        second = await env.step(action)
        assert second.episode_done
        assert second.metrics["context_overflow"] == 1
        assert second.next_observation.length == 0

    asyncio.run(run())


def test_append_retains_chunks_and_full_positional_context(renderer: GptOssRenderer) -> None:
    """New tool chunks retain their type; adjacent tool messages get lookahead."""

    async def run() -> None:
        env = make_env(renderer)
        initial = await env.initial_observation()
        assert isinstance(initial, tuple)
        # ModelInput chunks need not be text. A synthetic chunk avoids image processing
        # while testing that the adapter never flattens/re-tokenizes their payload.
        image = ImageChunk(data=b"image", format="png", expected_tokens=4)
        original = tinker.ModelInput(chunks=[*initial[0].chunks, image])
        env._latest_observation = original
        action = renderer.tokenizer.encode(ANALYSIS + REORDERED, add_special_tokens=False)
        parsed, _ = renderer.parse_response(action)
        tools: list[Message] = [
            {"role": "tool", "name": "search", "content": "first"},
            {"role": "tool", "name": "search", "content": "second"},
        ]
        messages: list[Message] = [{"role": "user", "content": "Find it."}, parsed, *tools]
        result = MessageStepResult(0, False, messages, appended_messages=tools)
        contexts: list[RenderContext] = []

        def render(message: Message, ctx: RenderContext) -> RenderedMessage:
            contexts.append(ctx)
            return RenderedMessage(
                header=tinker.EncodedTextChunk(tokens=[10]),
                output=[image, tinker.EncodedTextChunk(tokens=[])],
            )

        with patch.object(renderer, "render_message", side_effect=render):
            appended = env._append_sampled_tokens(action, result)
        assert _is_prefix(
            _flatten_chunks(original.chunks) + action, _flatten_chunks(appended.chunks)
        )
        assert sum(isinstance(c, ImageChunk) for c in appended.chunks) == 3
        assert all(not isinstance(c, tinker.EncodedTextChunk) or c.tokens for c in appended.chunks)
        assert [ctx.idx for ctx in contexts] == [2, 3, 4]
        assert contexts[0].prev_message == parsed
        assert contexts[0].next_message == tools[1]
        assert contexts[1].next_message is None
        assert [ctx.is_last for ctx in contexts] == [False, True, True]
        assert contexts[2].prev_message == tools[-1]
        assert all(ctx.last_user_index == 0 and ctx.in_last_assistant_turn for ctx in contexts)

    asyncio.run(run())


@pytest.mark.parametrize("preserve", [False, True])
def test_structural_failure_preserves_legacy_behavior(
    renderer: GptOssRenderer, preserve: bool
) -> None:
    async def run() -> None:
        env = make_env(renderer, preserve)
        env.terminate_on_parse_error = False
        await env.initial_observation()
        action = renderer.tokenizer.encode(ANALYSIS + REORDERED, add_special_tokens=False)
        await env.step(action)
        history = list(await env.message_env.initial_observation())
        broken = renderer.tokenizer.encode("broken", add_special_tokens=False)
        for _ in range(2):
            failure = await env.step(broken)
            assert not failure.episode_done
            assert failure.reward == env.failed_parse_reward
            assert failure.metrics["parse_error"] == 1.0
            assert await env.message_env.initial_observation() == history
            assert failure.next_observation.length == 0
            assert env._latest_observation is None
        success = await env.step(action)
        assert not success.episode_done
        messages = await env.message_env.initial_observation()
        assert (
            success.next_observation.to_ints()
            == renderer.build_generation_prompt(messages).to_ints()
        )
        if preserve:
            next_step = await env.step(action)
            assert _is_prefix(
                success.next_observation.to_ints() + action, next_step.next_observation.to_ints()
            )

    asyncio.run(run())


@pytest.mark.parametrize("explicit_policy", [False, True])
def test_structural_failure_still_terminates_when_configured(
    renderer: GptOssRenderer, explicit_policy: bool
) -> None:
    async def run() -> None:
        env = make_env(renderer)
        await env.initial_observation()
        if explicit_policy:
            env.terminate_on_parse_error = False
            env.set_parse_error_policy(ParseErrorPolicy(max_consecutive=2))
        failure = await env.step(renderer.tokenizer.encode("broken", add_special_tokens=False))
        assert failure.episode_done
        assert failure.next_observation.length == 0
        assert failure.metrics["stop/parse_error"] == 1.0

    asyncio.run(run())


@pytest.mark.parametrize("invalid", ["wrong_tail", "no_assistant"])
def test_incremental_render_rejects_invalid_append_declaration(
    renderer: GptOssRenderer, invalid: str
) -> None:
    async def run() -> None:
        env = make_env(renderer)
        await env.initial_observation()
        tool_result: Message = {"role": "tool", "name": "search", "content": "result"}
        if invalid == "wrong_tail":
            messages: list[Message] = [{"role": "assistant", "content": ""}, tool_result]
            appended: list[Message] = [{"role": "tool", "name": "search", "content": "other"}]
        else:
            messages = [{"role": "user", "content": "question"}, tool_result]
            appended = [tool_result]
        step = MessageStepResult(0, False, messages, appended_messages=appended)
        with pytest.raises(ValueError, match="appended_messages must"):
            env._append_sampled_tokens([], step)

    asyncio.run(run())


def test_terminal_tool_result_is_preserved(renderer: GptOssRenderer) -> None:
    async def run() -> None:
        def finish(q: str) -> ToolResult:
            return simple_tool_result(f"Finished {q}", should_stop=True)

        env = build_agent_tool_env(
            renderer=renderer,
            tools=[FunctionTool(finish)],
            initial_messages=[{"role": "user", "content": "Finish the task."}],
            reward_fn=reward,
            preserve_sampled_tokens=True,
        )
        initial = await env.initial_observation()
        assert isinstance(initial, tuple)
        action = renderer.tokenizer.encode(
            (ANALYSIS + REORDERED).replace("functions.search", "functions.finish"),
            add_special_tokens=False,
        )
        result = await env.step(action)
        assert result.episode_done
        assert result.reward == 1.0
        assert _is_prefix(initial[0].to_ints() + action, result.next_observation.to_ints())
        tail = renderer.tokenizer.decode(
            result.next_observation.to_ints()[initial[0].length + len(action) :]
        )
        assert "Finished x" in tail
        assert tail.endswith("<|start|>assistant")

    asyncio.run(run())
