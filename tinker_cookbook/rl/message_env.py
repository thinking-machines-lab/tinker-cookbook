"""Message-level environment abstraction.

MessageEnv operates at the message level (list[Message]) rather than token level.

EnvFromMessageEnv bridges MessageEnv to the token-level Env interface used by
the RL training loop.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import tinker

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message, ParseTermination, classify_parse_failure
from tinker_cookbook.rl import types
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits

logger = logging.getLogger(__name__)


@dataclass
class MessageStepResult:
    """Result of a message-level step."""

    reward: float
    episode_done: bool
    next_messages: list[Message]
    metrics: dict[str, float] = field(default_factory=dict)
    logs: types.Logs = field(default_factory=dict)
    next_stop_condition: StopCondition | None = None


class MessageEnv(ABC):
    """Abstract base class for message-level environments."""

    @abstractmethod
    async def initial_observation(self) -> list[Message]:
        """Return the initial conversation history as renderer messages."""
        ...

    @abstractmethod
    async def step(self, message: Message) -> MessageStepResult:
        """Process an assistant message and return reward/next state."""
        ...

    async def observe_truncated_response(self, message: Message) -> list[Message] | None:
        """Record a truncated assistant response without running a full step.

        Used by the continue-past-truncation path (the sampler hit the
        per-turn ``max_tokens`` cap, ``stop_reason == "length"``, and
        ``EnvFromMessageEnv`` has ``terminate_on_length=False``): the
        truncated message is appended to
        the conversation as-is (no tool execution, no completion check) so the
        rollout can continue on the next turn.

        Returns:
            list[Message] | None: The updated message history, or ``None`` if
                this environment does not support recording truncated
                responses (the default), in which case the caller falls back
                to the default terminate-on-length behavior.
        """
        return None

    async def add_messages(self, messages: list[Message]) -> list[Message] | None:
        """Append externally injected messages to the conversation.

        Used by rollout hooks (``on_turn_begin``) to inject messages before a
        sampling call.

        Returns:
            list[Message] | None: The updated message history, or ``None`` if
                this environment does not support message injection (the
                default).
        """
        return None


class EnvFromMessageEnv(types.Env):
    """Adapter that wraps a MessageEnv to implement the token-level Env interface.

    This bridges the message-level abstraction to the token-level interface
    expected by the RL training loop.
    """

    def __init__(
        self,
        renderer: Renderer,
        message_env: MessageEnv,
        failed_parse_reward: float = -1.0,
        terminate_on_parse_error: bool = True,
        max_trajectory_tokens: int | None = None,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
        terminate_on_length: bool = True,
        parse_error_policy: ParseErrorPolicy | None = None,
        rollout_limits: RolloutLimits | None = None,
    ):
        self.renderer = renderer
        self.message_env = message_env
        self.failed_parse_reward = failed_parse_reward
        self.terminate_on_parse_error = terminate_on_parse_error
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self.terminate_on_length = terminate_on_length
        self.parse_error_policy = parse_error_policy
        # Budgets this env wants a rollout runner to enforce. The runner reads
        # this when it is given no limits of its own (run_rollout's fallback),
        # so envs built from a RolloutConfig keep their token budgets in the
        # training loop, which calls run_rollout with no configuration.
        self.rollout_limits = rollout_limits
        self._base_stop_condition = renderer.get_stop_sequences()

        # Forward example_id from the inner MessageEnv for trajectory storage.
        # This ensures truncated examples (where MessageEnv.step() never runs)
        # still get the correct example_id in stored trajectories.
        self.example_id: str | None = getattr(message_env, "example_id", None)

    async def _render_in_thread(self, messages: list[Message], **kwargs) -> tinker.ModelInput:
        """Run build_generation_prompt in a thread to avoid blocking the event loop.

        Tokenization is CPU-bound. With many concurrent tasks on the same event
        loop, running it synchronously starves other coroutines. HuggingFace
        tokenizers release the GIL, so threads give true parallelism.
        """
        return await asyncio.to_thread(self.renderer.build_generation_prompt, messages, **kwargs)

    def _exceeds_context_limit(self, observation_length: int) -> bool:
        """Check if the observation + generation budget exceeds the context limit."""
        if self.max_trajectory_tokens is None:
            return False
        generation_reserve = self.max_generation_tokens or 0
        return observation_length + generation_reserve > self.max_trajectory_tokens

    async def initial_observation(
        self,
    ) -> tuple[tinker.ModelInput, StopCondition] | types.InitialObservationOverflow:
        messages = await self.message_env.initial_observation()
        model_input = await self._render_in_thread(messages)

        if self._exceeds_context_limit(model_input.length):
            # The prompt is too long before the first sampling call. End the
            # rollout gracefully (same flat context_overflow_reward and
            # MAX_TOKENS stop reason as a mid-turn "length" stop) instead of
            # raising, so one oversized prompt doesn't fail the whole group.
            generation_reserve = self.max_generation_tokens or 0
            return types.InitialObservationOverflow(
                reward=self.context_overflow_reward,
                metrics={
                    "max_tokens_reached": 1.0,
                    f"{types.STOP_METRIC_PREFIX}{types.StopReason.MAX_TOKENS}": 1.0,
                },
                logs={
                    "initial_observation_overflow": (
                        f"Initial observation ({model_input.length} tokens) + "
                        f"max_generation_tokens ({generation_reserve}) = "
                        f"{model_input.length + generation_reserve} exceeds "
                        f"max_trajectory_tokens ({self.max_trajectory_tokens}). "
                        f"This task's prompt is too long for the model's context window."
                    )
                },
            )

        return model_input, self._base_stop_condition

    async def step(
        self, action: types.Action, *, extra: types.ActionExtra | None = None
    ) -> types.StepResult:
        """Parse tokens to a message, delegate to MessageEnv, and render response."""
        # If the model hit max_tokens without producing a stop sequence, terminate
        # the episode early. Previous turns' logprobs are preserved in the trajectory.
        # With terminate_on_length=False (LENGTH-continue, for use with a rollout
        # runner enforcing cumulative budgets), the truncated turn is kept and the
        # episode continues instead.
        stop_reason = (extra or {}).get("stop_reason", "stop")
        if stop_reason == "length":
            if not self.terminate_on_length:
                continue_result = await self._continue_after_truncation(action)
                if continue_result is not None:
                    return continue_result
                # The inner MessageEnv can't record truncated responses:
                # fall back to the default terminate-on-length behavior.
            return types.StepResult(
                reward=self.context_overflow_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={
                    "max_tokens_reached": 1.0,
                    f"{types.STOP_METRIC_PREFIX}{types.StopReason.MAX_TOKENS}": 1.0,
                },
            )

        assistant_message, termination = self.renderer.parse_response(action)

        if not termination.is_clean:
            # STRUCTURAL parse failure: the response never produced its stop
            # signal, so the message boundary is unknown and the conversation
            # state is corrupted. Never retried (unlike content failures):
            # re-rendering a broken-framing turn would put the model on a
            # garbage observation.
            if self.parse_error_policy is not None:
                return self._structural_parse_error_step(assistant_message, termination)
            parse_metrics: types.Metrics = {"parse_error": 1.0}
            if self.terminate_on_parse_error:
                parse_metrics[f"{types.STOP_METRIC_PREFIX}{types.StopReason.PARSE_ERROR}"] = 1.0
            return types.StepResult(
                reward=self.failed_parse_reward,
                episode_done=self.terminate_on_parse_error,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics=parse_metrics,
            )

        msg_step = await self.message_env.step(assistant_message)
        next_observation = await self._render_in_thread(msg_step.next_messages)
        next_stop_condition = msg_step.next_stop_condition or self._base_stop_condition

        # Check if the full trajectory + generation budget fits in the context window.
        # next_observation is the entire rendered conversation so far, which becomes
        # the prompt for the next sampling call. Only check when the episode continues —
        # if episode_done, there is no next sampling call and the real reward should be kept.
        if not msg_step.episode_done and self._exceeds_context_limit(next_observation.length):
            return types.StepResult(
                reward=self.context_overflow_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={
                    **msg_step.metrics,
                    "context_overflow": 1.0,
                    f"{types.STOP_METRIC_PREFIX}{types.StopReason.CONTEXT_OVERFLOW}": 1.0,
                },
                logs=msg_step.logs,
            )

        return types.StepResult(
            reward=msg_step.reward,
            episode_done=msg_step.episode_done,
            next_observation=next_observation,
            next_stop_condition=next_stop_condition,
            metrics=msg_step.metrics,
            logs=msg_step.logs,
        )

    def _structural_parse_error_step(
        self, assistant_message: Message, termination: ParseTermination
    ) -> types.StepResult:
        """End the episode on a structural parse failure (policy configured).

        Immediate stop with ``StopReason.PARSE_ERROR`` and the policy's
        ``terminal_reward``; supersedes the default one-shot semantics
        (``failed_parse_reward`` / ``terminate_on_parse_error``).  See :class:`ParseErrorPolicy`
        for why structural failures are never retried.
        """
        assert self.parse_error_policy is not None
        classified = classify_parse_failure(assistant_message, termination)
        assert classified is not None
        kind, detail = classified
        metrics: types.Metrics = {
            "parse_error": 1.0,
            f"{types.STOP_METRIC_PREFIX}{types.StopReason.PARSE_ERROR}": 1.0,
        }
        if self.parse_error_policy.mask_error_turns:
            metrics[types.PARSE_ERROR_MASKED_METRIC_KEY] = 1.0
        return types.StepResult(
            reward=self.parse_error_policy.terminal_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self._base_stop_condition,
            metrics=metrics,
            logs={
                "parse_failure_kind": str(kind),
                "parse_failure_detail": detail,
            },
        )

    def set_parse_error_policy(self, policy: ParseErrorPolicy) -> None:
        """Apply a rollout-level parse-error policy (seam used by the rollout
        runner: the narrow method interface through which it pushes
        configuration into the env).

        Configures this adapter's structural-failure handling and forwards the
        policy to the inner :class:`MessageEnv` when it supports one (content
        failures — unparsable tool calls — are handled there).
        """
        self.parse_error_policy = policy
        setter = getattr(self.message_env, "set_parse_error_policy", None)
        if setter is not None:
            setter(policy)
        elif policy.max_consecutive > 0:
            logger.warning(
                "ParseErrorPolicy.max_consecutive=%d is configured, but %s does not "
                "support a parse-error policy; content parse failures will not be "
                "retried (structural handling still applies).",
                policy.max_consecutive,
                type(self.message_env).__name__,
            )

    async def _continue_after_truncation(self, action: types.Action) -> types.StepResult | None:
        """LENGTH-continue: keep the truncated turn and let the episode proceed.

        The truncated response is parsed leniently (a truncated action rarely
        parses cleanly, so the parse termination is deliberately ignored) and
        recorded in the conversation without tool execution or a completion
        check.  Returns ``None`` when the inner :class:`MessageEnv` does not
        support recording truncated responses.
        """
        assistant_message, _termination = self.renderer.parse_response(action)
        updated_messages = await self.message_env.observe_truncated_response(assistant_message)
        if updated_messages is None:
            return None
        next_observation = await self._render_in_thread(updated_messages)

        # The env-level trajectory budget (when configured) still applies to
        # the grown conversation, same as the post-step overflow check.
        if self._exceeds_context_limit(next_observation.length):
            return types.StepResult(
                reward=self.context_overflow_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={
                    "context_overflow": 1.0,
                    f"{types.STOP_METRIC_PREFIX}{types.StopReason.CONTEXT_OVERFLOW}": 1.0,
                },
            )

        return types.StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=next_observation,
            next_stop_condition=self._base_stop_condition,
            logs={"truncated_turn": "sampling stopped on 'length'; turn kept, episode continues"},
        )

    async def inject_messages(
        self, messages: list[Message]
    ) -> tuple[tinker.ModelInput, StopCondition]:
        """Inject externally supplied messages before the next sampling call.

        Seam used by rollout hooks (``on_turn_begin``).  Forwards to the inner
        :meth:`MessageEnv.add_messages` and re-renders the conversation, so
        injected tokens count toward any trajectory budget.

        Raises:
            TypeError: If the inner :class:`MessageEnv` does not support
                message injection.
        """
        updated_messages = await self.message_env.add_messages(messages)
        if updated_messages is None:
            raise TypeError(
                f"{type(self.message_env).__name__} does not support message injection "
                "(add_messages returned None)"
            )
        next_observation = await self._render_in_thread(updated_messages)
        return next_observation, self._base_stop_condition

    def set_max_tool_calls(self, max_tool_calls: int) -> None:
        """Forward a rollout-level tool-call limit to the inner MessageEnv.

        Seam used by the rollout runner for ``RolloutLimits.max_tool_calls``
        (tool dispatch happens inside the message env, so the runner cannot
        enforce it directly).  Logs a warning when the inner env does not
        support a tool-call limit, in which case the limit is not enforced.
        """
        setter = getattr(self.message_env, "set_max_tool_calls", None)
        if setter is None:
            logger.warning(
                "max_tool_calls=%d is configured, but %s does not support a "
                "tool-call limit; the limit will not be enforced.",
                max_tool_calls,
                type(self.message_env).__name__,
            )
            return
        setter(max_tool_calls)
