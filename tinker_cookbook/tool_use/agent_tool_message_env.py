"""Tool-using agent environment."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import (
    PARSE_FAILURE_DETAIL_MAX_CHARS,
    Message,
    ToolCall,
    UnparsedToolCall,
    format_content_as_string,
    get_text_content,
)
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, TerminationRewardPolicy
from tinker_cookbook.rl.rollout_presets import (
    RolloutConfig,
    ToolExecution,
    default_rollout_config_for_model,
)
from tinker_cookbook.tool_use.tools import handle_tool_call
from tinker_cookbook.tool_use.types import Tool, ToolResult

RewardResult = tuple[float, dict[str, float]]
RewardFn = Callable[[list[Message]], Awaitable[RewardResult]]
# TODO(tyler): Consider supporting stateful tools that need to grade rollouts based on
# information not contained in the message history (e.g., internal tool state that changes
# during execution).


@dataclass
class AgentToolMessageEnv(MessageEnv):
    """Generic tool-use MessageEnv for agents."""

    tools: list[Tool]
    initial_messages: list[Message]
    max_turns: int
    reward_fn: RewardFn
    failed_parse_reward: float = -0.1
    terminate_on_parse_error: bool = True
    max_tool_calls: int | None = None
    parse_error_policy: ParseErrorPolicy | None = None
    tool_execution: ToolExecution = "parallel"
    termination_policy: TerminationRewardPolicy | None = None
    history: list[Message] = field(default_factory=list)

    _turn_count: int = 0
    _consecutive_parse_errors: int = field(default=0, init=False)
    _tool_calls_made: int = field(default=0, init=False)
    _tool_dict: dict[str, Tool] = field(default_factory=dict, init=False)
    _should_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._tool_dict = {t.name: t for t in self.tools}
        if self.tool_execution == "concurrent_safe":
            # The public Tool protocol has no concurrency-safety marker yet,
            # so there is nothing to key the safe/unsafe split off. Reserved
            # until the tool contract grows one; use "sequential" (always
            # safe) or "parallel" (the default) instead.
            raise NotImplementedError(
                'tool_execution="concurrent_safe" is reserved: the Tool protocol has '
                "no concurrency-safety marker yet. Use 'sequential' or 'parallel'."
            )

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    def set_max_tool_calls(self, max_tool_calls: int) -> None:
        """Apply a rollout-level tool-call limit (seam used by the rollout runner).

        When a limit is already configured on the env, the tighter cap wins.
        """
        if self.max_tool_calls is None:
            self.max_tool_calls = max_tool_calls
        else:
            self.max_tool_calls = min(self.max_tool_calls, max_tool_calls)

    def set_parse_error_policy(self, policy: ParseErrorPolicy) -> None:
        """Apply a rollout-level parse-error policy (seam used by the rollout runner)."""
        self.parse_error_policy = policy

    async def observe_truncated_response(self, message: Message) -> list[Message]:
        """Record a truncated assistant response (LENGTH-continue support).

        The message is appended to the history as-is; no tools run and the
        turn/tool-call counters are untouched.
        """
        self.history.append(message)
        return self.history

    async def add_messages(self, messages: list[Message]) -> list[Message]:
        """Append externally injected messages (rollout-hook support)."""
        self.history.extend(messages)
        return self.history

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        """Execute tool calls and append results to history.

        Execution order follows ``tool_execution``: ``"parallel"`` (default)
        dispatches the whole batch concurrently via ``asyncio.gather``;
        ``"sequential"`` awaits each call in the order the model requested
        them (required when tools share mutable state and call order
        matters).  Result messages are appended in request order either way.

        Note: Tool metrics are not accumulated in the message history.
        Only messages and should_stop are used from ToolResult.
        """
        tool_results: list[ToolResult]
        if self.tool_execution == "sequential":
            tool_results = [await handle_tool_call(self._tool_dict, tc) for tc in tool_calls]
        else:
            tool_results = list(
                await asyncio.gather(*[handle_tool_call(self._tool_dict, tc) for tc in tool_calls])
            )

        all_messages: list[Message] = []

        for tool_result in tool_results:
            # Append messages to history
            for msg in tool_result.messages:
                self.history.append(msg)
                all_messages.append(msg)

            # Check if any tool signals to stop
            if tool_result.should_stop:
                self._should_stop = True

        return all_messages

    async def step(self, message: Message) -> MessageStepResult:
        """Execute any tools and return next messages.

        The episode ends when:
        - no tool calls in message (model decided to stop)
        - every tool call in the message failed to parse and
          ``terminate_on_parse_error`` is True
        - a tool returns should_stop=True
        - max_turns reached
        - the tool-call budget (``max_tool_calls``, when configured) is
          exhausted — checked pre-dispatch and mid-batch, so a turn asking
          for more calls than the remaining budget executes only the allowed
          prefix and then stops

        Tool calls whose arguments failed to parse (e.g. malformed JSON)
        arrive in ``message["unparsed_tool_calls"]``.  When the message
        contains *only* unparsed tool calls, the turn is a parse failure:
        ``failed_parse_reward`` is applied, ``metrics["parse_error"]`` is set,
        the error details are recorded in ``logs["parse_errors"]``, and
        ``reward_fn`` is skipped.  When valid and unparsed tool calls are
        mixed, the valid ones execute and the episode proceeds normally with
        no failure reward (some tool progress happened), but the parse error
        is still surfaced via ``metrics["parse_error"]`` and
        ``logs["parse_errors"]``.

        reward_fn is called once at episode end to grade the full trajectory.
        """
        self._turn_count += 1
        metrics: dict[str, float] = {}
        logs: types.Logs = {}

        # Append the message to history
        self.history.append(message)

        # Log assistant content (handles both str and multimodal content)
        assistant_text = get_text_content(message)
        if assistant_text:
            logs["assistant_content"] = assistant_text

        tool_calls: list[ToolCall] = list(message.get("tool_calls") or [])
        unparsed_tool_calls: list[UnparsedToolCall] = list(message.get("unparsed_tool_calls") or [])
        max_turns_reached = self._turn_count >= self.max_turns

        if unparsed_tool_calls:
            metrics["parse_error"] = 1.0
            logs["parse_errors"] = "\n".join(tc.error for tc in unparsed_tool_calls)

        if unparsed_tool_calls and not tool_calls:
            # The model tried to call tools but every call was malformed —
            # a content-level parse failure (the framing itself was clean).
            self._consecutive_parse_errors += 1
            if self.parse_error_policy is not None:
                return self._content_parse_error_step(
                    unparsed_tool_calls, metrics, logs, max_turns_reached
                )
            # Default one-shot semantics (no policy): failed_parse_reward,
            # reward_fn skipped, terminate_on_parse_error respected.
            done = self.terminate_on_parse_error or max_turns_reached
            if max_turns_reached:
                metrics["max_turns"] = 1.0
            if done:
                metrics[f"{types.STOP_METRIC_PREFIX}{types.StopReason.PARSE_ERROR}"] = 1.0
            return MessageStepResult(
                reward=self.failed_parse_reward,
                episode_done=done,
                next_messages=self.history,
                metrics=metrics,
                logs=logs,
            )

        # This turn was not a parse failure: the consecutive-error budget
        # (ParseErrorPolicy.max_consecutive) applies to *consecutive* errors.
        self._consecutive_parse_errors = 0

        # Enforce the tool-call budget when configured (default None = no cap):
        # pre-dispatch (a batch arriving with no budget left dispatches nothing)
        # and mid-batch (a batch larger than the remaining budget is cut at the
        # boundary; only the allowed prefix executes). Either way the episode
        # ends with StopReason.MAX_TOOL_CALLS.
        tool_calls_capped = False
        tool_calls_to_run = tool_calls
        if tool_calls and self.max_tool_calls is not None:
            remaining_budget = max(self.max_tool_calls - self._tool_calls_made, 0)
            if len(tool_calls) > remaining_budget:
                tool_calls_capped = True
                tool_calls_to_run = tool_calls[:remaining_budget]
                logs["tool_calls_dropped"] = (
                    f"{len(tool_calls) - remaining_budget} tool call(s) dropped: "
                    f"max_tool_calls={self.max_tool_calls} reached"
                )

        # Execute valid tool calls if present
        if tool_calls_to_run:
            for i, tc in enumerate(tool_calls_to_run):
                logs[f"tool_call_{i}"] = f"{tc.function.name}({tc.function.arguments})"

            tool_result_messages = await self._handle_tool_calls(tool_calls_to_run)
            self._tool_calls_made += len(tool_calls_to_run)

            for i, msg in enumerate(tool_result_messages):
                logs[f"tool_result_{i}"] = format_content_as_string(msg["content"])

        # Determine if episode is done
        no_tool_calls = len(tool_calls) == 0
        done = no_tool_calls or max_turns_reached or self._should_stop or tool_calls_capped

        if max_turns_reached and not no_tool_calls:
            metrics["max_turns"] = 1.0
        if self._should_stop:
            metrics["tool_stopped"] = 1.0
        if tool_calls_capped:
            metrics["max_tool_calls"] = 1.0

        if done:
            if no_tool_calls:
                stop_reason = types.StopReason.COMPLETED
            elif self._should_stop:
                stop_reason = types.StopReason.TOOL_STOPPED
            elif tool_calls_capped:
                stop_reason = types.StopReason.MAX_TOOL_CALLS
            else:
                stop_reason = types.StopReason.MAX_TURNS
            metrics[f"{types.STOP_METRIC_PREFIX}{stop_reason}"] = 1.0

        reward = 0.0
        if done:
            reward, reward_metrics = await self._grade()
            metrics.update(reward_metrics)

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=self.history,
            metrics=metrics,
            logs=logs,
        )

    async def _grade(self) -> RewardResult:
        """Run ``reward_fn`` under the termination policy's grader knobs.

        With no ``termination_policy`` (the default), this is exactly the
        default call: ``reward_fn(self.history)``, unbounded, full history.
        With a policy installed (e.g. via ``build_agent_tool_env``'s
        ``rollout_config``):

        - the grader sees only the completion suffix — the messages generated
          during the rollout, excluding ``initial_messages`` — unless
          ``pass_all_messages_to_grader`` is True;
        - the call is bounded by ``grader_timeout_seconds`` when set.  A
          timeout raises ``TimeoutError`` (there is no usable reward),
          composing with error-tolerant rollout strategies that retry or
          drop the rollout.

        Note the rollout-timeout case never reaches this method: a
        runner-imposed ``rollout_timeout`` stop ends the rollout without the
        env observing episode end, so the env-level grader structurally never
        runs for it (``skip_grading_on_timeout`` concerns the group-level
        grader; see ``do_group_rollout``).
        """
        policy = self.termination_policy
        messages = self.history
        if policy is not None and not policy.pass_all_messages_to_grader:
            messages = self.history[len(self.initial_messages) :]
        timeout = policy.grader_timeout_seconds if policy is not None else None
        async with asyncio.timeout(timeout):
            return await self.reward_fn(messages)

    def _content_parse_error_step(
        self,
        unparsed_tool_calls: list[UnparsedToolCall],
        metrics: dict[str, float],
        logs: types.Logs,
        max_turns_reached: bool,
    ) -> MessageStepResult:
        """Handle a content parse failure under a :class:`ParseErrorPolicy`.

        While the consecutive-error budget lasts (and the turn cap has not
        been hit), inject the policy's corrective message as the next user
        message and continue with ``-penalty_per_error`` as this turn's
        reward; the injected tokens count toward the trajectory token budget
        because the conversation is re-rendered into the next observation.
        Once the budget is exceeded, end the episode with
        ``StopReason.PARSE_ERROR`` and ``terminal_reward`` (``reward_fn`` is
        skipped either way).  Supersedes the default one-shot semantics
        (``failed_parse_reward`` / ``terminate_on_parse_error``).
        """
        policy = self.parse_error_policy
        assert policy is not None
        logs["parse_failure_kind"] = "content"
        if policy.mask_error_turns:
            metrics[types.PARSE_ERROR_MASKED_METRIC_KEY] = 1.0

        if self._consecutive_parse_errors <= policy.max_consecutive and not max_turns_reached:
            details = "\n".join(tc.error for tc in unparsed_tool_calls)
            details = details[:PARSE_FAILURE_DETAIL_MAX_CHARS]
            retry_message: Message = {
                "role": "user",
                "content": policy.retry_message_template.format(details=details),
            }
            self.history.append(retry_message)
            logs["parse_error_retry"] = (
                f"injected corrective message (consecutive parse error "
                f"{self._consecutive_parse_errors}, budget {policy.max_consecutive})"
            )
            return MessageStepResult(
                reward=-policy.penalty_per_error,
                episode_done=False,
                next_messages=self.history,
                metrics=metrics,
                logs=logs,
            )

        if max_turns_reached:
            metrics["max_turns"] = 1.0
        metrics[f"{types.STOP_METRIC_PREFIX}{types.StopReason.PARSE_ERROR}"] = 1.0
        return MessageStepResult(
            reward=policy.terminal_reward,
            episode_done=True,
            next_messages=self.history,
            metrics=metrics,
            logs=logs,
        )


def build_agent_tool_env(
    renderer: Renderer,
    tools: list[Tool],
    initial_messages: list[Message],
    reward_fn: RewardFn,
    *,
    rollout_config: RolloutConfig | None = None,
    model_name: str | None = None,
    max_turns: int | None = None,
    failed_parse_reward: float = -0.1,
    terminate_on_parse_error: bool = True,
    max_tool_calls: int | None = None,
    max_trajectory_tokens: int | None = None,
    max_generation_tokens: int | None = None,
    context_overflow_reward: float = -0.1,
    terminate_on_length: bool | None = None,
    parse_error_policy: ParseErrorPolicy | None = None,
) -> EnvFromMessageEnv:
    """Convenience method to build an EnvFromMessageEnv for tool-using agents.

    The primary configuration surface is ``rollout_config``: one composite
    :class:`~tinker_cookbook.rl.rollout_presets.RolloutConfig` (e.g. from
    :func:`~tinker_cookbook.rl.rollout_presets.simple` or
    :func:`~tinker_cookbook.rl.rollout_presets.agentic`) that drives the
    env-side knobs here, the runner-side budgets when the same config is
    passed to ``run_rollout``, and the reward semantics when its
    ``termination`` is passed to ``do_group_rollout``.  The individual
    keyword arguments below remain supported; where both are given, the
    explicit keyword wins.

    Args:
        renderer: The renderer for tokenizing messages.
        tools: List of tools the agent can call (must implement Tool protocol).
        initial_messages: Initial conversation history (system prompt, user message, etc.).
        reward_fn: Function that grades a completed episode. Takes the message
            history and returns (reward, metrics). Called once at episode end.
            Receives the full history by default; with a rollout_config whose
            termination policy is set, it receives only the completion suffix
            (messages generated during the rollout) unless
            ``pass_all_messages_to_grader`` is True, and is bounded by
            ``grader_timeout_seconds`` when set.
        rollout_config: Composite rollout configuration. Applies the env-side
            pieces: ``parse_errors`` (unless ``parse_error_policy`` is given),
            ``limits.max_turns`` (unless ``max_turns`` is given; the env
            enforces the cap itself so the turn-capped episode still grades,
            feeding grade-then-clamp), ``limits.max_tool_calls`` (combined
            with ``max_tool_calls``, tighter cap wins), ``tool_execution``,
            and ``termination`` (grader timeout/scope). When the config sets
            any cumulative token budget (``max_trajectory_tokens`` /
            ``max_sampled_tokens`` / ``max_turn_tokens``), the env defaults to
            ``terminate_on_length=False`` so the runner owns LENGTH handling
            and budget enforcement; the limits are also advertised to the
            runner via ``EnvFromMessageEnv.rollout_limits``, so the training
            loop (which calls ``run_rollout`` with no configuration) still
            enforces them. Default ``None`` resolves through ``model_name``
            when given (see below); with neither, every knob keeps its
            documented default.
        model_name: The model this env will run against. When set and
            ``rollout_config`` is not, the model's default configuration
            applies (:func:`~tinker_cookbook.rl.rollout_presets.default_rollout_config_for_model`:
            ``thinkingmachines/Inkling`` defaults to
            :func:`~tinker_cookbook.rl.rollout_presets.agentic`). Pass
            ``rollout_config=simple()`` to opt out of a model default.
        max_turns: Maximum turns before episode ends. Default ``None`` means
            ``rollout_config.limits.max_turns`` when set, else 5.
        failed_parse_reward: Reward when model output fails to parse. Applied both
            when the renderer cannot parse the response and when a turn's tool
            calls are all malformed (``unparsed_tool_calls``).
        terminate_on_parse_error: Whether a parse failure ends the episode.
            Defaults to True.
        max_tool_calls: Maximum tool calls per episode (checked pre-dispatch and
            mid-batch; the episode ends with StopReason.MAX_TOOL_CALLS). Default
            None = no cap. A rollout runner configured with
            ``RolloutLimits.max_tool_calls`` also sets this (the tighter cap wins).
        max_trajectory_tokens: Maximum tokens in trajectory before terminating episode.
        max_generation_tokens: Maximum tokens per generation. When set, the episode
            terminates if the trajectory + generation budget would exceed
            *max_trajectory_tokens*, preventing context overflow errors.
        context_overflow_reward: Reward assigned when the episode is terminated due to
            context overflow. Defaults to -0.1.
        terminate_on_length: Whether a per-turn sampler "length" stop ends the
            episode. Set False (LENGTH-continue) to keep the truncated turn
            and continue — intended for use with a rollout runner enforcing
            cumulative token budgets via ``RolloutLimits``, which is then what
            bounds the episode. Default ``None`` resolves to False when
            ``rollout_config`` configures a cumulative token budget (the
            runner owns LENGTH handling), else True (the default).
        parse_error_policy: Optional :class:`ParseErrorPolicy` governing parse
            failures. When set, it supersedes the one-shot
            ``failed_parse_reward`` / ``terminate_on_parse_error`` semantics
            for parse-error turns: structural failures (broken framing) stop
            immediately with ``StopReason.PARSE_ERROR``; content failures
            (unparsable tool calls) are retried with an injected corrective
            message up to ``max_consecutive`` times. Default ``None`` keeps
            those one-shot semantics. A rollout runner configured with a policy also
            sets this via ``set_parse_error_policy``.

    Returns:
        An EnvFromMessageEnv ready for RL training.
    """
    cfg = rollout_config
    if cfg is None and model_name is not None:
        cfg = default_rollout_config_for_model(model_name)
    if max_turns is None:
        if cfg is not None and cfg.limits.max_turns is not None:
            max_turns = cfg.limits.max_turns
        else:
            max_turns = 5
    if parse_error_policy is None and cfg is not None:
        parse_error_policy = cfg.parse_errors
    if cfg is not None and cfg.limits.max_tool_calls is not None:
        max_tool_calls = (
            cfg.limits.max_tool_calls
            if max_tool_calls is None
            else min(max_tool_calls, cfg.limits.max_tool_calls)
        )
    if terminate_on_length is None:
        runner_owns_length = cfg is not None and (
            cfg.limits.max_trajectory_tokens is not None
            or cfg.limits.max_sampled_tokens is not None
            or cfg.limits.max_turn_tokens is not None
        )
        terminate_on_length = not runner_owns_length

    msg_env = AgentToolMessageEnv(
        tools=tools,
        initial_messages=initial_messages,
        max_turns=max_turns,
        reward_fn=reward_fn,
        failed_parse_reward=failed_parse_reward,
        terminate_on_parse_error=terminate_on_parse_error,
        max_tool_calls=max_tool_calls,
        parse_error_policy=parse_error_policy,
        tool_execution=cfg.tool_execution if cfg is not None else "parallel",
        termination_policy=cfg.termination if cfg is not None else None,
    )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=msg_env,
        failed_parse_reward=failed_parse_reward,
        terminate_on_parse_error=terminate_on_parse_error,
        max_trajectory_tokens=max_trajectory_tokens,
        max_generation_tokens=max_generation_tokens,
        context_overflow_reward=context_overflow_reward,
        terminate_on_length=terminate_on_length,
        parse_error_policy=parse_error_policy,
        rollout_limits=cfg.limits if cfg is not None else None,
    )
