"""Configurable single-rollout runner.

:func:`run_rollout` drives one episode of an :class:`~tinker_cookbook.rl.types.Env`
with a :class:`~tinker_cookbook.completers.TokenCompleter` policy.  It is the
single rollout-loop implementation in the cookbook:
:func:`~tinker_cookbook.rl.rollouts.do_single_rollout` delegates to it with no
configuration, and with ``limits=None`` and ``hooks=None`` this loop behaves
identically to ``do_single_rollout`` with no configuration (pinned by
``rollout_semantics_regression_test.py``).

Configuring :class:`~tinker_cookbook.rl.rollout_limits.RolloutLimits` and/or
:class:`RolloutHooks` layers additional, runner-owned behavior on top:

Per-turn token budget
---------------------

Before **every** policy call the runner computes the per-call token budget as
the minimum over the non-``None`` of:

- ``limits.max_turn_tokens`` (a pure per-turn cap),
- ``limits.max_trajectory_tokens - <current observation length>`` (the
  observation is the full rendered conversation, so it already folds in all
  previous action, tool, and hook-injected tokens), and
- ``limits.max_sampled_tokens - <action tokens sampled so far>``.

The budget is passed to the policy as the per-call ``max_tokens`` override
(implementations combine it with their own limit by taking the minimum; the
completer's ``context_window`` handling is unchanged and stays inside
``TinkerTokenCompleter``).  When a *cumulative* budget leaves a remainder of
``<= 1`` the rollout stops before sampling: ``StopReason.MAX_TOKENS`` when the
trajectory budget is the exhausted bound, ``StopReason.MAX_SAMPLED_TOKENS``
when the sampled budget is; when both are exhausted, the trajectory budget
wins.  A remainder of 1 is treated as exhausted because a 1-token cap cannot
produce a useful turn: ``max_tokens=1`` would sample a single token and
immediately stop.  ``max_turn_tokens`` never triggers this stop — it is a per-turn cap,
not a cumulative budget, so ``max_turn_tokens=1`` is a valid "one token per
turn" configuration.

LENGTH-continues
----------------

A per-turn sampler ``"length"`` stop does not by itself end a rollout that
opted into length-continues; the runner counts it (emitted as
``metrics["num_truncations_max_turn_tokens"]`` on the final transition) and
continues until a cumulative budget is exhausted.  By default
``EnvFromMessageEnv`` terminates on ``"length"`` *inside the env*, before the
runner can react, so the continue behavior is opt-in via the env's
``terminate_on_length=False`` constructor flag (default ``True`` preserves the
default terminate semantics exactly).  The runner-side counting activates for
any env that continues past a ``"length"`` stop; the budget enforcement above
is what bounds the continued rollout.

max_turns / max_tool_calls
--------------------------

``limits.max_turns`` is enforced by the runner before each policy call and
produces ``StopReason.MAX_TURNS`` (env-level turn caps keep working
independently and grade as before; the runner-level stop does not grade — the
env never observes the episode ending).  ``limits.max_tool_calls`` cannot be
enforced by the runner directly (tool dispatch happens inside the env), so the
runner pushes the limit into the env at rollout start through the
:class:`SupportsToolCallLimit` seam (``set_max_tool_calls``); envs that
support it (``AgentToolMessageEnv`` via ``EnvFromMessageEnv``) enforce it
pre-dispatch and mid-batch and stop with ``StopReason.MAX_TOOL_CALLS``.  If
the env does not support the seam, a warning is logged and the limit is not
enforced.

Parse errors
------------

``parse_errors`` (a :class:`~tinker_cookbook.rl.rollout_limits.ParseErrorPolicy`)
configures how parse failures in the model output are handled.  Like
``max_tool_calls``, parsing happens inside the env (the renderer's
``parse_response`` runs in ``EnvFromMessageEnv.step``), so the runner pushes
the policy into the env at rollout start through the
:class:`SupportsParseErrorPolicy` seam (``set_parse_error_policy``).  Envs
that support it distinguish the renderer contract's two failure kinds:
structural failures (broken framing — the response never produced its stop
signal, so the conversation state is corrupted and cannot be retried) stop
immediately with ``StopReason.PARSE_ERROR``, while content failures
(unparsable tool calls) are retried with an injected corrective user message
up to ``max_consecutive`` times — the injected tokens are re-rendered into
the next observation, so they count toward the trajectory token budget
enforced above.  With ``parse_errors=None`` (default), the default one-shot
parse-failure semantics (``failed_parse_reward`` / ``terminate_on_parse_error``)
apply unchanged.

Timeouts
--------

``limits.rollout_timeout_seconds`` bounds the whole rollout with an outer
``asyncio.timeout`` plus a between-turns deadline check.  Both are *graceful*:
the partial trajectory collected so far is returned with
``StopReason.ROLLOUT_TIMEOUT`` (still gradeable downstream), never an
exception.  The between-turns check passes through ``on_stop_reason`` and can
be cleared by a hook; the outer timeout cannot (the loop has already been
cancelled when it fires).

``limits.sampling_turn_timeout_seconds`` bounds each individual policy call
and **raises** :class:`SamplingTurnTimeoutError` instead.  The two are
deliberately different: a rollout that runs out of wall-clock still has a
usable, gradeable prefix, while a sampling call that timed out leaves no
usable partial turn — raising lets error-tolerant rollout strategies
(e.g. ``RetryOnFailure``) retry the whole rollout.

Stop reasons
------------

The runner sets ``Trajectory.stop_reason`` and the one-hot ``stop/<reason>``
metric on the final transition exactly once per rollout, following the
stop-metric convention established in ``rl/types.py``: env-reported stops keep
the env's metric; runner-imposed stops write their own; a hook that overrides
or clears a stop rewrites/removes the metric so at most one ``stop/`` key
survives on the final transition.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Protocol

import tinker

from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import (
    STOP_METRIC_PREFIX,
    ActionExtra,
    Env,
    InitialObservationOverflow,
    Metrics,
    StopReason,
    Trajectory,
    Transition,
)
from tinker_cookbook.utils import trace

logger = logging.getLogger(__name__)

TRUNCATION_METRIC_KEY = "num_truncations_max_turn_tokens"
"""Metric key (on the final transition) counting per-turn ``"length"``
truncations that the rollout continued past."""


class SamplingTurnTimeoutError(Exception):
    """A single sampling call exceeded ``sampling_turn_timeout_seconds``.

    Raised (rather than handled gracefully like the whole-rollout timeout)
    because a timed-out sampling call leaves no usable partial turn; raising
    composes with error-tolerant rollout strategies such as
    ``RetryOnFailure``, which retry the rollout from scratch.
    """


class SupportsToolCallLimit(Protocol):
    """Seam (the narrow method interface through which the runner pushes
    configuration into the env) for ``limits.max_tool_calls``.

    Tool dispatch happens inside the environment, so the runner cannot enforce
    a tool-call budget itself.  Environments that can enforce it (pre-dispatch
    and mid-batch) implement this method; the runner calls it once at rollout
    start when ``limits.max_tool_calls`` is configured.
    """

    def set_max_tool_calls(self, max_tool_calls: int) -> None: ...


class SupportsParseErrorPolicy(Protocol):
    """Seam through which the runner pushes a ``ParseErrorPolicy`` into an env.

    Response parsing happens inside the environment (the renderer's
    ``parse_response`` runs in the env's ``step``), so the runner cannot
    classify or retry parse failures itself.  Environments that support the
    policy (``EnvFromMessageEnv`` over ``AgentToolMessageEnv``) implement this
    method; the runner calls it once at rollout start when ``parse_errors``
    is configured.
    """

    def set_parse_error_policy(self, policy: ParseErrorPolicy) -> None: ...


@dataclass
class RolloutState:
    """Snapshot of rollout progress passed to :class:`RolloutHooks` methods.

    Attributes:
        turn_index (int): Number of completed policy calls so far (0 on the
            first ``on_turn_begin``).
        observation_tokens (int): Length of the current observation (the full
            rendered conversation that would be the next sampling prompt).
        sampled_tokens (int): Cumulative action (sampled) tokens so far.
        num_length_truncations (int): Per-turn ``"length"`` stops that the
            rollout continued past.
        elapsed_seconds (float): Wall-clock time since the rollout started.
        transitions (list[Transition]): The trajectory so far (live view —
            hooks must not mutate it).
    """

    turn_index: int
    observation_tokens: int
    sampled_tokens: int
    num_length_truncations: int
    elapsed_seconds: float
    transitions: list[Transition]


class RolloutHooks(Protocol):
    """Optional callbacks into the rollout loop.

    All methods have no-op defaults, so subclasses override only what they
    need (``class MyHooks(RolloutHooks): ...``).  Any object implementing the
    same methods also works (structural typing).
    """

    async def on_turn_begin(self, state: RolloutState) -> list[Message] | None:
        """Called before each turn's limit checks complete and sampling starts.

        Return messages to inject into the conversation before this turn's
        sampling call (requires an env supporting message injection, e.g.
        ``EnvFromMessageEnv`` over ``AgentToolMessageEnv``; a ``TypeError`` is
        raised otherwise).  Injected messages are re-rendered into the
        observation, so they count toward the trajectory token budget.
        Return ``None`` to inject nothing.
        """
        return None

    async def on_stop_reason(self, reason: str, state: RolloutState) -> str | None:
        """Called whenever a stop reason is about to end the rollout.

        Applies to both env-reported stops (e.g. ``completed``) and
        runner-imposed stops (budgets, runner ``max_turns``, the between-turns
        timeout check).  Return:

        - the same ``reason`` to accept the stop (the default),
        - a different string (including custom, non-:class:`StopReason`
          values) to override the recorded stop reason, or
        - ``None`` to *clear* the stop and continue the rollout.

        The outer whole-rollout timeout is **not** clearable: when it fires,
        the loop has already been cancelled, so this hook is not consulted.
        Clearing a cumulative-budget stop samples the next turn without the
        exhausted budget term; the check re-fires on the following turn.
        """
        return reason

    async def on_grade(self, state: RolloutState) -> None:
        """Called once after the env finishes a terminal step (i.e. after any
        env-side grading has run), before ``on_artifacts``.  Not called for
        runner-imposed stops or timeouts, where the env never observes the
        episode ending and no grading occurs."""
        return None

    async def on_artifacts(self, state: RolloutState) -> None:
        """Called once right before the trajectory is returned, on every
        graceful path (including whole-rollout timeout).  Not called when the
        rollout raises (fatal errors, per-sample timeout)."""
        return None


def stop_reason_from_metrics(metrics: Metrics) -> str | None:
    """Extract the one-hot ``stop/<reason>`` metric key, if present."""
    return next(
        (
            key.removeprefix(STOP_METRIC_PREFIX)
            for key in metrics
            if key.startswith(STOP_METRIC_PREFIX)
        ),
        None,
    )


async def run_rollout(
    policy: TokenCompleter,
    env: Env,
    *,
    limits: RolloutLimits | None = None,
    hooks: RolloutHooks | None = None,
    parse_errors: ParseErrorPolicy | None = None,
    config: RolloutConfig | None = None,
) -> Trajectory:
    """Run a single rollout (one complete episode) in the given environment.

    With ``limits=None`` and ``hooks=None`` this behaves identically to the
    unconfigured rollout loop (``do_single_rollout`` delegates here).  See the
    module docstring for the semantics each knob adds.

    Args:
        policy (TokenCompleter): The token-level policy used to generate
            actions (token sequences) from observations.
        env (Env): A single-use environment instance.  Must not be reused
            after this call returns.
        limits (RolloutLimits | None): Optional budgets bounding this rollout.
            ``None`` (default) means unlimited.
        hooks (RolloutHooks | None): Optional callbacks into the loop.
        parse_errors (ParseErrorPolicy | None): Optional policy for parse
            failures, pushed into the env at rollout start (see the module
            docstring).  ``None`` (default) keeps the default one-shot
            parse-failure semantics.
        config (RolloutConfig | None): Composite rollout configuration (e.g.
            from :func:`~tinker_cookbook.rl.rollout_presets.agentic`).  Its
            runner-side pieces (``config.limits``, ``config.parse_errors``)
            are used where the explicit ``limits`` / ``parse_errors``
            arguments were not given.  The env-side pieces (tool execution,
            termination policy on the grader) are *not* applied here — pass
            the same config to ``build_agent_tool_env`` for those, and its
            ``termination`` to ``do_group_rollout`` for the reward semantics.

    Returns:
        Trajectory: The complete (or gracefully truncated) sequence of
            transitions plus the final observation, with
            ``Trajectory.stop_reason`` set when the reason is known.

    Raises:
        SamplingTurnTimeoutError: When ``limits.sampling_turn_timeout_seconds``
            is set and a single sampling call exceeds it.
    """
    if config is not None:
        limits = limits if limits is not None else config.limits
        parse_errors = parse_errors if parse_errors is not None else config.parse_errors
    if limits is None:
        # Envs built with a RolloutConfig advertise their token budgets (see
        # EnvFromMessageEnv.rollout_limits), so callers that pass only an env
        # (the training loop's do_single_rollout) still get budget enforcement.
        limits = getattr(env, "rollout_limits", None)
    return await _RolloutRun(
        policy=policy, env=env, limits=limits, hooks=hooks, parse_errors=parse_errors
    ).run()


@dataclass
class _RolloutRun:
    """State and control flow for one rollout. Single-use, like the env."""

    policy: TokenCompleter
    env: Env
    limits: RolloutLimits | None
    hooks: RolloutHooks | None
    parse_errors: ParseErrorPolicy | None = None

    transitions: list[Transition] = field(default_factory=list)
    turn_index: int = 0
    sampled_tokens: int = 0
    num_truncations: int = 0
    _start_time: float = field(default_factory=time.monotonic)
    ob: tinker.ModelInput = field(default_factory=tinker.ModelInput.empty)
    stop_condition: StopCondition = field(default_factory=list)

    async def run(self) -> Trajectory:
        self._push_tool_call_limit()
        self._push_parse_error_policy()
        timeout = self.limits.rollout_timeout_seconds if self.limits is not None else None
        if timeout is None:
            return await self._run_loop()
        try:
            async with asyncio.timeout(timeout):
                return await self._run_loop()
        except TimeoutError:
            # Graceful whole-rollout timeout: keep the trajectory prefix
            # collected so far (still gradeable downstream) instead of
            # raising. The loop was cancelled, so this stop is not
            # hook-clearable. (SamplingTurnTimeoutError is not a TimeoutError
            # subclass, so the raising per-sample timeout passes through.)
            return await self._finalize_runner_stop(StopReason.ROLLOUT_TIMEOUT)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> Trajectory:
        async with trace.scope_span("env_initial_observation"):
            initial = await self.env.initial_observation()
        if isinstance(initial, InitialObservationOverflow):
            return await self._finalize_initial_overflow(initial)
        self.ob, self.stop_condition = initial

        while True:
            # 1. Pre-turn checks: between-turns deadline, then runner max_turns.
            reason = self._pre_turn_stop_reason()
            if reason is not None:
                resolved = await self._resolve_stop(reason)
                if resolved is not None:
                    return await self._finalize_runner_stop(resolved)

            # 2. Hook injection (before the budget math, so injected tokens
            #    count toward the trajectory budget).
            if self.hooks is not None:
                injected = await self.hooks.on_turn_begin(self._state())
                if injected:
                    self.ob, self.stop_condition = await self._inject_messages(injected)

            # 3. Cumulative-budget stop check (trajectory wins over sampled
            #    when both are exhausted), then the per-call budget.
            reason = self._budget_stop_reason()
            if reason is not None:
                resolved = await self._resolve_stop(reason)
                if resolved is not None:
                    return await self._finalize_runner_stop(resolved)
                # Cleared: sample this turn without the exhausted budget
                # term(s); the check re-fires next turn.
            budget = self._per_call_budget()

            # 4. Sample.
            ac_with_logprobs = await self._sample(budget)

            # 5. Step the env and record the transition.
            async with trace.scope_span("env_step"):
                step_result = await self.env.step(
                    ac_with_logprobs.tokens,
                    extra=ActionExtra(stop_reason=ac_with_logprobs.stop_reason),
                )
            transition = Transition(
                ob=self.ob,
                ac=ac_with_logprobs,
                reward=step_result.reward,
                episode_done=step_result.episode_done,
                metrics=step_result.metrics,
                logs=step_result.logs,
            )
            self.transitions.append(transition)
            self.turn_index += 1
            self.sampled_tokens += len(ac_with_logprobs.tokens)
            if ac_with_logprobs.stop_reason == "length" and not step_result.episode_done:
                # The env continued past a per-turn truncation (LENGTH-continue).
                self.num_truncations += 1
            self.ob = step_result.next_observation
            self.stop_condition = step_result.next_stop_condition

            if step_result.episode_done:
                env_reason = stop_reason_from_metrics(transition.metrics)
                if self.hooks is not None and env_reason is not None:
                    resolved = await self.hooks.on_stop_reason(env_reason, self._state())
                    if resolved is None:
                        # Cleared: un-terminate the recorded transition and
                        # keep rolling from the env's next observation.
                        transition.episode_done = False
                        self._strip_stop_metrics(transition)
                        continue
                    if str(resolved) != env_reason:
                        self._strip_stop_metrics(transition)
                        transition.metrics[f"{STOP_METRIC_PREFIX}{resolved}"] = 1.0
                        env_reason = str(resolved)
                return await self._finalize_env_stop(env_reason)

    # ------------------------------------------------------------------
    # Limit checks and budget math
    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    def _pre_turn_stop_reason(self) -> str | None:
        """Between-turns deadline and runner-level max_turns."""
        lim = self.limits
        if lim is None:
            return None
        if (
            lim.rollout_timeout_seconds is not None
            and self._elapsed() >= lim.rollout_timeout_seconds
        ):
            return StopReason.ROLLOUT_TIMEOUT
        if lim.max_turns is not None and self.turn_index >= lim.max_turns:
            return StopReason.MAX_TURNS
        return None

    def _budget_stop_reason(self) -> str | None:
        """Cumulative token budgets. Trajectory wins over sampled when both bind."""
        lim = self.limits
        if lim is None:
            return None
        if lim.max_trajectory_tokens is not None and (
            lim.max_trajectory_tokens - self.ob.length <= 1
        ):
            return StopReason.MAX_TOKENS
        if lim.max_sampled_tokens is not None and (
            lim.max_sampled_tokens - self.sampled_tokens <= 1
        ):
            return StopReason.MAX_SAMPLED_TOKENS
        return None

    def _per_call_budget(self) -> int | None:
        """The per-turn token budget: min over the non-None limit terms.

        Exhausted cumulative terms (remainder <= 1, only reachable when a hook
        cleared the budget stop) are dropped for this turn rather than passed
        as a nonsensical cap.
        """
        lim = self.limits
        if lim is None:
            return None
        terms: list[int] = []
        if lim.max_turn_tokens is not None:
            terms.append(lim.max_turn_tokens)
        if lim.max_trajectory_tokens is not None:
            remaining = lim.max_trajectory_tokens - self.ob.length
            if remaining > 1:
                terms.append(remaining)
        if lim.max_sampled_tokens is not None:
            remaining = lim.max_sampled_tokens - self.sampled_tokens
            if remaining > 1:
                terms.append(remaining)
        return min(terms) if terms else None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    async def _sample(self, budget: int | None) -> TokensWithLogprobs:
        turn_timeout = (
            self.limits.sampling_turn_timeout_seconds if self.limits is not None else None
        )
        async with trace.scope_span("policy_sample"):
            if turn_timeout is None:
                return await self._call_policy(budget)
            try:
                async with asyncio.timeout(turn_timeout):
                    return await self._call_policy(budget)
            except TimeoutError as e:
                # Note: if the *outer* rollout timeout fires during sampling,
                # the cancellation passes through here as CancelledError (this
                # inner timeout did not expire), so it is not misclassified.
                raise SamplingTurnTimeoutError(
                    f"Sampling call exceeded sampling_turn_timeout_seconds={turn_timeout}"
                ) from e

    async def _call_policy(self, budget: int | None) -> TokensWithLogprobs:
        # Only pass the kwarg when a budget applies, so unconfigured rollouts
        # keep working with TokenCompleter implementations whose __call__ does
        # not accept a max_tokens kwarg.
        if budget is None:
            return await self.policy(self.ob, self.stop_condition)
        return await self.policy(self.ob, self.stop_condition, max_tokens=budget)

    # ------------------------------------------------------------------
    # Hook plumbing and env seams
    # ------------------------------------------------------------------

    def _state(self) -> RolloutState:
        return RolloutState(
            turn_index=self.turn_index,
            observation_tokens=self.ob.length,
            sampled_tokens=self.sampled_tokens,
            num_length_truncations=self.num_truncations,
            elapsed_seconds=self._elapsed(),
            transitions=self.transitions,
        )

    async def _resolve_stop(self, reason: str) -> str | None:
        """Pass a runner-imposed stop through on_stop_reason. None = cleared."""
        if self.hooks is None:
            return reason
        resolved = await self.hooks.on_stop_reason(str(reason), self._state())
        return None if resolved is None else str(resolved)

    async def _inject_messages(
        self, messages: list[Message]
    ) -> tuple[tinker.ModelInput, StopCondition]:
        inject = getattr(self.env, "inject_messages", None)
        if inject is None:
            raise TypeError(
                "on_turn_begin returned messages to inject, but the environment "
                f"({type(self.env).__name__}) does not support message injection "
                "(no inject_messages method)"
            )
        return await inject(list(messages))

    def _push_tool_call_limit(self) -> None:
        if self.limits is None or self.limits.max_tool_calls is None:
            return
        setter = getattr(self.env, "set_max_tool_calls", None)
        if setter is None:
            logger.warning(
                "RolloutLimits.max_tool_calls=%d is configured, but the environment "
                "(%s) does not support a tool-call limit (no set_max_tool_calls "
                "method); the limit will not be enforced.",
                self.limits.max_tool_calls,
                type(self.env).__name__,
            )
            return
        setter(self.limits.max_tool_calls)

    def _push_parse_error_policy(self) -> None:
        if self.parse_errors is None:
            return
        setter = getattr(self.env, "set_parse_error_policy", None)
        if setter is None:
            logger.warning(
                "A ParseErrorPolicy is configured, but the environment (%s) does "
                "not support one (no set_parse_error_policy method); the default "
                "one-shot parse-failure semantics (failed_parse_reward / "
                "terminate_on_parse_error) will apply.",
                type(self.env).__name__,
            )
            return
        setter(self.parse_errors)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_stop_metrics(transition: Transition) -> None:
        for key in [k for k in transition.metrics if k.startswith(STOP_METRIC_PREFIX)]:
            del transition.metrics[key]

    def _attach_truncation_metric(self) -> None:
        if self.num_truncations > 0 and self.transitions:
            self.transitions[-1].metrics[TRUNCATION_METRIC_KEY] = float(self.num_truncations)

    async def _run_end_hooks(self, *, graded: bool) -> None:
        if self.hooks is None:
            return
        state = self._state()
        if graded:
            await self.hooks.on_grade(state)
        await self.hooks.on_artifacts(state)

    async def _finalize_initial_overflow(self, initial: InitialObservationOverflow) -> Trajectory:
        # The initial prompt already exceeds the env's token budget: end the
        # rollout gracefully with an empty-episode trajectory — one synthetic
        # transition (empty ob/ac, episode_done) carrying the overflow reward
        # and stop/<reason> metric. It contributes no training tokens
        # (trajectory_to_data emits no datum for it) but its reward still
        # counts toward the group-mean baseline used for advantages; the
        # group survives.
        transition = Transition(
            ob=tinker.ModelInput.empty(),
            ac=TokensWithLogprobs(tokens=[], maybe_logprobs=[]),
            reward=initial.reward,
            episode_done=True,
            metrics=initial.metrics,
            logs=initial.logs,
        )
        self.transitions.append(transition)
        trajectory = Trajectory(
            transitions=self.transitions,
            final_ob=tinker.ModelInput.empty(),
            stop_reason=stop_reason_from_metrics(transition.metrics),
        )
        await self._run_end_hooks(graded=False)
        return trajectory

    async def _finalize_runner_stop(self, reason: str) -> Trajectory:
        """Finalize a runner-imposed stop (budgets, max_turns, timeouts).

        The env never observed the episode ending, so no grading ran; the last
        recorded transition is marked terminal and carries the stop metric.
        """
        if self.transitions:
            last = self.transitions[-1]
            metrics = {
                k: v for k, v in last.metrics.items() if not k.startswith(STOP_METRIC_PREFIX)
            }
            metrics[f"{STOP_METRIC_PREFIX}{reason}"] = 1.0
            self.transitions[-1] = replace(last, episode_done=True, metrics=metrics)
        else:
            # Stopped before any turn completed: synthesize an empty terminal
            # transition (same shape as the initial-overflow path) to carry
            # the stop metric. No training tokens.
            self.transitions.append(
                Transition(
                    ob=tinker.ModelInput.empty(),
                    ac=TokensWithLogprobs(tokens=[], maybe_logprobs=[]),
                    reward=0.0,
                    episode_done=True,
                    metrics={f"{STOP_METRIC_PREFIX}{reason}": 1.0},
                )
            )
        self._attach_truncation_metric()
        trajectory = Trajectory(
            transitions=self.transitions,
            final_ob=self.ob,
            stop_reason=str(reason),
        )
        await self._run_end_hooks(graded=False)
        return trajectory

    async def _finalize_env_stop(self, reason: str | None) -> Trajectory:
        self._attach_truncation_metric()
        trajectory = Trajectory(
            transitions=self.transitions,
            final_ob=self.ob,
            stop_reason=reason,
        )
        await self._run_end_hooks(graded=True)
        return trajectory
