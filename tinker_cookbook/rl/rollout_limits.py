"""Trajectory-level rollout limits and policies.

:class:`RolloutLimits` is the declarative budget configuration for a rollout:
how many turns, tokens, tool calls, and seconds a single trajectory may
consume before it is stopped.  Every limit defaults to ``None`` (unlimited),
so a default-constructed ``RolloutLimits()`` changes nothing.

:class:`ParseErrorPolicy` configures how a rollout responds to parse failures
in the model's output, keyed off the renderer's structural-vs-content split
(:class:`~tinker_cookbook.renderers.base.ParseFailureKind`).  With no policy
configured, the default one-shot semantics (the first parse failure is
penalized once and, by default, ends the episode) apply unchanged.

:class:`TerminationRewardPolicy` configures how rewards interact with the stop
reason a trajectory ended with: whether limit-stopped trajectories have their
reward clamped to zero, whether grading is skipped for timed-out rollouts, how
long a grader may run, and what messages the grader sees.  With no policy
configured (``None``, the default everywhere), rewards flow through untouched.

Enforcement lives in the configured rollout runner (which recomputes the
per-turn token budget before every sampling call and passes it to the
completer via the per-call ``max_tokens`` override on
:meth:`~tinker_cookbook.completers.TokenCompleter.__call__`).  This module
only defines the configuration surface; constructing a ``RolloutLimits`` has
no effect until a runner consumes it.

Check ordering
--------------

The runner checks limits in a fixed order each turn:

1. **Pre-sample:** ``max_turns`` and ``rollout_timeout_seconds`` (between
   turns), then ``max_tool_calls`` *pre-dispatch* (before any tool call in
   the upcoming batch is issued).
2. **Sampling:** the per-turn token budget is the minimum of
   ``max_turn_tokens``, the remaining trajectory budget
   (``max_trajectory_tokens`` minus the current observation length), and the
   remaining sampled budget (``max_sampled_tokens`` minus tokens sampled so
   far).  ``sampling_turn_timeout_seconds`` bounds the sampling call itself.
3. **Post-sample:** a sampler ``"length"`` stop is handled first (LENGTH
   handling: the sampler stopped at the per-turn ``max_tokens`` cap,
   ``stop_reason == "length"``); the ``max_sampled_tokens`` check runs
   *after* LENGTH handling,
   so a truncated turn that also exhausts the sampled budget reports
   ``max_sampled_tokens``.
4. **Mid-batch:** ``max_tool_calls`` is re-checked between tool calls within
   a single turn's batch, so a turn requesting more calls than the remaining
   budget stops partway through the batch.
"""

from typing import Any

import chz

from tinker_cookbook.rl.types import StopReason


def _positive_or_none(obj: Any, name: str) -> None:
    """chz field validator: the value must be strictly positive or ``None``."""
    value = getattr(obj, name)
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be a positive value or None (unlimited), got {value!r}")


def _non_negative(obj: Any, name: str) -> None:
    """chz field validator: the value must be >= 0."""
    value = getattr(obj, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _contains_details_placeholder(obj: Any, name: str) -> None:
    """chz field validator: the template must contain the ``{details}`` placeholder."""
    value = getattr(obj, name)
    if "{details}" not in value:
        raise ValueError(f"{name} must contain the '{{details}}' placeholder, got {value!r}")


DEFAULT_RETRY_MESSAGE_TEMPLATE = "The previous message had a formatting issue.\nError: {details}"
"""Default corrective message injected after a recoverable (content) parse
failure.  ``{details}`` is replaced with the parse-error detail string."""


@chz.chz
class ParseErrorPolicy:
    """How a rollout responds to parse failures in the model's output.

    Built on the renderer contract's structural-vs-content split
    (:func:`~tinker_cookbook.renderers.base.classify_parse_failure`):

    - **Structural** failures (broken framing: the response never produced
      its stop signal) end the episode immediately with
      ``StopReason.PARSE_ERROR`` and ``terminal_reward``.  They are *never*
      retried: without the stop signal the message boundary is unknown, so
      the conversation state is corrupted and re-rendering it for another
      sampling call would put the model on a garbage observation.
    - **Content** failures (clean framing, unparsable tool calls — invalid
      JSON, unterminated tool blocks) are recoverable.  Up to
      ``max_consecutive`` consecutive content failures are retried: a
      corrective user message (``retry_message_template`` formatted with the
      error details) is injected and the rollout continues; the injected
      tokens count toward the trajectory token budget, and the error turn's
      reward is ``-penalty_per_error``.  Exceeding ``max_consecutive`` ends
      the episode with ``StopReason.PARSE_ERROR`` and ``terminal_reward``.  A
      turn that parses cleanly resets the consecutive counter.

    When a policy is configured it supersedes the default one-shot parse-error
    semantics (``failed_parse_reward`` / ``terminate_on_parse_error``) for
    parse-error turns.  With no policy (``None``, the default everywhere),
    those one-shot semantics apply unchanged.
    """

    max_consecutive: int = chz.field(
        default=0,
        validator=_non_negative,
        doc=(
            "Maximum consecutive content parse failures to retry with an injected "
            "corrective message. 0 (the default) means the first parse failure ends "
            "the episode. Structural failures always end the episode regardless."
        ),
    )
    retry_message_template: str = chz.field(
        default=DEFAULT_RETRY_MESSAGE_TEMPLATE,
        validator=_contains_details_placeholder,
        doc=(
            "Template for the corrective user message injected after a recoverable "
            "parse failure; '{details}' is replaced with the error detail (truncated "
            "to ~16KB). The injected tokens count toward the trajectory token budget."
        ),
    )
    penalty_per_error: float = chz.field(
        default=0.0,
        validator=_non_negative,
        doc=(
            "Penalty subtracted on each retried parse-error turn (that turn's reward "
            "is -penalty_per_error). Not applied on the terminal parse-error turn, "
            "which gets terminal_reward instead."
        ),
    )
    terminal_reward: float = chz.field(
        default=0.0,
        doc=(
            "Flat reward for the turn that ends the episode with "
            "StopReason.PARSE_ERROR (a structural failure, or content failures "
            "exceeding max_consecutive)."
        ),
    )
    mask_error_turns: bool = chz.field(
        default=False,
        doc=(
            "Mark parse-error turns (structural or content) with "
            "metrics['parse_error_masked']=1.0; trajectory_to_data then zeroes the "
            "action-token mask and advantages for those turns, so malformed samples "
            "are excluded from the training loss while the rest of the trajectory "
            "trains normally."
        ),
    )


@chz.chz
class RolloutLimits:
    """Budgets that bound a single rollout (one trajectory).

    All fields default to ``None``, meaning unlimited — a default-constructed
    ``RolloutLimits()`` changes nothing.  Each limit, when hit,
    ends the episode gracefully with the indicated
    :class:`~tinker_cookbook.rl.types.StopReason` (recorded as the
    ``stop/<reason>`` metric and :attr:`Trajectory.stop_reason`); no limit
    raises an exception, so the rest of the group is unaffected.

    See the module docstring for the exact ordering in which the limits are
    checked during a turn.
    """

    max_turns: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum number of agent turns (sampling calls) per rollout. "
            "Checked before each turn; reaching the cap produces StopReason.MAX_TURNS."
        ),
    )
    max_trajectory_tokens: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum total tokens in the trajectory, counting both observation "
            "(prompt) and action (sampled) tokens. Feeds the per-turn token budget "
            "as (max_trajectory_tokens - observation length); exhausting it produces "
            "StopReason.MAX_TOKENS. An initial prompt that already exceeds this "
            "budget ends the rollout immediately with StopReason.MAX_TOKENS."
        ),
    )
    max_sampled_tokens: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum cumulative sampled (action-only) tokens per rollout; "
            "observation tokens do not count. Checked after each turn, *after* "
            "LENGTH handling; exhausting it produces StopReason.MAX_SAMPLED_TOKENS "
            "(distinct from MAX_TOKENS, which covers the obs+action budget)."
        ),
    )
    max_tool_calls: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum tool calls per rollout. Checked pre-dispatch (before a turn's "
            "tool batch starts) and mid-batch (between calls within the batch); "
            "exhausting it produces StopReason.MAX_TOOL_CALLS."
        ),
    )
    max_turn_tokens: int | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Maximum sampled tokens for a single turn. A turn truncated at this "
            "cap (sampler stop_reason 'length') does not by itself end the rollout: "
            "the loop continues unless a cumulative budget (max_trajectory_tokens / "
            "max_sampled_tokens) is also exhausted, in which case the corresponding "
            "stop reason is reported."
        ),
    )
    rollout_timeout_seconds: float | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Wall-clock budget for the whole rollout. Checked between turns and "
            "enforced by an outer timeout around the loop; hitting it ends the "
            "episode gracefully with StopReason.ROLLOUT_TIMEOUT (the trajectory is "
            "kept and remains gradeable, unlike a strategy-level timeout which "
            "cancels and retries the rollout)."
        ),
    )
    sampling_turn_timeout_seconds: float | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Wall-clock budget for a single sampling call. Unlike the other limits "
            "this one raises (the sampling call failed; there is no partial turn to "
            "keep), composing with error-tolerant rollout strategies that retry."
        ),
    )


DEFAULT_LIMIT_STOP_REASONS: tuple[str, ...] = (
    StopReason.MAX_TOKENS,
    StopReason.MAX_SAMPLED_TOKENS,
    StopReason.MAX_TURNS,
    StopReason.MAX_TOOL_CALLS,
    StopReason.ROLLOUT_TIMEOUT,
)
"""The stop reasons treated as *limit* stops by default: the trajectory did
not end on its own terms, a configured budget cut it off."""


@chz.chz
class TerminationRewardPolicy:
    """How rewards interact with the stop reason a trajectory ended with.

    The core semantics is **grade normally, then clamp**: a limit-stopped
    trajectory (one whose :attr:`~tinker_cookbook.rl.types.Trajectory.stop_reason`
    is in ``limit_stop_reasons``) is still graded like any other, and its
    total reward is then clamped to ``min(reward, 0.0)``.  The model never
    profits from running into a budget, but informative *negative* rewards
    (and the group-relative comparison against members that finished) are
    preserved — unlike a flat truncation penalty, which erases the grader's
    signal entirely.

    Applied in :func:`~tinker_cookbook.rl.rollouts.do_group_rollout` after
    ``compute_group_rewards``, keyed off ``Trajectory.stop_reason``.  Only
    trajectories that carry a stop reason are affected; with no policy
    configured (``None``, the default everywhere), rewards flow through
    untouched.

    Grader-facing knobs (``grader_timeout_seconds``,
    ``pass_all_messages_to_grader``) take effect where grading actually runs:
    the group-level ``compute_group_rewards`` call in ``do_group_rollout``,
    and the episode-end ``reward_fn`` call inside
    :class:`~tinker_cookbook.tool_use.agent_tool_message_env.AgentToolMessageEnv`
    when the policy is installed on the env (e.g. via the ``rollout_config``
    parameter of ``build_agent_tool_env``).
    """

    zero_reward_on_limit: bool = chz.field(
        default=False,
        doc=(
            "When True, a trajectory whose stop_reason is in limit_stop_reasons "
            "has its total reward clamped to min(reward, 0.0) after grading. "
            "Grading still runs (grade-then-clamp); negative rewards pass through "
            "unchanged. Default False: no clamping."
        ),
    )
    limit_stop_reasons: tuple[str, ...] = chz.field(
        default=DEFAULT_LIMIT_STOP_REASONS,
        doc=(
            "Stop reasons treated as limit stops for zero_reward_on_limit. "
            "Defaults to the five budget-imposed reasons: max_tokens, "
            "max_sampled_tokens, max_turns, max_tool_calls, rollout_timeout."
        ),
    )
    skip_grading_on_timeout: bool = chz.field(
        default=False,
        doc=(
            "Skip the (possibly expensive) grader call and use 0.0 when the "
            "reward would be clamped to zero anyway. This requires a triple "
            "condition: (1) the trajectory stopped with rollout_timeout, "
            "(2) zero_reward_on_limit is True, and (3) rollout_timeout is in "
            "limit_stop_reasons — only then is the graded value guaranteed not "
            "to raise the reward above 0.0, so skipping cannot change the "
            "clamped outcome (beyond discarding a possibly-negative grade). "
            "See do_group_rollout for the exact application scope."
        ),
    )
    grader_timeout_seconds: float | None = chz.field(
        default=None,
        validator=_positive_or_none,
        doc=(
            "Wall-clock budget for a grading call, enforced with asyncio.timeout "
            "around (a) compute_group_rewards in do_group_rollout and (b) the "
            "env-level reward_fn call in AgentToolMessageEnv when this policy is "
            "installed on the env. A timeout raises TimeoutError (there is no "
            "usable reward), composing with error-tolerant rollout strategies. "
            "None (default) = unbounded."
        ),
    )
    pass_all_messages_to_grader: bool = chz.field(
        default=False,
        doc=(
            "What the env-level grader (AgentToolMessageEnv.reward_fn) sees when "
            "this policy is installed on the env: False (default) passes only the "
            "completion suffix — the messages generated during the rollout, "
            "excluding the initial prompt messages; True passes the full history. "
            "With no policy installed the default behavior (full history) is "
            "unchanged. Group-level graders always receive full trajectories."
        ),
    )
