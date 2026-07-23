"""Rollout configuration presets.

:class:`RolloutConfig` bundles the per-rollout policy objects
(:class:`~tinker_cookbook.rl.rollout_limits.RolloutLimits`,
:class:`~tinker_cookbook.rl.rollout_limits.ParseErrorPolicy`,
:class:`~tinker_cookbook.rl.rollout_limits.TerminationRewardPolicy`) plus the
tool execution mode into one config object, so recipes configure rollout
behavior with a single value instead of threading four knobs separately.

Two presets are provided:

- :func:`simple` — the minimal configuration: no limits, one-shot
  parse-failure semantics, rewards untouched, parallel tool execution.
  Identical to passing no configuration at all (pinned by
  ``rollout_semantics_regression_test.py``).
- :func:`agentic` — an opinionated preset for multi-turn tool-use RL: turn /
  trajectory-token / tool-call budgets, parse-error retries, rewards clamped
  to <= 0 for limit-stopped trajectories (``zero_reward_on_limit``), bounded
  grading, and sequential tool execution.  See its docstring for the
  reasoning behind each choice.

Consume a config in three places:

- ``build_agent_tool_env(..., rollout_config=cfg)`` applies the env-side
  pieces (parse policy, tool-call cap, tool execution mode, termination
  policy on the grader, LENGTH-continue when token budgets are configured).
- ``run_rollout(policy, env, config=cfg)`` applies the runner-side pieces
  (``cfg.limits`` and ``cfg.parse_errors``).
- ``do_group_rollout(..., termination=cfg.termination)`` applies the reward
  clamp / skip-grading semantics after group reward computation.
"""

from typing import Literal, TypeAlias

import chz

from tinker_cookbook.rl.rollout_limits import (
    ParseErrorPolicy,
    RolloutLimits,
    TerminationRewardPolicy,
)
from tinker_cookbook.rl.rollout_strategy import FailFast, MinViableGroup, RolloutStrategy

ToolExecution: TypeAlias = Literal["sequential", "parallel", "concurrent_safe"]
"""How an env executes a turn's batch of tool calls.

- ``"parallel"`` (default): all calls dispatched concurrently via
  ``asyncio.gather``.
- ``"sequential"``: calls awaited one at a time, in the order the model
  requested them.  Required when tools share mutable state (a filesystem, a
  shell session) and the model's call order matters.
- ``"concurrent_safe"``: reserved for running concurrency-safe tools in
  parallel and the rest sequentially.  The public tool contract
  (:class:`~tinker_cookbook.tool_use.types.Tool`) has no safety marker yet,
  so this mode currently raises ``NotImplementedError``.
"""


@chz.chz
class RolloutConfig:
    """One config object bundling all rollout policy knobs.

    Every field defaults to the unconfigured behavior, so
    ``RolloutConfig()`` (== :func:`simple`) changes nothing.  See the module
    docstring for where each piece is consumed.
    """

    limits: RolloutLimits = chz.field(
        default_factory=RolloutLimits,
        doc="Per-rollout budgets (turns, tokens, tool calls, wall clock), enforced by the rollout runner. Default: all unlimited.",
    )
    parse_errors: ParseErrorPolicy | None = chz.field(
        default=None,
        doc="Parse-failure handling (structural stop, content retries). Default None: one-shot parse-failure semantics (failed_parse_reward / terminate_on_parse_error).",
    )
    termination: TerminationRewardPolicy | None = chz.field(
        default=None,
        doc="Stop-reason-keyed reward semantics: rewards clamped to <= 0 for limit-stopped trajectories (zero_reward_on_limit), grader timeout/scope. Default None: rewards untouched.",
    )
    tool_execution: ToolExecution = chz.field(
        default="parallel",
        doc="How a turn's tool calls execute: 'parallel' (default, asyncio.gather), 'sequential' (in-order awaits), or 'concurrent_safe' (reserved; currently raises NotImplementedError).",
    )


def simple() -> RolloutConfig:
    """The minimal rollout configuration.

    No budgets, one-shot parse-failure semantics, rewards untouched by stop
    reasons, parallel tool execution.  Identical to passing no configuration
    at all (pinned by ``rollout_semantics_regression_test.py``), so
    ``simple()`` vs ``agentic()`` on the same recipe is a controlled
    comparison.
    """
    return RolloutConfig()


def agentic() -> RolloutConfig:
    """An opinionated preset for multi-turn tool-use RL.

    Where :func:`simple` leaves every behavior at its minimal default, this
    preset takes a position on each failure mode an agentic rollout can hit:

    - **Budgets** (10 turns, 65536 trajectory tokens, 30 tool calls): agent
      rollouts can have a long tail: trajectories that loop or ramble would
      otherwise dominate wall-clock and token spend.
      Budgets convert that tail into recorded stop reasons instead of
      stragglers.  Truncation alone doesn't end the episode; only a
      cumulative budget does, so one long turn isn't fatal.
    - **Parse errors** (up to 2 consecutive content failures retried with a
      corrective message): a deployed agent harness answers an unparsable
      tool call with an error and lets the model try again, so training with
      bounded retries matches what the model will experience and teaches
      recovery rather than "one bad turn ends the episode".  Structural
      failures (broken message framing) stop immediately — the conversation
      state is corrupted and there is nothing faithful to retry on.
    - **Rewards** (grade normally, then clamp limit-stopped trajectories to
      ``min(reward, 0.0)``): a trajectory that hit a limit should never be
      *rewarded* — otherwise the model can learn that blowing the budget is
      fine as long as the task eventually succeeded — but a flat penalty
      would erase the grader's ordering among failures.  Clamping keeps the
      negative-side ordering (a near-miss scores better than a disaster)
      while capping the positive side at zero.  Timed-out rollouts skip
      grading entirely (the reward would be clamped to zero anyway, so
      grading is wasted work); grading is bounded at 900 seconds so a hung
      grader cannot stall the batch.
    - **Tools** (sequential execution, preserving the model's call order):
      correct for tools that share mutable state (a filesystem, a shell,
      a browser session), at some latency cost.  Use ``"parallel"`` when
      all tools are read-only and independent.

    All of these compose with the same machinery :func:`simple` uses, so
    running one recipe under both presets is a controlled comparison of the
    rollout policy itself.
    """
    return RolloutConfig(
        limits=RolloutLimits(
            max_turns=10,
            max_trajectory_tokens=65536,
            max_tool_calls=30,
        ),
        parse_errors=ParseErrorPolicy(max_consecutive=2),
        termination=TerminationRewardPolicy(
            zero_reward_on_limit=True,
            skip_grading_on_timeout=True,
            grader_timeout_seconds=900,
        ),
        tool_execution="sequential",
    )


def default_rollout_config_for_model(model_name: str) -> RolloutConfig:
    """The rollout configuration a model gets when none is specified.

    ``thinkingmachines/Inkling`` (including its ``:peft:`` long-context
    variants) defaults to :func:`agentic` — it is a tool-use model and its
    recommended renderer (``tml_v0``) is built for multi-turn agents.  Every
    other model defaults to :func:`simple`.

    Pass an explicit ``rollout_config`` (e.g. ``simple()``) anywhere a
    config is accepted to override the model default.
    """
    base_model = model_name.split(":", 1)[0]
    if base_model == "thinkingmachines/Inkling":
        return agentic()
    return simple()


def default_rollout_strategy_for_model(model_name: str) -> RolloutStrategy:
    """The group rollout strategy a model gets when none is specified.

    ``thinkingmachines/Inkling`` defaults to
    :class:`~tinker_cookbook.rl.rollout_strategy.MinViableGroup` (agentic
    rollouts run against real tool backends, where occasional infrastructure
    failures shouldn't discard whole groups); every other model defaults to
    :class:`~tinker_cookbook.rl.rollout_strategy.FailFast`.

    Set ``Config.rollout_error_tolerance`` explicitly (``False``, ``True``,
    or a strategy instance) to override the model default.
    """
    base_model = model_name.split(":", 1)[0]
    if base_model == "thinkingmachines/Inkling":
        return MinViableGroup()
    return FailFast()
