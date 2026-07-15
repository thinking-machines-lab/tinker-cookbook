"""Tests for the rollout presets and the composite RolloutConfig surface.

Covers exact preset values (``agentic``), the all-defaults ``simple`` preset,
and the composite config driving a full fake rollout end-to-end through
``build_agent_tool_env`` + ``run_rollout`` + ``do_group_rollout``.
"""

from __future__ import annotations

import asyncio
import pickle
from collections.abc import Sequence
from typing import Any, cast

import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
)
from tinker_cookbook.rl.rollout_limits import (
    ParseErrorPolicy,
    RolloutLimits,
    TerminationRewardPolicy,
)
from tinker_cookbook.rl.rollout_presets import (
    RolloutConfig,
    agentic,
    default_rollout_config_for_model,
    default_rollout_strategy_for_model,
    simple,
)
from tinker_cookbook.rl.rollout_runner import run_rollout
from tinker_cookbook.rl.rollout_strategy import (
    FailFast,
    MinViableGroup,
    RolloutResult,
    RolloutStrategy,
)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, Trajectory
from tinker_cookbook.tool_use.agent_tool_message_env import (
    AgentToolMessageEnv,
    build_agent_tool_env,
)
from tinker_cookbook.tool_use.tools import simple_tool_result
from tinker_cookbook.tool_use.types import ToolInput, ToolResult

# ---------------------------------------------------------------------------
# Preset construction
# ---------------------------------------------------------------------------


class TestSimplePreset:
    def test_simple_is_all_defaults(self):
        cfg = simple()
        assert cfg == RolloutConfig()
        assert cfg.limits == RolloutLimits()  # all limits None (unlimited)
        assert cfg.limits.max_turns is None
        assert cfg.limits.max_trajectory_tokens is None
        assert cfg.limits.max_sampled_tokens is None
        assert cfg.limits.max_tool_calls is None
        assert cfg.limits.max_turn_tokens is None
        assert cfg.limits.rollout_timeout_seconds is None
        assert cfg.limits.sampling_turn_timeout_seconds is None
        assert cfg.parse_errors is None
        assert cfg.termination is None
        assert cfg.tool_execution == "parallel"

    def test_simple_env_matches_unconfigured_env(self):
        """build_agent_tool_env with simple() resolves to the same env
        settings as passing no config at all (the regression-locked
        defaults)."""
        renderer = _ScriptedRenderer(parse_results=[])
        kwargs: dict[str, Any] = {
            "renderer": cast(Renderer, renderer),
            "tools": [],
            "initial_messages": INITIAL_MESSAGES,
            "reward_fn": _RecordingReward(),
        }
        unconfigured = build_agent_tool_env(**kwargs)
        configured = build_agent_tool_env(rollout_config=simple(), **kwargs)

        for env in (unconfigured, configured):
            msg_env = cast(AgentToolMessageEnv, env.message_env)
            assert msg_env.max_turns == 5
            assert msg_env.max_tool_calls is None
            assert msg_env.parse_error_policy is None
            assert msg_env.tool_execution == "parallel"
            assert msg_env.termination_policy is None
            assert env.terminate_on_length is True
            assert env.parse_error_policy is None


class TestAgenticPreset:
    def test_agentic_values_exact(self):
        cfg = agentic()
        assert cfg.limits == RolloutLimits(
            max_turns=10,
            max_trajectory_tokens=65536,
            max_tool_calls=30,
        )
        assert cfg.limits.max_sampled_tokens is None
        assert cfg.limits.max_turn_tokens is None
        assert cfg.limits.rollout_timeout_seconds is None
        assert cfg.parse_errors == ParseErrorPolicy(max_consecutive=2)
        assert cfg.termination == TerminationRewardPolicy(
            zero_reward_on_limit=True,
            skip_grading_on_timeout=True,
            grader_timeout_seconds=900,
        )
        assert cfg.tool_execution == "sequential"

    def test_pickle_round_trip(self):
        cfg = agentic()
        assert pickle.loads(pickle.dumps(cfg)) == cfg


# ---------------------------------------------------------------------------
# Scripted fakes for the end-to-end test
# ---------------------------------------------------------------------------

INITIAL_MESSAGES: list[Message] = [{"role": "user", "content": "please do the task"}]


class _ScriptedCompleter(TokenCompleter):
    def __init__(self, results: list[TokensWithLogprobs]):
        self._results = list(results)
        self.calls = 0

    async def __call__(
        self, model_input: tinker.ModelInput, stop: Any, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        self.calls += 1
        return self._results.pop(0)


class _ScriptedRenderer:
    """Fixed tokens per message; parse_response replays a script."""

    def __init__(
        self,
        parse_results: list[tuple[Message, ParseTermination]],
        tokens_per_message: int = 10,
    ):
        self._parse_results = list(parse_results)
        self.tokens_per_message = tokens_per_message

    def get_stop_sequences(self) -> list[str]:
        return ["<stop>"]

    def build_generation_prompt(self, messages: list[Message], **kwargs: Any) -> tinker.ModelInput:
        return tinker.ModelInput.from_ints(list(range(self.tokens_per_message * len(messages))))

    def parse_response(self, action: list[int]) -> tuple[Message, ParseTermination]:
        return self._parse_results.pop(0)


class _RecordingReward:
    def __init__(self, reward: float = 0.9):
        self.calls = 0
        self.reward = reward
        self.received: list[Message] | None = None

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        self.calls += 1
        self.received = list(history)
        return self.reward, {"graded": 1.0}


class _EchoTool:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo tool"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def to_spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input: ToolInput) -> ToolResult:
        self.calls += 1
        return simple_tool_result("echoed", call_id=input.call_id or "", name=self.name)


def _tokens(tokens: list[int]) -> TokensWithLogprobs:
    return TokensWithLogprobs(tokens=tokens, maybe_logprobs=[0.0] * len(tokens))


def _tool_call_message() -> Message:
    return {
        "role": "assistant",
        "content": "calling a tool",
        "tool_calls": [
            ToolCall(id="call_1", function=ToolCall.FunctionBody(name="echo", arguments="{}"))
        ],
    }


def _unparsed_message() -> Message:
    return {
        "role": "assistant",
        "content": "broken tool call",
        "unparsed_tool_calls": [
            UnparsedToolCall(raw_text='{"name": "echo", bad json}', error="Invalid JSON")
        ],
    }


class _PrebuiltStrategy(RolloutStrategy):
    def __init__(self, trajectories: list[Trajectory], envs: list[Env]):
        self._trajectories = trajectories
        self._envs = envs

    async def execute(
        self, env_group_builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> RolloutResult:
        return RolloutResult(trajectories=self._trajectories, envs=self._envs, errors=[])


class _NoEnvsBuilder(EnvGroupBuilder):
    async def make_envs(self) -> Sequence[Env]:
        return []


# ---------------------------------------------------------------------------
# End-to-end: agentic() through build_agent_tool_env + run_rollout +
# do_group_rollout
# ---------------------------------------------------------------------------


class TestAgenticEndToEnd:
    def test_agentic_config_drives_a_full_fake_rollout(self):
        """The agentic preset flows through the whole pipeline as ONE object:

        - env side (``build_agent_tool_env(rollout_config=...)``): turn cap
          10, tool-call cap 30, sequential tools, parse-error retries,
          termination policy on the grader, LENGTH-continue;
        - runner side (``run_rollout(config=...)``): budgets + parse policy;
        - reward side (``do_group_rollout(termination=...)``): the positive
          graded reward of the turn-capped trajectory is clamped to 0.0.

        Script: turns 1-3 tool calls; turns 4-5 content parse errors (each
        retried with an injected corrective message per max_consecutive=2);
        turns 6-10 tool calls; turn 10 hits the max_turns=10 budget, so the
        episode is graded (reward 0.9) and stopped with stop_reason
        'max_turns'.
        """
        cfg = agentic()

        parse_results: list[tuple[Message, ParseTermination]] = (
            [(_tool_call_message(), ParseTermination.STOP_SEQUENCE)] * 3
            + [(_unparsed_message(), ParseTermination.STOP_SEQUENCE)] * 2
            + [(_tool_call_message(), ParseTermination.STOP_SEQUENCE)] * 5
        )
        renderer = _ScriptedRenderer(parse_results=parse_results)
        tool = _EchoTool()
        reward_fn = _RecordingReward(reward=0.9)

        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[tool],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=reward_fn,
            rollout_config=cfg,
        )

        # The composite config resolved the env-side knobs.
        msg_env = cast(AgentToolMessageEnv, env.message_env)
        assert msg_env.max_turns == 10  # from cfg.limits.max_turns
        assert msg_env.max_tool_calls == 30
        assert msg_env.tool_execution == "sequential"
        assert msg_env.termination_policy == cfg.termination
        assert msg_env.parse_error_policy == cfg.parse_errors
        assert env.terminate_on_length is False  # runner owns LENGTH/budgets

        policy = _ScriptedCompleter([_tokens([1, 2]) for _ in range(10)])
        traj = asyncio.run(run_rollout(policy, env, config=cfg))

        # Ten turns sampled; the parse-error turns injected corrective
        # messages and continued (budgets + parse policy active).
        assert policy.calls == 10
        assert len(traj.transitions) == 10
        assert traj.stop_reason == "max_turns"
        retry_messages = [
            m
            for m in msg_env.history
            if m["role"] == "user" and "formatting issue" in str(m.get("content", ""))
        ]
        assert len(retry_messages) == 2
        assert tool.calls == 8  # turns 1-3 and 6-10

        # Grading ran once (grade-then-clamp, not skip) over the completion
        # suffix (termination policy default: initial messages excluded).
        assert reward_fn.calls == 1
        assert traj.transitions[-1].reward == 0.9
        assert reward_fn.received is not None
        assert INITIAL_MESSAGES[0] not in reward_fn.received

        # Reward side: the limit-stopped trajectory's positive grade is
        # clamped to min(reward, 0.0) after compute_group_rewards.
        group = asyncio.run(
            do_group_rollout(
                _NoEnvsBuilder(),
                policy,
                strategy=_PrebuiltStrategy([traj], [env]),
                termination=cfg.termination,
            )
        )
        assert group.get_total_rewards() == [0.0]
        assert group.metrics_G[0]["zero_reward_on_limit"] == 1.0

    def test_agentic_trajectory_token_budget_active_through_runner(self):
        """The runner enforces cfg.limits.max_trajectory_tokens from the same
        composite config: an env whose conversation outgrows 65536 tokens
        stops with 'max_tokens' instead of sampling forever."""
        cfg = agentic()
        # 30000 tokens/message: initial obs 30000; after turn 1 the history is
        # [user, assistant, tool] = 90000 > 65536 - 1 -> budget stop.
        renderer = _ScriptedRenderer(
            parse_results=[(_tool_call_message(), ParseTermination.STOP_SEQUENCE)],
            tokens_per_message=30000,
        )
        reward_fn = _RecordingReward()
        env = build_agent_tool_env(
            renderer=cast(Renderer, renderer),
            tools=[_EchoTool()],
            initial_messages=INITIAL_MESSAGES,
            reward_fn=reward_fn,
            rollout_config=cfg,
        )
        policy = _ScriptedCompleter([_tokens([1, 2])])

        traj = asyncio.run(run_rollout(policy, env, config=cfg))

        assert policy.calls == 1
        assert traj.stop_reason == "max_tokens"
        assert reward_fn.calls == 0  # runner-imposed stop: env never grades


class TestModelDefaults:
    def test_inkling_defaults_to_agentic(self):
        cfg = default_rollout_config_for_model("thinkingmachines/Inkling")
        assert cfg == agentic()

    def test_inkling_peft_variants_default_to_agentic(self):
        for name in (
            "thinkingmachines/Inkling:peft:131072",
            "thinkingmachines/Inkling:peft:262144",
        ):
            assert default_rollout_config_for_model(name) == agentic()

    def test_other_models_default_to_simple(self):
        for name in ("Qwen/Qwen3.5-4B", "meta-llama/Llama-3.2-1B", "thinkingmachines/Model"):
            assert default_rollout_config_for_model(name) == simple()

    def test_inkling_strategy_is_min_viable_group(self):
        assert isinstance(
            default_rollout_strategy_for_model("thinkingmachines/Inkling"), MinViableGroup
        )
        assert isinstance(
            default_rollout_strategy_for_model("thinkingmachines/Inkling:peft:131072"),
            MinViableGroup,
        )

    def test_other_models_strategy_is_fail_fast(self):
        assert isinstance(default_rollout_strategy_for_model("Qwen/Qwen3.5-4B"), FailFast)


class TestTrainerDefaultResolution:
    def _config(self, model_name: str, **overrides: Any):
        from dataclasses import dataclass

        from tinker_cookbook.rl import train

        @dataclass
        class _Cfg:
            model_name: str
            termination: Any = None
            rollout_error_tolerance: Any = None

        return train, cast(Any, _Cfg(model_name=model_name, **overrides))

    def test_inkling_gets_agentic_termination_and_min_viable_group(self):
        train, cfg = self._config("thinkingmachines/Inkling")
        assert train.Config.effective_termination(cfg) == agentic().termination
        assert isinstance(train.Config.effective_rollout_strategy(cfg), MinViableGroup)

    def test_other_models_get_no_termination_and_fail_fast(self):
        train, cfg = self._config("Qwen/Qwen3.5-4B")
        assert train.Config.effective_termination(cfg) is None
        assert isinstance(train.Config.effective_rollout_strategy(cfg), FailFast)

    def test_explicit_values_win_over_model_default(self):
        explicit = TerminationRewardPolicy(zero_reward_on_limit=False)
        train, cfg = self._config(
            "thinkingmachines/Inkling",
            termination=explicit,
            rollout_error_tolerance=False,
        )
        assert train.Config.effective_termination(cfg) is explicit
        assert isinstance(train.Config.effective_rollout_strategy(cfg), FailFast)


class TestModelNameDefault:
    """build_agent_tool_env with no rollout_config resolves the model default."""

    def _build(self, model_name=None):
        return build_agent_tool_env(
            renderer=cast(Renderer, _ScriptedRenderer([])),
            tools=[],
            initial_messages=[{"role": "user", "content": "hi"}],
            reward_fn=_RecordingReward(),
            model_name=model_name,
        )

    def test_inkling_model_name_defaults_to_agentic(self):
        env = self._build(model_name="thinkingmachines/Inkling")
        expected = agentic()
        msg_env = cast(AgentToolMessageEnv, env.message_env)
        assert msg_env.max_turns == expected.limits.max_turns
        assert msg_env.parse_error_policy == expected.parse_errors
        assert msg_env.tool_execution == expected.tool_execution
        assert env.rollout_limits == expected.limits
        assert env.terminate_on_length is False

    def test_other_model_names_keep_documented_defaults(self):
        env = self._build(model_name="Qwen/Qwen3.5-4B")
        msg_env = cast(AgentToolMessageEnv, env.message_env)
        assert msg_env.max_turns == 5
        assert msg_env.parse_error_policy is None
        assert msg_env.tool_execution == "parallel"
        # simple()'s empty limits are advertised but behaviorally identical
        # to no limits: every runner check is per-field.
        assert env.rollout_limits == RolloutLimits()
        assert env.terminate_on_length is True

    def test_no_model_name_keeps_documented_defaults(self):
        env = self._build()
        msg_env = cast(AgentToolMessageEnv, env.message_env)
        assert msg_env.max_turns == 5
        assert env.rollout_limits is None
        assert env.terminate_on_length is True


class TestRunnerReadsEnvLimits:
    """run_rollout with no limits picks up the env-advertised rollout_limits."""

    def test_env_advertises_model_default_budgets(self):
        env = build_agent_tool_env(
            renderer=cast(Renderer, _ScriptedRenderer([])),
            tools=[],
            initial_messages=[{"role": "user", "content": "hi"}],
            reward_fn=_RecordingReward(),
            model_name="thinkingmachines/Inkling",
        )
        assert env.rollout_limits is not None
        assert env.rollout_limits.max_trajectory_tokens == 65536
