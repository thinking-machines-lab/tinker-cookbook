"""Regression tests for opt-in preservation of sampled tokens in tool rollouts."""

import asyncio
from unittest.mock import patch

import pytest
import tinker
from tinker.types import ImageChunk

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.renderers.base import Message, RenderContext, RenderedMessage
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.rl.data_processing import _flatten_chunks, _is_prefix, trajectory_to_data
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use.agent_tool_message_env import build_agent_tool_env
from tinker_cookbook.tool_use.tools import FunctionTool, simple_tool_result
from tinker_cookbook.tool_use.types import ToolResult

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


@pytest.mark.parametrize("fallback", ["injection", "truncation", "parse_retry", "structural_error"])
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
        else:
            env.terminate_on_parse_error = False
            step = await env.step(renderer.tokenizer.encode("broken", add_special_tokens=False))
            assert not step.episode_done
            assert step.next_observation.length == 0
            # No valid observation remains after this legacy error path. The next
            # successful turn must fully render before incremental appends resume.
            step = await env.step(action)
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
        env._current_observation = original
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
