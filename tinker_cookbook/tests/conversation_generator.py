"""
Seeded random conversation generator for renderer testing.

Usage:
    convo = generate_conversation(seed=42)

    @pytest.mark.parametrize("seed", QUICK_TEST_SEEDS)
    def test_renderer(seed):
        convo = generate_conversation(seed)
"""

import json
import random
import uuid

from tinker_cookbook.renderers.base import (
    ContentPart,
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
)


def _rand_str(rng: random.Random, length: int = 8) -> str:
    """Generate a random hex string."""
    return uuid.UUID(int=rng.getrandbits(128)).hex[:length]


def _rand_tool_call(rng: random.Random) -> ToolCall:
    """Generate a random tool call."""
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=f"tool_{_rand_str(rng, 6)}",
            arguments=json.dumps({"arg": _rand_str(rng)}),
        ),
        id=f"call_{_rand_str(rng)}",
    )


def generate_conversation(
    seed: int,
    *,
    include_system: bool = True,
    include_tool_calls: bool = True,
    include_thinking: bool = True,
    min_turns: int = 2,
    max_turns: int = 10,
    end_with_assistant: bool = True,
) -> list[Message]:
    """
    Generate a random conversation.

    Args:
        seed: Random seed for reproducibility.
        include_system: Whether to potentially include a system message.
        include_tool_calls: Whether to potentially include tool calls.
        include_thinking: Whether to potentially include thinking parts.
        min_turns: Minimum number of user-assistant turn pairs.
        max_turns: Maximum number of user-assistant turn pairs.
        end_with_assistant: Whether to end with assistant or user message.

    Returns:
        A list of Message objects.
    """
    rng = random.Random(seed)
    messages: list[Message] = []

    # Maybe add system message
    if include_system and rng.random() < 0.5:
        messages.append(Message(role="system", content=f"system_{_rand_str(rng)}"))

    num_turns = rng.randint(min_turns, max_turns)

    for turn in range(num_turns):
        is_last_turn = turn == num_turns - 1

        # User message
        messages.append(Message(role="user", content=f"user_{_rand_str(rng)}"))

        if is_last_turn and not end_with_assistant:
            break

        # Assistant message
        has_thinking = include_thinking and rng.random() < 0.5
        has_tool_call = include_tool_calls and rng.random() < 0.3

        # Build content
        if has_thinking:
            content: str | list[ContentPart] = [
                ThinkingPart(type="thinking", thinking=f"think_{_rand_str(rng)}"),
                TextPart(type="text", text=f"asst_{_rand_str(rng)}"),
            ]
        else:
            content = f"asst_{_rand_str(rng)}"

        if has_tool_call:
            tool_call = _rand_tool_call(rng)
            messages.append(Message(role="assistant", content=content, tool_calls=[tool_call]))
            # Tool response
            assert tool_call.id is not None  # Always set by _rand_tool_call
            messages.append(
                Message(
                    role="tool",
                    content=json.dumps({"result": _rand_str(rng)}),
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                )
            )
            # Follow-up assistant
            messages.append(Message(role="assistant", content=f"followup_{_rand_str(rng)}"))
        else:
            messages.append(Message(role="assistant", content=content))

    return messages


def generate_simple_conversation(seed: int, end_with_assistant: bool = True) -> list[Message]:
    """No tools or thinking."""
    return generate_conversation(
        seed,
        include_tool_calls=False,
        include_thinking=False,
        end_with_assistant=end_with_assistant,
    )
