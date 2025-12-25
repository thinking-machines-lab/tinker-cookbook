"""
Tests for consistency between build_generation_prompt and build_supervised_example.

When you call build_supervised_example with train_on_what=ALL_ASSISTANT_MESSAGES,
the tokens with weight=0 are observations and weight=1 are actions (assistant responses).

This test verifies that for each assistant message:
- The observation tokens (weight=0 prefix before that action) match what you get
  from build_generation_prompt called on the conversation prefix up to that assistant message.

This consistency is important because:
1. During training, we use build_supervised_example to create training data
2. During inference, we use build_generation_prompt to create prompts
3. If these don't match, the model sees different token sequences at train vs inference time
"""

from functools import cache
from typing import Any

import pytest
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.model_info import get_model_attributes, get_recommended_renderer_name
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat, get_renderer
from tinker_cookbook.tokenizer_utils import Tokenizer


@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    """Get tokenizer with chat template."""
    kwargs: dict[str, Any] = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)


def split_supervised_into_ob_ac_pairs(
    tokens: list[int], weights: torch.Tensor
) -> list[tuple[list[int], list[int]]]:
    """
    Split a supervised example into (observation, action) pairs based on weight transitions.

    The weights tensor has 0 for observation tokens and 1 for action tokens.
    When train_on_what=ALL_ASSISTANT_MESSAGES, the pattern looks like:
        000001111100001111
    where each block of 1s is a separate assistant action.

    Returns:
        List of (observation_tokens, action_tokens) tuples.
        Each observation is ALL tokens from the beginning up to the start of that action.
        Each action is a contiguous block of tokens with weight=1.

    Important: The observation for each action includes the FULL prefix from the beginning,
    not just the tokens since the last action. This matches what build_generation_prompt
    returns - the full conversation history up to that point.
    """
    assert len(tokens) == len(weights), f"Token count {len(tokens)} != weight count {len(weights)}"

    pairs: list[tuple[list[int], list[int]]] = []
    weights_list = weights.tolist()

    # Find all contiguous blocks of 1s (actions)
    i = 0

    while i < len(weights_list):
        if weights_list[i] == 1:
            # Found start of an action block
            action_start = i

            # Find end of action block
            while i < len(weights_list) and weights_list[i] == 1:
                i += 1
            action_end = i

            # Observation is EVERYTHING from the beginning up to this action
            observation = tokens[:action_start]
            action = tokens[action_start:action_end]
            pairs.append((observation, action))
        else:
            i += 1

    return pairs


def get_generation_prompts_for_assistant_messages(
    messages: list[Message], renderer: Renderer
) -> list[list[int]]:
    """
    For each assistant message in the conversation, get the generation prompt
    for the prefix up to (but not including) that assistant message.

    Returns:
        List of token sequences, one for each assistant message.
    """
    prompts: list[list[int]] = []

    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            # Get prefix up to this assistant message
            prefix = messages[:i]
            prompt = renderer.build_generation_prompt(prefix)
            prompts.append(prompt.to_ints())

    return prompts


# =============================================================================
# Test Conversation Definitions
# =============================================================================


def get_single_turn_conversation() -> list[Message]:
    """Single user->assistant exchange."""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]


def get_multi_turn_conversation() -> list[Message]:
    """Multi-turn conversation with 2 assistant responses."""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]


def get_three_turn_conversation() -> list[Message]:
    """Three assistant responses."""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome!"},
    ]


def get_system_message_conversation() -> list[Message]:
    """Conversation with system message."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]


CONVERSATIONS = [
    ("single_turn", get_single_turn_conversation),
    ("multi_turn", get_multi_turn_conversation),
    ("three_turn", get_three_turn_conversation),
    ("with_system", get_system_message_conversation),
]

# Models to test
MODELS_TO_TEST = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "moonshotai/Kimi-K2-Thinking",
]

# Known inconsistencies between build_generation_prompt and build_supervised_example.
# These expose real renderer-level inconsistencies.
#
# Note: Qwen3-VL-Instruct PASSES because it uses Qwen3VLInstructRenderer which
# doesn't add <think> blocks (it's not a thinking model).
KNOWN_INCONSISTENCIES = {
    # Qwen3 (thinking models): In build_supervised_example, render_message is called with
    # is_last=True for the last assistant message, which adds <think>\n to the observation.
    # But in build_generation_prompt, the partial assistant message is rendered with
    # is_last=False (default), so <think>\n is NOT added.
    ("Qwen/Qwen3-30B-A3B", "single_turn"): (
        "Qwen3 adds <think>\\n to observation in supervised (is_last=True) but not in generation prompt (is_last=False)"
    ),
    ("Qwen/Qwen3-30B-A3B", "multi_turn"): (
        "Qwen3 adds <think>\\n to observation in supervised (is_last=True) but not in generation prompt (is_last=False)"
    ),
    ("Qwen/Qwen3-30B-A3B", "three_turn"): (
        "Qwen3 adds <think>\\n to observation in supervised (is_last=True) but not in generation prompt (is_last=False)"
    ),
    ("Qwen/Qwen3-30B-A3B", "with_system"): (
        "Qwen3 adds <think>\\n to observation in supervised (is_last=True) but not in generation prompt (is_last=False)"
    ),
    # DeepSeek: In build_generation_prompt, the partial assistant message renders </think>
    # as part of the prompt (in render_message, ac_str starts with "</think>").
    # But in build_supervised_example, </think> is part of the action (weight=1), not observation.
    ("deepseek-ai/DeepSeek-V3.1", "single_turn"): (
        "DeepSeek includes </think> in the generation prompt but treats it as action in supervised example"
    ),
    ("deepseek-ai/DeepSeek-V3.1", "multi_turn"): (
        "DeepSeek includes </think> in the generation prompt but treats it as action in supervised example"
    ),
    ("deepseek-ai/DeepSeek-V3.1", "three_turn"): (
        "DeepSeek includes </think> in the generation prompt but treats it as action in supervised example"
    ),
    ("deepseek-ai/DeepSeek-V3.1", "with_system"): (
        "DeepSeek includes </think> in the generation prompt but treats it as action in supervised example"
    ),
}


def _format_token_comparison(
    tokenizer: Tokenizer,
    expected: list[int],
    actual: list[int],
    context: str,
) -> str:
    """Format a helpful comparison message for token mismatches."""
    # Find first difference
    first_diff = -1
    for i in range(min(len(expected), len(actual))):
        if expected[i] != actual[i]:
            first_diff = i
            break
    if first_diff == -1 and len(expected) != len(actual):
        first_diff = min(len(expected), len(actual))

    msg = f"\n{'='*60}\n{context}\n{'='*60}\n"
    msg += f"Expected length: {len(expected)}, Actual length: {len(actual)}\n"

    if first_diff >= 0:
        msg += f"First difference at position {first_diff}\n\n"

        # Show context around the difference
        start = max(0, first_diff - 5)
        end_expected = min(len(expected), first_diff + 10)
        end_actual = min(len(actual), first_diff + 10)

        msg += f"Expected tokens [{start}:{end_expected}]: {expected[start:end_expected]}\n"
        msg += f"Actual tokens   [{start}:{end_actual}]: {actual[start:end_actual]}\n\n"

        msg += f"Expected decoded: {repr(tokenizer.decode(expected))}\n"
        msg += f"Actual decoded:   {repr(tokenizer.decode(actual))}\n"
    else:
        msg += "Tokens are identical\n"

    return msg


# =============================================================================
# The Main Consistency Test
# =============================================================================


@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
@pytest.mark.parametrize("conv_name,conv_factory", CONVERSATIONS)
def test_generation_supervised_consistency(
    model_name: str, conv_name: str, conv_factory: callable
):
    """
    Test that observations from build_supervised_example match build_generation_prompt.

    Uses RAW conversations without any preparation/modification. This exposes real
    renderer-level inconsistencies.

    For each assistant message in the conversation:
    1. Get the observation tokens from build_supervised_example (tokens with weight=0 before that action)
    2. Get the generation prompt from build_generation_prompt (prefix up to that assistant message)
    3. Verify they match

    Known inconsistencies:
    - Qwen3: Adds <think>\\n to observation in supervised (is_last=True) but not
      in generation prompt (is_last=False in build_generation_prompt)
    - DeepSeek: Includes </think> in generation prompt but treats it as action
      in supervised example
    """
    known_issue = KNOWN_INCONSISTENCIES.get((model_name, conv_name))

    tokenizer = get_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer, image_processor)

    # Use RAW conversation - no preparation!
    convo = conv_factory()

    # Get supervised example with ALL_ASSISTANT_MESSAGES
    model_input, weights = renderer.build_supervised_example(
        convo, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    tokens = model_input.to_ints()

    # Split into (observation, action) pairs
    ob_ac_pairs = split_supervised_into_ob_ac_pairs(tokens, weights)

    # Get generation prompts for each assistant message
    gen_prompts = get_generation_prompts_for_assistant_messages(convo, renderer)

    # Count assistant messages
    num_assistant_messages = sum(1 for m in convo if m["role"] == "assistant")

    # Verify we got the right number of pairs
    assert len(ob_ac_pairs) == num_assistant_messages, (
        f"Expected {num_assistant_messages} (ob, ac) pairs from supervised example, "
        f"but got {len(ob_ac_pairs)}. Model: {model_name}, Conv: {conv_name}"
    )

    assert len(gen_prompts) == num_assistant_messages, (
        f"Expected {num_assistant_messages} generation prompts, "
        f"but got {len(gen_prompts)}. Model: {model_name}, Conv: {conv_name}"
    )

    # Check each observation matches the corresponding generation prompt
    mismatches = []
    for i, ((ob, ac), gen_prompt) in enumerate(zip(ob_ac_pairs, gen_prompts)):
        if ob != gen_prompt:
            context = (
                f"Model: {model_name}, Conv: {conv_name}, Assistant message {i+1}/{num_assistant_messages}"
            )
            mismatch_msg = _format_token_comparison(tokenizer, gen_prompt, ob, context)
            mismatches.append(mismatch_msg)

    if mismatches:
        full_msg = f"\n\nFound {len(mismatches)} mismatch(es) for {model_name} / {conv_name}:\n"
        full_msg += "\n".join(mismatches)

        if known_issue:
            pytest.xfail(f"Known issue: {known_issue}\n{full_msg}")
        else:
            pytest.fail(full_msg)


# =============================================================================
# Additional diagnostic test to see the actual ob/ac structure
# =============================================================================


@pytest.mark.parametrize("model_name", MODELS_TO_TEST[:2])  # Just test a couple
def test_show_ob_ac_structure(model_name: str):
    """
    Diagnostic test to visualize the observation/action structure.
    This test always passes but prints useful debugging info.
    """
    tokenizer = get_tokenizer(model_name)
    attributes = get_model_attributes(model_name)
    image_processor = get_image_processor(model_name) if attributes.is_vl else None
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer, image_processor)

    convo = get_multi_turn_conversation()

    model_input, weights = renderer.build_supervised_example(
        convo, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    tokens = model_input.to_ints()

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Renderer: {renderer_name}")
    print(f"{'='*60}")

    # Show weight pattern
    weight_str = "".join(str(int(w)) for w in weights.tolist())
    print(f"Weight pattern ({len(weight_str)} tokens): {weight_str}")

    # Split and show each pair
    ob_ac_pairs = split_supervised_into_ob_ac_pairs(tokens, weights)
    print(f"\nFound {len(ob_ac_pairs)} (observation, action) pairs:")

    for i, (ob, ac) in enumerate(ob_ac_pairs):
        print(f"\n--- Pair {i+1} ---")
        print(f"Observation ({len(ob)} tokens): {repr(tokenizer.decode(ob)[:100])}...")
        print(f"Action ({len(ac)} tokens): {repr(tokenizer.decode(ac))}")

    # Also show generation prompts
    gen_prompts = get_generation_prompts_for_assistant_messages(convo, renderer)
    print(f"\nGeneration prompts ({len(gen_prompts)} total):")
    for i, prompt in enumerate(gen_prompts):
        print(f"\n--- Gen Prompt {i+1} ---")
        print(f"({len(prompt)} tokens): {repr(tokenizer.decode(prompt)[:100])}...")
