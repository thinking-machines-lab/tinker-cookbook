#!/usr/bin/env python3
"""
End-to-end test script for tool calling across different model families.

This script queries production models with tool-calling prompts and verifies
that tool calls are correctly rendered, generated, and parsed.

NOT a unit test - requires API access and queries real models.

Usage:
    uv run python tinker_cookbook/scripts/test_tool_calling_e2e.py [--model MODEL_NAME]
"""

import argparse
import asyncio

import tinker

from tinker_cookbook.renderers import (
    Message,
    ToolSpec,
    get_renderer,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


# Sample tool specifications
SAMPLE_TOOLS: list[ToolSpec] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'San Francisco'"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
]


# Model configurations for testing
MODEL_CONFIGS = [
    {
        "model_name": "Qwen/Qwen3-8B",
        "renderer_name": "qwen3",
    },
    {
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "renderer_name": "qwen3_instruct",
    },
    {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "renderer_name": "llama3",
    },
    {
        "model_name": "deepseek-ai/DeepSeek-V3.1",
        "renderer_name": "deepseekv3",
    },
    {
        "model_name": "moonshotai/Kimi-K2-Thinking",
        "renderer_name": "kimi_k2",
    },
    {
        "model_name": "openai/gpt-oss-20b",
        "renderer_name": "gpt_oss_medium_reasoning",
    },
]


def print_result(
    model_name: str,
    success: bool,
    message: Message,
    raw_response: str,
):
    """Print formatted test result."""
    status = "✓" if success else "✗"
    print(f"\n{'=' * 60}")
    print(f"{status} {model_name}")
    print(f"{'=' * 60}")

    if "tool_calls" in message and message["tool_calls"]:
        print(f"Tool calls found: {len(message['tool_calls'])}")
        for i, tc in enumerate(message["tool_calls"]):
            print(f"  [{i}] {tc.function.name}({tc.function.arguments})")
    else:
        print("No tool calls found")

    print(f"\nContent: {message.get('content', '')[:200]}...")
    print(f"\nRaw response (first 500 chars):\n{raw_response[:500]}")


async def test_model(
    service_client: tinker.ServiceClient,
    model_name: str,
    renderer_name: str,
    tools: list[ToolSpec],
    system_prompt: str,
    user_prompt: str,
) -> tuple[bool, Message, str]:
    """
    Test tool calling for a single model.

    Returns:
        Tuple of (success, parsed_message, raw_response)
    """
    print(f"\nTesting {model_name}...")

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Build messages using the unified interface
    prefix_messages = renderer.create_conversation_prefix_with_tools(tools, system_prompt)
    messages: list[Message] = prefix_messages + [{"role": "user", "content": user_prompt}]

    # Build prompt
    prompt = renderer.build_generation_prompt(messages)
    stop_sequences = renderer.get_stop_sequences()

    # Create sampling client
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            stop=stop_sequences,
            max_tokens=512,
            temperature=0.0,  # Deterministic for testing
        ),
    )

    # Parse response
    response_tokens = result.sequences[0].tokens
    raw_response = tokenizer.decode(response_tokens)
    message, parse_success = renderer.parse_response(response_tokens)

    # Check if we got tool calls
    has_tool_calls = "tool_calls" in message and len(message.get("tool_calls", [])) > 0
    success = parse_success and has_tool_calls

    return success, message, raw_response


async def main():
    parser = argparse.ArgumentParser(description="Test tool calling across models")
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to test (default: test all)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What's the weather like in San Francisco?",
        help="User prompt to send",
    )
    args = parser.parse_args()

    # Filter models if specific one requested
    configs = MODEL_CONFIGS
    if args.model:
        configs = [c for c in configs if args.model in c["model_name"]]
        if not configs:
            print(f"No matching model found for: {args.model}")
            print(f"Available models: {[c['model_name'] for c in MODEL_CONFIGS]}")
            return

    print("=" * 60)
    print("Tool Calling End-to-End Test")
    print("=" * 60)
    print(f"User prompt: {args.prompt}")
    print(f"Models to test: {[c['model_name'] for c in configs]}")

    # Create service client (shared across all model tests)
    service_client = tinker.ServiceClient()

    system_prompt = "You are a helpful assistant."
    results = []
    for config in configs:
        try:
            success, message, raw_response = await test_model(
                service_client=service_client,
                model_name=config["model_name"],
                renderer_name=config["renderer_name"],
                tools=SAMPLE_TOOLS,
                system_prompt=system_prompt,
                user_prompt=args.prompt,
            )
            print_result(config["model_name"], success, message, raw_response)
            results.append((config["model_name"], success))
        except Exception as e:
            print(f"\n✗ {config['model_name']}: Error - {e}")
            results.append((config["model_name"], False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    for model, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {model}")


if __name__ == "__main__":
    asyncio.run(main())
