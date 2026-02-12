"""Shared fixtures for taubench tests."""

import pytest


@pytest.fixture
def sample_system_prompt():
    """Sample system prompt with ask_sonnet instructions."""
    return (
        "You are a helpful customer service agent."
        "\n\nIMPORTANT: You have access to a special tool called `ask_sonnet` "
        "that delegates the current turn to a more capable AI assistant (Claude Sonnet). "
        "Use this tool when:\n- You are unsure how to proceed with a complex request\n"
        "- You need help understanding the customer's needs\n"
        "- You want to verify your approach before taking an action\n"
        "- The task requires careful reasoning or nuanced judgment\n\n"
        "When you call `ask_sonnet`, Claude Sonnet will see the full conversation "
        "and respond on your behalf. Use this tool liberally when uncertain - "
        "it's better to ask for help than to make mistakes.\n\n"
        "NOTE: Always greet the customer yourself on the first turn. "
        "Do not use `ask_sonnet` for the initial greeting - handle it directly, "
        "then use `ask_sonnet` for subsequent turns if needed."
    )


@pytest.fixture
def sample_tools():
    """Sample tool definitions in OpenAI function format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Look up order details by order ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order ID"}
                    },
                    "required": ["order_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_sonnet",
                "description": "Delegate this turn to Claude Sonnet.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I need help with my order"},
        {"role": "assistant", "content": "I'd be happy to help! What's your order number?"},
        {"role": "user", "content": "Order #12345"},
    ]
