"""End-to-end smoke test for the LiteLLM Tinker provider.

Requires TINKER_API_KEY to be set (skipped otherwise, see conftest.py).
"""

import litellm
import pytest

from tinker_cookbook.third_party.litellm import register_tinker_provider

# Use a small model for fast smoke testing
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture(scope="module")
def tinker_provider():
    provider = register_tinker_provider()
    yield provider
    # Clean up the registration
    litellm.custom_provider_map[:] = [
        entry for entry in litellm.custom_provider_map if entry["custom_handler"] is not provider
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acompletion_basic(tinker_provider) -> None:
    """Basic async completion returns a valid response with tokens."""
    response = await litellm.acompletion(
        model="tinker/test",
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        base_model=BASE_MODEL,
        temperature=0.0,
        max_tokens=32,
    )

    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.message.content is not None
    assert len(choice.message.content) > 0
    assert choice.finish_reason in ("stop", "length")

    # Verify raw tokens are accessible
    fields = choice.message.provider_specific_fields
    assert fields is not None
    assert isinstance(fields["prompt_token_ids"], list)
    assert isinstance(fields["completion_token_ids"], list)
    assert len(fields["prompt_token_ids"]) > 0
    assert len(fields["completion_token_ids"]) > 0

    # Usage should be populated
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acompletion_with_system_message(tinker_provider) -> None:
    """System messages are handled correctly."""
    response = await litellm.acompletion(
        model="tinker/test",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "Say hello."},
        ],
        base_model=BASE_MODEL,
        temperature=0.0,
        max_tokens=32,
    )

    assert response.choices[0].message.content is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acompletion_multi_turn(tinker_provider) -> None:
    """Multi-turn conversations work."""
    response = await litellm.acompletion(
        model="tinker/test",
        messages=[
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! How can I help you?"},
            {"role": "user", "content": "What is my name?"},
        ],
        base_model=BASE_MODEL,
        temperature=0.0,
        max_tokens=32,
    )

    assert response.choices[0].message.content is not None


@pytest.mark.integration
def test_completion_sync(tinker_provider) -> None:
    """Sync completion also works."""
    response = litellm.completion(
        model="tinker/test",
        messages=[{"role": "user", "content": "Say hi."}],
        base_model=BASE_MODEL,
        temperature=0.0,
        max_tokens=16,
    )

    assert response.choices[0].message.content is not None
    fields = response.choices[0].message.provider_specific_fields
    assert fields is not None
    assert len(fields["completion_token_ids"]) > 0
