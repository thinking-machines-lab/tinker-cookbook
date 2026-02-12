"""Tests for ExternalLLMClient — external LLM calls with retry logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from tinker_cookbook.recipes.taubench.components.types import ExternalLLMConfig
from tinker_cookbook.recipes.taubench.components.external_llm import (
    ExternalLLMClient,
    LLMCallResult,
    is_retryable_error,
)


def _make_litellm_exception(cls, message="test"):
    """Create a litellm exception with required args."""
    # litellm exceptions require specific constructor args
    try:
        return cls(message)
    except TypeError:
        try:
            return cls(message=message, model="test", llm_provider="test")
        except TypeError:
            return cls(message, model="test", llm_provider="test")


class TestIsRetryableError:
    """Test error classification for retry logic."""

    def test_connection_error_retryable(self):
        exc = _make_litellm_exception(litellm.APIConnectionError, "connection reset")
        assert is_retryable_error(exc)

    def test_rate_limit_retryable(self):
        exc = _make_litellm_exception(litellm.RateLimitError, "too many requests")
        assert is_retryable_error(exc)

    def test_server_error_retryable(self):
        exc = _make_litellm_exception(litellm.InternalServerError, "internal error")
        assert is_retryable_error(exc)

    def test_bad_gateway_retryable(self):
        # BadGatewayError might not exist in all litellm versions, fall back to string check
        if hasattr(litellm, "BadGatewayError"):
            exc = _make_litellm_exception(litellm.BadGatewayError, "502 bad gateway")
            assert is_retryable_error(exc)
        else:
            exc = Exception("502 bad gateway")
            assert is_retryable_error(exc)

    def test_credit_balance_retryable(self):
        exc = Exception("Your credit balance is too low")
        assert is_retryable_error(exc)

    def test_503_in_message_retryable(self):
        exc = Exception("503 service unavailable")
        assert is_retryable_error(exc)

    def test_overloaded_retryable(self):
        exc = Exception("API is overloaded")
        assert is_retryable_error(exc)

    def test_timeout_retryable(self):
        if hasattr(litellm, "Timeout"):
            exc = _make_litellm_exception(litellm.Timeout, "request timed out")
            assert is_retryable_error(exc)

    def test_billing_retryable(self):
        exc = Exception("billing issue detected")
        assert is_retryable_error(exc)

    def test_upstream_connect_error_retryable(self):
        exc = Exception("upstream connect error or disconnect/reset before headers")
        assert is_retryable_error(exc)

    def test_temporarily_unavailable_retryable(self):
        exc = Exception("service temporarily unavailable")
        assert is_retryable_error(exc)

    def test_generic_error_not_retryable(self):
        exc = ValueError("bad input")
        assert is_retryable_error(exc) is False

    def test_auth_error_not_retryable(self):
        exc = Exception("authentication failed")
        assert is_retryable_error(exc) is False


class TestLLMCallResult:
    def test_construction(self):
        r = LLMCallResult(content="hello", input_tokens=10, output_tokens=5)
        assert r.content == "hello"
        assert r.input_tokens == 10
        assert r.output_tokens == 5


class TestExternalLLMClient:
    def _make_client(self) -> ExternalLLMClient:
        return ExternalLLMClient(
            ExternalLLMConfig(model="test-model", temperature=0.0, max_tokens=100)
        )

    @pytest.mark.asyncio
    async def test_call_returns_content(self):
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response text"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20

        with patch(
            "tinker_cookbook.recipes.taubench.components.external_llm.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.call([{"role": "user", "content": "hi"}])
            assert result == "response text"

    @pytest.mark.asyncio
    async def test_call_with_usage_returns_tokens(self):
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "answer"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 30

        with patch(
            "tinker_cookbook.recipes.taubench.components.external_llm.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.call_with_usage([{"role": "user", "content": "q"}])
            assert isinstance(result, LLMCallResult)
            assert result.content == "answer"
            assert result.input_tokens == 100
            assert result.output_tokens == 30

    @pytest.mark.asyncio
    async def test_call_with_none_content_returns_empty_string(self):
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 0

        with patch(
            "tinker_cookbook.recipes.taubench.components.external_llm.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.call_with_usage([{"role": "user", "content": "q"}])
            assert result.content == ""

    @pytest.mark.asyncio
    async def test_retryable_error_is_retried(self):
        """Verify that a retryable error triggers at least one retry before succeeding."""
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "success after retry"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        # First call raises retryable error, second call succeeds
        mock_acompletion = AsyncMock(
            side_effect=[Exception("503 service unavailable"), mock_response]
        )

        with patch(
            "tinker_cookbook.recipes.taubench.components.external_llm.litellm.acompletion",
            mock_acompletion,
        ):
            result = await client.call_with_usage([{"role": "user", "content": "q"}])
            assert result.content == "success after retry"
            assert mock_acompletion.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self):
        """Non-retryable errors should propagate without retry."""
        client = self._make_client()

        mock_acompletion = AsyncMock(
            side_effect=ValueError("bad input - not retryable")
        )

        with patch(
            "tinker_cookbook.recipes.taubench.components.external_llm.litellm.acompletion",
            mock_acompletion,
        ):
            with pytest.raises(ValueError, match="bad input"):
                await client.call_with_usage([{"role": "user", "content": "q"}])
            assert mock_acompletion.call_count == 1  # No retry
