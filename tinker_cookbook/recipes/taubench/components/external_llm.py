"""ExternalLLMClient - Handles external LLM (e.g., Sonnet) interactions."""

import logging
from dataclasses import dataclass

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from tinker_cookbook.recipes.taubench.components.types import ExternalLLMConfig

logger = logging.getLogger(__name__)


def is_retryable_error(exception: BaseException) -> bool:
    """Check if an exception is retryable."""
    if isinstance(exception, (litellm.APIConnectionError, litellm.InternalServerError, litellm.RateLimitError, litellm.BadGatewayError, litellm.Timeout)):
        return True
    error_str = str(exception).lower()
    if "credit balance" in error_str or "billing" in error_str:
        return True
    if any(pattern in error_str for pattern in [
        "503", "502", "service unavailable", "bad gateway",
        "upstream connect error", "connection termination", "reset before headers",
        "overloaded", "temporarily unavailable"
    ]):
        return True
    return False


@dataclass
class LLMCallResult:
    """Result from an LLM call including content and token usage."""
    content: str
    input_tokens: int
    output_tokens: int


class ExternalLLMClient:
    """Client for calling external LLM (e.g., Claude Sonnet)."""

    def __init__(self, config: ExternalLLMConfig):
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def call(self, messages: list[dict]) -> str:
        result = await self.call_with_usage(messages)
        return result.content

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        retry=retry_if_exception(is_retryable_error),
        before_sleep=lambda retry_state: logger.warning(
            "Retrying external LLM call (attempt %d) after error: %s",
            retry_state.attempt_number,
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        ),
    )
    async def call_with_usage(self, messages: list[dict]) -> LLMCallResult:
        logger.info(
            "Calling external LLM (%s) with %d messages",
            self.model,
            len(messages),
        )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            if is_retryable_error(e):
                logger.warning("Retryable error: %s", e)
                raise
            self._log_error(messages, e)
            raise

        content = response.choices[0].message.content or ""

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        logger.info(
            "External LLM (%s) response: %s (in=%d, out=%d tokens)",
            self.model,
            content[:100] + "..." if len(content) > 100 else content,
            input_tokens,
            output_tokens,
        )

        return LLMCallResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _log_error(self, messages: list[dict], error: Exception) -> None:
        print("\n" + "=" * 80)
        print("ERROR calling external LLM - dumping messages:")
        print("=" * 80)
        for i, msg in enumerate(messages):
            print(f"\n--- Message {i} ---")
            print(f"  role: {msg.get('role')}")
            content = msg.get('content', '')
            preview = repr(content[:200]) + "..." if len(content) > 200 else repr(content)
            print(f"  content: {preview}")
        print("=" * 80 + "\n")
        logger.error("External LLM call failed: %s", error)
