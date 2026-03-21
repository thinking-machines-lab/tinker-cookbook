"""
LiteLLM custom provider for Tinker sampling.

Enables using Tinker's native SamplingClient through LiteLLM's unified interface,
giving optimal sampling performance while exposing raw token IDs for training.

Usage::

    from tinker_cookbook.third_party.litellm import register_litellm_provider
    import litellm

    register_litellm_provider()

    response = await litellm.acompletion(
        model="tinker/my-model",
        messages=[{"role": "user", "content": "Hello!"}],
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )

    # Access raw tokens for training
    fields = response.choices[0].message.provider_specific_fields
    prompt_tokens = fields["prompt_token_ids"]
    completion_tokens = fields["completion_token_ids"]
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import httpx
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.third_party.openai_compat import (
    SamplingResult,
    extract_sampling_params,
    sample_chat_completion,
    sampling_result_to_openai_dict,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

try:
    from litellm.llms.custom_llm import CustomLLM
    from litellm.types.utils import Choices, Message, ModelResponse, Usage
except ImportError:
    raise ImportError(
        "litellm is required for the Tinker LiteLLM integration. "
        "Install it with: uv pip install -e '.[litellm]'"
    ) from None


def _build_model_response(
    result: SamplingResult,
    model_response: ModelResponse,
) -> ModelResponse:
    """Populate a LiteLLM ModelResponse from a SamplingResult."""
    completion_dict = sampling_result_to_openai_dict(result)

    choice_data = completion_dict["choices"][0]
    message_data = choice_data["message"]

    model_response.choices = [
        Choices(
            finish_reason=choice_data["finish_reason"],
            index=0,
            message=Message(
                content=message_data.get("content"),
                role="assistant",
                tool_calls=message_data.get("tool_calls"),
                provider_specific_fields={
                    "prompt_token_ids": result.prompt_token_ids,
                    "completion_token_ids": result.completion_token_ids,
                },
            ),
        )
    ]

    usage_data = completion_dict["usage"]
    model_response.usage = Usage(  # type: ignore[assignment]
        prompt_tokens=usage_data["prompt_tokens"],
        completion_tokens=usage_data["completion_tokens"],
        total_tokens=usage_data["total_tokens"],
    )
    model_response.model = result.model_name

    return model_response


def _map_tinker_error(exc: Exception) -> Exception:
    """Map Tinker SDK exceptions to LiteLLM-compatible errors."""
    import litellm.exceptions

    if isinstance(exc, tinker.AuthenticationError):
        return litellm.exceptions.AuthenticationError(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    if isinstance(exc, tinker.RateLimitError):
        return litellm.exceptions.RateLimitError(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    if isinstance(exc, tinker.APITimeoutError):
        return litellm.exceptions.Timeout(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    if isinstance(exc, tinker.APIConnectionError):
        return litellm.exceptions.APIConnectionError(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    if isinstance(exc, tinker.BadRequestError):
        return litellm.exceptions.BadRequestError(
            message=str(exc),
            llm_provider="tinker",
            model="",
        )
    # Fallback: re-raise as-is
    return exc


# ---------------------------------------------------------------------------
# Client bundle and provider
# ---------------------------------------------------------------------------


@dataclass
class _ClientBundle:
    """Cached group of objects needed to sample from a specific model."""

    sampling_client: tinker.SamplingClient
    renderer: renderers.Renderer
    tokenizer: Tokenizer
    base_model: str


class TinkerLiteLLMProvider(CustomLLM):
    """LiteLLM custom provider that routes calls through Tinker's native SamplingClient."""

    def __init__(
        self,
        service_client: tinker.ServiceClient | None = None,
    ) -> None:
        super().__init__()
        self._clients: dict[str, _ClientBundle] = {}
        self._service_client = service_client

    def _get_service_client(self) -> tinker.ServiceClient:
        if self._service_client is None:
            self._service_client = tinker.ServiceClient()
        return self._service_client

    def _get_or_create_client(self, base_model: str) -> _ClientBundle:
        """Get or lazily create a client bundle for the given base model."""
        if base_model not in self._clients:
            tokenizer = get_tokenizer(base_model)
            renderer_name = get_recommended_renderer_name(base_model)
            renderer = renderers.get_renderer(renderer_name, tokenizer)
            sampling_client = self._get_service_client().create_sampling_client(
                base_model=base_model
            )
            self._clients[base_model] = _ClientBundle(
                sampling_client=sampling_client,
                renderer=renderer,
                tokenizer=tokenizer,
                base_model=base_model,
            )
        return self._clients[base_model]

    def set_client(
        self,
        sampling_client: tinker.SamplingClient,
    ) -> None:
        """Inject a custom SamplingClient (e.g., for a fine-tuned checkpoint).

        The base model is read from the client via ``get_base_model()``,
        and used to resolve the correct renderer and tokenizer. If a bundle
        for that base model already exists, only the sampling client is replaced.
        """
        base_model = sampling_client.get_base_model()
        if base_model in self._clients:
            self._clients[base_model].sampling_client = sampling_client
        else:
            tokenizer = get_tokenizer(base_model)
            renderer_name = get_recommended_renderer_name(base_model)
            renderer = renderers.get_renderer(renderer_name, tokenizer)
            self._clients[base_model] = _ClientBundle(
                sampling_client=sampling_client,
                renderer=renderer,
                tokenizer=tokenizer,
                base_model=base_model,
            )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},  # noqa: B006
        timeout: Union[float, httpx.Timeout] | None = None,
        client=None,
    ) -> ModelResponse:
        base_model: str = (litellm_params or {}).get("base_model", "")
        if not base_model:
            raise ValueError(
                "base_model is required for the Tinker provider. "
                "Pass it as: litellm.acompletion(..., base_model='Qwen/Qwen3-4B-Instruct-2507')"
            )

        bundle = self._get_or_create_client(base_model)
        sampling_params = extract_sampling_params(optional_params)

        try:
            result = await sample_chat_completion(
                sampling_client=bundle.sampling_client,
                renderer=bundle.renderer,
                messages=messages,
                tools=optional_params.get("tools"),
                model_name=model,
                **sampling_params,
            )
        except tinker.TinkerError as exc:
            raise _map_tinker_error(exc) from exc

        return _build_model_response(result, model_response)

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},  # noqa: B006
        timeout: Union[float, httpx.Timeout] | None = None,
        client=None,
    ) -> ModelResponse:
        base_model: str = (litellm_params or {}).get("base_model", "")
        if not base_model:
            raise ValueError(
                "base_model is required for the Tinker provider. "
                "Pass it as: litellm.completion(..., base_model='Qwen/Qwen3-4B-Instruct-2507')"
            )

        bundle = self._get_or_create_client(base_model)
        sampling_params = extract_sampling_params(optional_params)

        coro = sample_chat_completion(
            sampling_client=bundle.sampling_client,
            renderer=bundle.renderer,
            messages=messages,
            tools=optional_params.get("tools"),
            model_name=model,
            **sampling_params,
        )

        try:
            # If there's already a running event loop (e.g., Jupyter), use it.
            # Otherwise, create a new one.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            else:
                result = asyncio.run(coro)
        except tinker.TinkerError as exc:
            raise _map_tinker_error(exc) from exc

        return _build_model_response(result, model_response)


_registered_provider: TinkerLiteLLMProvider | None = None


def register_litellm_provider(
    *,
    service_client: tinker.ServiceClient | None = None,
) -> TinkerLiteLLMProvider:
    """Register the Tinker provider with LiteLLM.

    Safe to call multiple times — returns the same provider instance after
    the first call. Use the returned instance to inject custom SamplingClients
    via ``provider.set_client(sampling_client)``.

    Args:
        service_client: Optional pre-configured ``tinker.ServiceClient``.
            Useful for custom deployments with a non-default ``base_url``.
            If None, a default ``ServiceClient`` is created on first use.
            Ignored on subsequent calls (singleton already exists).
    """
    import litellm

    global _registered_provider
    if _registered_provider is not None:
        return _registered_provider

    provider = TinkerLiteLLMProvider(service_client=service_client)
    litellm.custom_provider_map.append({"provider": "tinker", "custom_handler": provider})
    _registered_provider = provider
    return provider
