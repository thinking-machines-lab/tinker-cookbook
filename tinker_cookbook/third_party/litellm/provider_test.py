"""Tests for the LiteLLM integration."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class FakeSampledSequence:
    tokens: list[int]
    logprobs: list[float] | None
    stop_reason: str = "stop"


@dataclass
class FakeSampleResponse:
    sequences: list[FakeSampledSequence]


# ---------------------------------------------------------------------------
# LiteLLM provider
# ---------------------------------------------------------------------------


class TestTinkerLiteLLMProvider:
    def test_register_adds_to_provider_map(self) -> None:
        import litellm

        import tinker_cookbook.third_party.litellm.provider as provider_mod
        from tinker_cookbook.third_party.litellm import register_litellm_provider

        # Reset the singleton so we can test fresh registration
        old_registered = provider_mod._registered_provider
        provider_mod._registered_provider = None

        provider = None
        try:
            original_len = len(litellm.custom_provider_map)
            provider = register_litellm_provider()
            assert len(litellm.custom_provider_map) == original_len + 1
            entry = litellm.custom_provider_map[-1]
            assert entry["provider"] == "tinker"
            assert entry["custom_handler"] is provider

            # Calling again returns the same instance without adding a duplicate
            provider2 = register_litellm_provider()
            assert provider2 is provider
            assert len(litellm.custom_provider_map) == original_len + 1
        finally:
            # Clean up
            if provider is not None:
                litellm.custom_provider_map[:] = [
                    e
                    for e in litellm.custom_provider_map
                    if e.get("custom_handler") is not provider
                ]
            provider_mod._registered_provider = old_registered

    def test_set_client_creates_bundle(self) -> None:
        from tinker_cookbook.third_party.litellm.provider import TinkerLiteLLMProvider

        provider = TinkerLiteLLMProvider()
        mock_client = MagicMock()
        mock_client.get_base_model.return_value = "Qwen/Qwen3-8B"

        with (
            patch("tinker_cookbook.third_party.litellm.provider.get_tokenizer") as mock_get_tok,
            patch(
                "tinker_cookbook.third_party.litellm.provider.get_recommended_renderer_name",
                return_value="qwen3",
            ),
            patch("tinker_cookbook.third_party.litellm.provider.renderers.get_renderer"),
        ):
            mock_get_tok.return_value = MagicMock()
            provider.set_client(mock_client)

        assert "Qwen/Qwen3-8B" in provider._clients
        assert provider._clients["Qwen/Qwen3-8B"].sampling_client is mock_client

    def test_set_client_updates_existing_bundle(self) -> None:
        from tinker_cookbook.third_party.litellm.provider import (
            TinkerLiteLLMProvider,
            _ClientBundle,
        )

        provider = TinkerLiteLLMProvider()
        old_client = MagicMock()
        new_client = MagicMock()
        new_client.get_base_model.return_value = "Qwen/Qwen3-8B"

        provider._clients["Qwen/Qwen3-8B"] = _ClientBundle(
            sampling_client=old_client,
            renderer=MagicMock(),
            tokenizer=MagicMock(),
            base_model="Qwen/Qwen3-8B",
        )

        provider.set_client(new_client)
        assert provider._clients["Qwen/Qwen3-8B"].sampling_client is new_client

    @pytest.mark.asyncio
    async def test_acompletion_requires_base_model(self) -> None:
        from tinker_cookbook.third_party.litellm.provider import TinkerLiteLLMProvider

        provider = TinkerLiteLLMProvider()
        model_response = MagicMock()

        with pytest.raises(ValueError, match="base_model is required"):
            await provider.acompletion(
                model="tinker/test",
                messages=[],
                api_base="",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=print,
                encoding=None,
                api_key=None,
                logging_obj=MagicMock(),
                optional_params={},
                litellm_params={},
            )

    @pytest.mark.asyncio
    async def test_acompletion_basic(self) -> None:
        from tinker_cookbook.third_party.litellm.provider import (
            TinkerLiteLLMProvider,
            _ClientBundle,
        )

        provider = TinkerLiteLLMProvider()

        fake_response = FakeSampleResponse(
            sequences=[FakeSampledSequence(tokens=[10, 20], logprobs=[0.1, 0.2])]
        )
        mock_sampling_client = MagicMock()
        mock_sampling_client.sample_async = AsyncMock(return_value=fake_response)

        mock_renderer = MagicMock()
        mock_renderer.build_generation_prompt.return_value = MagicMock()
        mock_renderer.build_generation_prompt.return_value.to_ints.return_value = [1, 2, 3]
        mock_renderer.get_stop_sequences.return_value = ["<|end|>"]
        mock_renderer.parse_response.return_value = (
            {"role": "assistant", "content": "Hello!"},
            True,
        )

        provider._clients["Qwen/Qwen3-8B"] = _ClientBundle(
            sampling_client=mock_sampling_client,
            renderer=mock_renderer,
            tokenizer=MagicMock(),
            base_model="Qwen/Qwen3-8B",
        )

        model_response = MagicMock()

        result = await provider.acompletion(
            model="tinker/my-model",
            messages=[{"role": "user", "content": "Hi"}],
            api_base="",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=print,
            encoding=None,
            api_key=None,
            logging_obj=MagicMock(),
            optional_params={"temperature": 0.7, "max_tokens": 64},
            litellm_params={"base_model": "Qwen/Qwen3-8B"},
        )

        assert result is model_response
        # Verify the response was populated
        fields = result.choices[0].message.provider_specific_fields
        assert fields is not None
        assert fields["prompt_token_ids"] == [1, 2, 3]
        assert fields["completion_token_ids"] == [10, 20]
