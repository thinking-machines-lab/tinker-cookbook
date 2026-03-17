"""Downstream compatibility tests for tinker_cookbook.renderers.

Validates that the renderer public API surface — types, registry functions,
factory, renderer method signatures, and built-in renderer names — remains
stable for downstream consumers.
"""

import inspect

import pytest

from tinker_cookbook import renderers
from tinker_cookbook.renderers import (
    ContentPart,
    DeepSeekV3ThinkingRenderer,
    GptOssRenderer,
    ImagePart,
    Message,
    MessageDelta,
    Qwen3Renderer,
    RenderContext,
    Renderer,
    Role,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    Utf8TokenDecoder,
    ensure_text,
    format_content_as_string,
    get_registered_renderer_names,
    get_text_content,
    is_renderer_registered,
    parse_content_blocks,
    register_renderer,
    unregister_renderer,
)
from tinker_cookbook.renderers.base import ensure_list
from tinker_cookbook.renderers.kimi_k2 import KimiK2Renderer
from tinker_cookbook.renderers.kimi_k25 import KimiK25Renderer

# ---------------------------------------------------------------------------
# Type exports
# ---------------------------------------------------------------------------


class TestTypeExports:
    """Verify that all types used by downstream are importable."""

    def test_message_is_typed_dict(self):
        # Message is used as a TypedDict / dict with role+content
        msg: Message = {"role": "user", "content": "hello"}
        assert msg["role"] == "user"
        assert msg["content"] == "hello"

    def test_text_part_constructable(self):
        part = TextPart(type="text", text="hello")
        assert part["text"] == "hello"

    def test_thinking_part_constructable(self):
        part = ThinkingPart(type="thinking", thinking="reasoning")
        assert part["thinking"] == "reasoning"

    def test_tool_call_constructable(self):
        tc = ToolCall(
            function=ToolCall.FunctionBody(
                name="my_tool",
                arguments='{"key": "value"}',
            ),
            id="call_123",
        )
        assert tc.function.name == "my_tool"
        assert tc.id == "call_123"

    def test_tool_spec_constructable(self):
        spec = ToolSpec(
            name="my_tool",
            description="A tool",
            parameters={"type": "object", "properties": {}},
        )
        assert spec["name"] == "my_tool"

    def test_train_on_what_has_expected_values(self):
        # Downstream uses at least LAST_ASSISTANT_MESSAGE
        assert hasattr(TrainOnWhat, "LAST_ASSISTANT_MESSAGE")

    def test_streaming_types_importable(self):
        # These are used by projects/tinker_chat
        assert StreamingMessageHeader is not None
        assert StreamingTextDelta is not None
        assert StreamingThinkingDelta is not None
        assert MessageDelta is not None

    def test_content_part_types(self):
        assert ContentPart is not None
        assert ImagePart is not None
        assert Role is not None

    def test_utf8_token_decoder_importable(self):
        assert Utf8TokenDecoder is not None

    def test_render_context_importable(self):
        assert RenderContext is not None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    def test_ensure_text_with_string(self):
        assert ensure_text("hello") == "hello"

    def test_format_content_as_string(self):
        result = format_content_as_string("hello")
        assert isinstance(result, str)

    def test_get_text_content(self):
        msg: Message = {"role": "user", "content": "hello"}
        assert get_text_content(msg) == "hello"

    def test_parse_content_blocks_exists(self):
        assert callable(parse_content_blocks)

    def test_ensure_list_importable(self):
        # Used by downstream rust extensions
        assert callable(ensure_list)


# ---------------------------------------------------------------------------
# Registry functions
# ---------------------------------------------------------------------------


class TestRendererRegistry:
    def test_register_and_unregister_roundtrip(self):
        name = "__test_downstream_compat_renderer__"
        assert not is_renderer_registered(name)

        def factory(tokenizer, image_processor=None):  # type: ignore[no-untyped-def]
            return Qwen3Renderer(tokenizer)

        register_renderer(name, factory)
        assert is_renderer_registered(name)
        assert name in get_registered_renderer_names()

        assert unregister_renderer(name) is True
        assert not is_renderer_registered(name)

    def test_unregister_nonexistent_returns_false(self):
        assert unregister_renderer("__nonexistent__") is False


# ---------------------------------------------------------------------------
# get_renderer: built-in renderer names
# ---------------------------------------------------------------------------

# These are the renderer names downstream projects depend on.
EXPECTED_RENDERER_NAMES = [
    "role_colon",
    "llama3",
    "qwen3",
    "qwen3_disable_thinking",
    "qwen3_instruct",
    "qwen3_5",
    "qwen3_5_disable_thinking",
    "deepseekv3",
    "deepseekv3_disable_thinking",
    "deepseekv3_thinking",
    "kimi_k2",
    "kimi_k25",
    "kimi_k25_disable_thinking",
    "gpt_oss_no_sysprompt",
    "gpt_oss_low_reasoning",
    "gpt_oss_medium_reasoning",
    "gpt_oss_high_reasoning",
]


@pytest.mark.parametrize("renderer_name", EXPECTED_RENDERER_NAMES)
def test_builtin_renderer_name_resolves(renderer_name):
    """get_renderer must not raise ValueError for any name downstream projects use."""
    # We don't actually instantiate (needs a real tokenizer), just verify the
    # name is handled in the factory's dispatch logic.
    src = inspect.getsource(renderers.get_renderer)
    # VL renderers use a different code path but the name must still appear
    assert renderer_name in src or renderer_name.replace("_", " ") in src, (
        f"Renderer name '{renderer_name}' not found in get_renderer dispatch"
    )


# ---------------------------------------------------------------------------
# Renderer abstract interface
# ---------------------------------------------------------------------------


class TestRendererInterface:
    """Verify the Renderer ABC exposes the methods downstream calls."""

    def test_build_generation_prompt_is_method(self):
        assert hasattr(Renderer, "build_generation_prompt")
        assert callable(Renderer.build_generation_prompt)

    def test_build_supervised_example_is_method(self):
        # Downstream calls build_supervised_example (singular)
        assert hasattr(Renderer, "build_supervised_example")

    def test_build_supervised_examples_is_method(self):
        # Some downstream code uses the plural form
        assert hasattr(Renderer, "build_supervised_examples")

    def test_parse_response_is_method(self):
        assert hasattr(Renderer, "parse_response")

    def test_get_stop_sequences_is_abstract(self):
        assert hasattr(Renderer, "get_stop_sequences")

    def test_has_extension_property(self):
        assert hasattr(Renderer, "has_extension_property")

    def test_tokenizer_attribute(self):
        # Downstream accesses renderer.tokenizer
        assert "tokenizer" in Renderer.__init__.__code__.co_varnames

    def test_pickle_metadata_attributes(self):
        # Downstream relies on pickle support
        assert hasattr(Renderer, "_renderer_name")
        assert hasattr(Renderer, "_model_name")
        assert hasattr(Renderer, "_has_image_processor")


# ---------------------------------------------------------------------------
# Signature checks
# ---------------------------------------------------------------------------


class TestSignatures:
    """Verify that key function signatures haven't changed."""

    def test_get_renderer_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(
            renderers.get_renderer, ["name", "tokenizer", "image_processor", "model_name"]
        )

    def test_register_renderer_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(register_renderer, ["name", "factory"])

    def test_unregister_renderer_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(unregister_renderer, ["name"])

    def test_build_generation_prompt_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(Renderer.build_generation_prompt, ["messages", "role", "prefill"])

    def test_build_supervised_example_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(Renderer.build_supervised_example, ["messages", "train_on_what"])

    def test_parse_response_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(Renderer.parse_response, ["response"])

    def test_ensure_text_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(ensure_text, ["content"])

    def test_format_content_as_string_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(format_content_as_string, ["content", "separator"])

    def test_get_text_content_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(get_text_content, ["message"])


# ---------------------------------------------------------------------------
# Specific renderer classes used directly by downstream
# ---------------------------------------------------------------------------


class TestRendererClasses:
    def test_deepseekv3_thinking_renderer_importable(self):
        assert issubclass(DeepSeekV3ThinkingRenderer, Renderer)

    def test_qwen3_renderer_importable(self):
        assert issubclass(Qwen3Renderer, Renderer)

    def test_gpt_oss_renderer_importable(self):
        assert issubclass(GptOssRenderer, Renderer)

    def test_kimi_k2_renderer_importable(self):
        assert issubclass(KimiK2Renderer, Renderer)

    def test_kimi_k25_renderer_importable(self):
        assert issubclass(KimiK25Renderer, Renderer)
