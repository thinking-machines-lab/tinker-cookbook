"""
Renderers for converting message lists into training and sampling prompts.

Use viz_sft_dataset to visualize the output of different renderers. E.g.,
    python -m tinker_cookbook.supervised.viz_sft_dataset dataset_path=Tulu3Builder renderer_name=role_colon

This module re-exports all renderer classes and types for backwards compatibility.
Code that imports from tinker_cookbook.renderers will continue to work unchanged.
"""

from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.tokenizer_utils import Tokenizer

# Base types and utilities
from tinker_cookbook.renderers.base import (
    # Pydantic base
    StrictBase,
    # Tool types
    ToolCall,
    UnparsedToolCall,
    # Content part types
    TextPart,
    ImagePart,
    ThinkingPart,
    ToolCallPart,
    UnparsedToolCallPart,
    ContentPart,
    # Message types
    Role,
    Content,
    Message,
    RenderContext,
    ToolSpec,
    # Utility functions
    ensure_text,
    ensure_list,
    remove_thinking,
    get_text_content,
    format_content_as_string,
    # Parsing functions
    parse_content_blocks,
    parse_think_blocks,
    # Renderer base class and related
    RenderedMessage,
    TrainOnWhat,
    Renderer,
    tokens_weights_from_strings_weights,
    parse_response_for_stop_token,
    # Image utilities
    ImageProcessorProtocol,
    image_to_chunk,
)

# Renderer implementations
from tinker_cookbook.renderers.role_colon import RoleColonRenderer
from tinker_cookbook.renderers.llama3 import Llama3Renderer
from tinker_cookbook.renderers.qwen3 import (
    Qwen3Renderer,
    Qwen3DisableThinkingRenderer,
    Qwen3InstructRenderer,
    Qwen3VLRenderer,
    Qwen3VLInstructRenderer,
)
from tinker_cookbook.renderers.deepseek_v3 import (
    DeepSeekV3ThinkingRenderer,
    DeepSeekV3DisableThinkingRenderer,
)
from tinker_cookbook.renderers.kimi_k2 import KimiK2Renderer
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer


def get_renderer(
    name: str, tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> Renderer:
    """Factory function to create renderers by name.

    Args:
        name: Renderer name. Supported values:
            - "role_colon": Simple role:content format
            - "llama3": Llama 3 chat format
            - "qwen3": Qwen3 with thinking enabled
            - "qwen3_vl": Qwen3 vision-language with thinking
            - "qwen3_vl_instruct": Qwen3 vision-language instruct (no thinking)
            - "qwen3_disable_thinking": Qwen3 with thinking disabled
            - "qwen3_instruct": Qwen3 instruct 2507 (no thinking)
            - "deepseekv3": DeepSeek V3 (defaults to non-thinking mode)
            - "deepseekv3_disable_thinking": DeepSeek V3 non-thinking (alias)
            - "deepseekv3_thinking": DeepSeek V3 thinking mode
            - "kimi_k2": Kimi K2 Thinking format
            - "gpt_oss_no_sysprompt": GPT-OSS without system prompt
            - "gpt_oss_low_reasoning": GPT-OSS with low reasoning
            - "gpt_oss_medium_reasoning": GPT-OSS with medium reasoning
            - "gpt_oss_high_reasoning": GPT-OSS with high reasoning
        tokenizer: The tokenizer to use.
        image_processor: Required for VL renderers.

    Returns:
        A Renderer instance.

    Raises:
        ValueError: If the renderer name is unknown.
        AssertionError: If a VL renderer is requested without an image_processor.
    """
    if name == "role_colon":
        return RoleColonRenderer(tokenizer)
    elif name == "llama3":
        return Llama3Renderer(tokenizer)
    elif name == "qwen3":
        return Qwen3Renderer(tokenizer)
    elif name == "qwen3_vl":
        assert image_processor is not None, "qwen3_vl renderer requires an image_processor"
        return Qwen3VLRenderer(tokenizer, image_processor)
    elif name == "qwen3_vl_instruct":
        assert image_processor is not None, "qwen3_vl_instruct renderer requires an image_processor"
        return Qwen3VLInstructRenderer(tokenizer, image_processor)
    elif name == "qwen3_disable_thinking":
        return Qwen3DisableThinkingRenderer(tokenizer)
    elif name == "qwen3_instruct":
        return Qwen3InstructRenderer(tokenizer)
    elif name == "deepseekv3":
        # Default to non-thinking mode (matches HF template default behavior)
        return DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "deepseekv3_disable_thinking":
        # Alias for backward compatibility
        return DeepSeekV3DisableThinkingRenderer(tokenizer)
    elif name == "deepseekv3_thinking":
        return DeepSeekV3ThinkingRenderer(tokenizer)
    elif name == "kimi_k2":
        return KimiK2Renderer(tokenizer)
    elif name == "gpt_oss_no_sysprompt":
        return GptOssRenderer(tokenizer, use_system_prompt=False)
    elif name == "gpt_oss_low_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="low")
    elif name == "gpt_oss_medium_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="medium")
    elif name == "gpt_oss_high_reasoning":
        return GptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="high")
    else:
        raise ValueError(f"Unknown renderer: {name}")


__all__ = [
    # Base types
    "StrictBase",
    "ToolCall",
    "UnparsedToolCall",
    "TextPart",
    "ImagePart",
    "ThinkingPart",
    "ToolCallPart",
    "UnparsedToolCallPart",
    "ContentPart",
    "Role",
    "Content",
    "Message",
    "RenderContext",
    "ToolSpec",
    # Utility functions
    "ensure_text",
    "ensure_list",
    "remove_thinking",
    "get_text_content",
    "format_content_as_string",
    "parse_content_blocks",
    "parse_think_blocks",
    # Renderer base
    "RenderedMessage",
    "TrainOnWhat",
    "Renderer",
    "tokens_weights_from_strings_weights",
    "parse_response_for_stop_token",
    # Image utilities
    "ImageProcessorProtocol",
    "image_to_chunk",
    # Renderer implementations
    "RoleColonRenderer",
    "Llama3Renderer",
    "Qwen3Renderer",
    "Qwen3DisableThinkingRenderer",
    "Qwen3InstructRenderer",
    "Qwen3VLRenderer",
    "Qwen3VLInstructRenderer",
    "DeepSeekV3ThinkingRenderer",
    "DeepSeekV3DisableThinkingRenderer",
    "KimiK2Renderer",
    "GptOssRenderer",
    # Factory function
    "get_renderer",
]
