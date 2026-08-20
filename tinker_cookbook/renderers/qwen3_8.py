"""
Qwen3.8 family renderer.

Qwen3.8 models share Qwen3.5/3.6's tokenizer, special tokens, vision tokens,
and XML tool-calling format, but the chat template differs in three ways:

1. Reasoning effort: when thinking is enabled the template accepts a
   ``reasoning_effort`` kwarg (``"xhigh"`` (default), ``"medium"``, ``"low"``).
   ``xhigh`` and ``low`` prepend an instruction sentence to the system message
   (creating one if the conversation has none); ``medium`` adds nothing.
2. Thinking is preserved in history by default: the HF template's
   ``preserve_thinking`` defaults to true, so every assistant message keeps its
   ``<think>`` block (empty when the turn did not reason). Qwen3.5/3.6 stripped
   history thinking by default. This renderer therefore defaults
   ``strip_thinking_from_history=False`` and has the extension property.
3. An empty system message renders nothing (Qwen3.5/3.6 emitted an empty
   ``<|im_start|>system`` block).

Reference: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja
"""

import json
from typing import cast

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    Role,
    TextPart,
)
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer

# Instruction sentences the HF template injects into the system message per
# reasoning effort. ``medium`` deliberately maps to no instruction.
REASONING_EFFORT_INSTRUCTIONS: dict[str, str] = {
    "xhigh": (
        "Reasoning effort is set to xhigh. Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize "
        "correctness, consistency, and clarity in the final answer."
    ),
    "medium": "",
    "low": (
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
        "directly to the conclusion without unnecessary elaboration."
    ),
}


class Qwen3_8Renderer(Qwen3_5Renderer):
    """
    Renderer for Qwen3.8 models.

    Subclasses Qwen3_5Renderer for the shared im_start/im_end/thinking/XML-tool
    infrastructure, overriding what the Qwen3.8 HF template changes:

    - build_generation_prompt / build_supervised_example: inject the reasoning
      effort instruction into the system message (prepended to an existing one,
      inserted as a new first message otherwise), and drop an empty system
      message the template would render nothing for.
    - _assistant_header_suffix: with the default strip_thinking_from_history=False
      (HF ``preserve_thinking`` defaults to true) every assistant message that did
      not reason gets an empty think block, regardless of position.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: ImageProcessor | None = None,
        strip_thinking_from_history: bool = False,
        merge_text_chunks: bool = True,
        reasoning_effort: str = "xhigh",
    ):
        """Initialize the Qwen3.8 renderer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding.
            image_processor (ImageProcessor | None): Processor for image content.
            strip_thinking_from_history (bool): When False (default, matching the
                HF template's ``preserve_thinking=true`` default), thinking blocks
                are kept on every assistant message, giving the extension property.
                Set to True for the Qwen3.5/3.6-style positional stripping
                (HF ``preserve_thinking=false``).
            merge_text_chunks (bool): When True (default), merges consecutive text
                parts into a single tokenization unit for HF template compatibility.
            reasoning_effort (str): One of ``"xhigh"`` (default), ``"medium"``,
                ``"low"``. Matches the HF template's ``reasoning_effort`` kwarg;
                ignored when thinking is disabled.
        """
        if reasoning_effort not in REASONING_EFFORT_INSTRUCTIONS:
            raise RendererError(
                f"Unexpected reasoning effort {reasoning_effort}. Supported types are "
                "xhigh (default), medium, and low."
            )
        super().__init__(
            tokenizer,
            image_processor=image_processor,
            strip_thinking_from_history=strip_thinking_from_history,
            merge_text_chunks=merge_text_chunks,
        )
        self.reasoning_effort = reasoning_effort

    @property
    def _reasoning_instructions(self) -> str:
        """The instruction sentence for this effort, empty when thinking is off."""
        if self.disables_thinking:
            return ""
        return REASONING_EFFORT_INSTRUCTIONS[self.reasoning_effort]

    @property
    def has_extension_property(self) -> bool:
        """Qwen3.8 cannot claim the extension property despite preserving thinking.

        A turn that did not reason is sampled after the prompt's open ``<think>\\n``
        (so its tokens continue with a lone ``\\n``), but history writes the closed
        empty block whose ``\\n\\n`` the tokenizer merges into a single token — the
        sampled sequence is therefore not a token-level prefix of the next prompt.
        Reporting False makes build_supervised_examples fall back to per-message
        splitting, which is correct for every conversation shape.
        """
        return False

    def _normalize_messages(self, messages: list[Message]) -> list[Message]:
        """Apply the template's system-message handling to the conversation.

        Prepends the reasoning instruction to the system message (the template
        puts it before both tool declarations and user-provided content), adds a
        system message carrying just the instruction when the conversation has
        none, and drops an empty system message the template renders nothing for.

        Idempotent: build_supervised_example internally rebuilds the generation
        prompt from the already-normalized messages, so a system message that
        already starts with the instruction is left alone. Rebuilt messages keep
        the original's other fields (e.g. ``trainable`` for CUSTOMIZED training),
        and a synthetic instruction message carries ``trainable=False``.
        """
        instructions = self._reasoning_instructions
        if messages and messages[0]["role"] == "system":
            first, rest = messages[0], list(messages[1:])
            content = first["content"]
            if isinstance(content, list):
                texts = [p["text"] for p in content if p["type"] == "text"]
                if len(texts) == len(content):
                    # The template concatenates text parts (render_content) and
                    # applies |trim, so a text-only list renders exactly like the
                    # joined string — including dropping a whitespace-only message.
                    content = "".join(texts)
                else:
                    # Non-text parts (the HF template rejects these in system
                    # messages): just prepend the instruction and pass through.
                    if not instructions:
                        return list(messages)
                    parts = list(content)
                    if parts[0]["type"] == "text" and parts[0]["text"].startswith(instructions):
                        return list(messages)
                    parts = [TextPart(type="text", text=f"{instructions}\n\n")] + parts
                    return [cast(Message, {**first, "content": parts})] + rest
            content = content.strip()
            if not content and not instructions:
                return rest
            if instructions and not (
                content == instructions or content.startswith(f"{instructions}\n\n")
            ):
                content = f"{instructions}\n\n{content}" if content else instructions
            return [cast(Message, {**first, "content": content})] + rest
        if instructions:
            synthetic = Message(role="system", content=instructions)
            # CUSTOMIZED training requires a trainable field on every message, while
            # every other mode requires its absence — mirror the conversation.
            if any("trainable" in m for m in messages):
                synthetic["trainable"] = False
            return [synthetic] + list(messages)
        return list(messages)

    def build_generation_prompt(self, messages: list[Message], *args: object, **kwargs: object):  # type: ignore[override]
        """Build generation prompt with the Qwen3.8 system-message normalization."""
        return super().build_generation_prompt(self._normalize_messages(messages), *args, **kwargs)  # type: ignore[arg-type]

    def build_supervised_example(self, messages: list[Message], *args: object, **kwargs: object):  # type: ignore[override]
        """Build supervised example with the Qwen3.8 system-message normalization."""
        return super().build_supervised_example(self._normalize_messages(messages), *args, **kwargs)  # type: ignore[arg-type]

    def _format_tool_call_argument(self, value: object) -> str:
        """Qwen3.8 (like 3.6) serializes every non-string value with ``|tojson``:
        booleans and null render as JSON ``true``/``null``, not Python's
        ``True``/``None`` (which the older Qwen3.5 template used)."""
        return value if isinstance(value, str) else json.dumps(value)

    def _strips_thinking_for(self, message: Message, ctx: RenderContext) -> bool:
        """Strip only at or before the last user message, matching the template.

        With ``preserve_thinking=false`` the HF template still keeps reasoning for
        every assistant after the last query, so a tool-call turn followed by tool
        results retains its thinking. (Qwen3.5 keeps its historical over-stripping
        for backward compatibility; see Qwen3VLRenderer._strips_thinking_for.)
        """
        return (
            self.strip_thinking_from_history
            and message["role"] == "assistant"
            and ctx.idx <= ctx.last_user_index
        )

    def _assistant_header_suffix(self, message: Message, ctx: RenderContext) -> str:
        """Insert an empty think block for assistant messages that did not reason.

        With thinking preserved in history (the default), the HF template frames
        every assistant message, so the empty block is positional-independent.
        With strip_thinking_from_history=True (HF ``preserve_thinking=false``) the
        template falls back to the Qwen3.5/3.6 positional rule.

        Only structured ThinkingParts count as reasoning: Qwen3.8's template no
        longer extracts inline ``<think>`` tags from string content (it reads
        reasoning_content only), so an inline block is ordinary content and the
        empty framing block is still written before it.
        """
        if self.strip_thinking_from_history and ctx.idx <= ctx.last_user_index:
            return ""
        content = message.get("content", "")
        has_reasoning = isinstance(content, list) and any(p["type"] == "thinking" for p in content)
        return "" if has_reasoning else "<think>\n\n</think>\n\n"


class Qwen3_8DisableThinkingRenderer(Qwen3_8Renderer):
    """
    Renderer for Qwen3.8 models with thinking disabled.

    Matches the Qwen3.8 HF template with enable_thinking=False: the generation
    suffix closes the think block (<think>\\n\\n</think>\\n\\n) and no reasoning
    effort instruction is injected. History framing is unchanged — the template
    still preserves (empty) think blocks on historical assistant messages.
    """

    disables_thinking = True

    def _generation_suffix_str(self, role: Role, ctx: RenderContext) -> str:
        maybe_newline = "\n" if ctx.idx > 0 else ""
        return f"{maybe_newline}<|im_start|>{role}\n<think>\n\n</think>\n\n"


__all__ = [
    "REASONING_EFFORT_INSTRUCTIONS",
    "Qwen3_8DisableThinkingRenderer",
    "Qwen3_8Renderer",
]
