"""
Nemotron-3 family renderer.

Nemotron-3 models (NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 and
NVIDIA-Nemotron-3-Super-120B-A12B-BF16) use a chat format similar to Qwen3.5
(im_start/im_end tokens, thinking blocks, XML-style tool calls) but differ in
the following ways:

1. Tool declarations: Nemotron-3 uses structured XML inside <tools>...</tools>
   (Qwen3.5 uses JSON per line).

2. System message ordering: system_prompt comes BEFORE tools text (Qwen3.5
   puts tools first).

3. Empty think block scope: Nemotron-3's HF template prepends <think></think>
   to ALL assistant messages that lack thinking, including historical ones
   (Qwen3.5 only does this for messages after the last user query).

4. Think block separator: one newline between </think> and text content
   (Qwen3.5 uses two newlines).

5. Disable-thinking generation suffix: <think></think> with no trailing
   newlines (Qwen3.5 uses <think>\\n\\n</think>\\n\\n).

6. Empty system message injection: Nemotron-3's HF template always outputs
   a system message block even when none is provided (it always sets
   system_message = "" which is "defined" in Jinja2). Our renderer
   prepends an empty system message in build_generation_prompt and
   build_supervised_example to match this behavior.

Reference: chat_template.jinja in
  /gcs/tml-us-south1-ckpt-01/checkpoints/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/
"""

import dataclasses
import json

from tinker_cookbook.renderers.base import (
    ImagePart,
    Message,
    RenderContext,
    RenderedMessage,
    Role,
    TextPart,
    ToolSpec,
)
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer


def _format_nemotron3_tool_declaration(tool: ToolSpec) -> str:
    """Format a single tool declaration in Nemotron-3's XML format.

    Mirrors the Jinja template logic from chat_template.jinja:
      <function>
      <name>...</name>
      <description>...</description>
      <parameters>
      <parameter>
      <name>...</name>
      <type>...</type>
      <description>...</description>
      <enum>[...]</enum>
      </parameter>
      <required>[...]</required>
      </parameters>
      </function>
    """
    lines = [
        "<function>",
        f"<name>{tool['name']}</name>",
    ]
    if "description" in tool:
        lines.append(f"<description>{tool['description'].strip()}</description>")
    lines.append("<parameters>")
    params = tool.get("parameters") or {}
    if isinstance(params, dict) and "properties" in params:
        for param_name, param_fields in params["properties"].items():
            lines.append("<parameter>")
            lines.append(f"<name>{param_name}</name>")
            if "type" in param_fields:
                lines.append(f"<type>{param_fields['type']!s}</type>")
            if "description" in param_fields:
                lines.append(f"<description>{param_fields['description'].strip()}</description>")
            if "enum" in param_fields:
                lines.append(f"<enum>{json.dumps(param_fields['enum'])}</enum>")
            lines.append("</parameter>")
    if isinstance(params, dict) and "required" in params:
        lines.append(f"<required>{json.dumps(params['required'])}</required>")
    lines.append("</parameters>")
    lines.append("</function>")
    return "\n".join(lines)


class Nemotron3Renderer(Qwen3_5Renderer):
    """Renderer for Nemotron-3 models.

    Subclasses Qwen3_5Renderer for the shared im_start/im_end/thinking/tool-call
    infrastructure, overriding the parts that differ from Qwen3.5:

    - _assistant_header_suffix: adds <think></think> to ALL assistant messages
      whose thinking will NOT appear in the output.
    - render_message: strips thinking only for messages before last_user_index
      (matching HF template's truncate_history_thinking condition).
    - _format_thinking_text: one separator newline after </think> (not two).
    - _format_tool_calls_chunks: single newline prefix + trailing newline after
      each </tool_call> (matching HF template format).
    - parse_response: strips one newline separator after </think> (not two).
    - create_conversation_prefix_with_tools: XML tool declarations, system
      prompt before tools.
    - build_generation_prompt / build_supervised_example: inject empty system
      message when none is present, matching HF template behavior.
    """

    def _normalize_messages(self, messages: list[Message]) -> list[Message]:
        """Prepend empty system message if not present.

        Nemotron-3's HF template always outputs a system message block even
        when none is provided (because it always sets system_message = "" which
        is "defined" in Jinja2). This ensures our token sequence matches.
        """
        if not messages or messages[0]["role"] != "system":
            return [Message(role="system", content="")] + list(messages)
        return messages

    def build_generation_prompt(self, messages: list[Message], *args: object, **kwargs: object):  # type: ignore[override]
        return super().build_generation_prompt(self._normalize_messages(messages), *args, **kwargs)  # type: ignore[arg-type]

    def build_supervised_example(self, messages: list[Message], *args: object, **kwargs: object):  # type: ignore[override]
        return super().build_supervised_example(self._normalize_messages(messages), *args, **kwargs)  # type: ignore[arg-type]

    def _assistant_header_suffix(self, message: Message, ctx: RenderContext) -> str:
        """Prepend <think></think> when thinking will not appear in the output.

        Nemotron-3's HF template prepends <think></think> to assistant message
        content when there are no <think> tags in the output:
        - Historical messages (idx < last_user_index): thinking is stripped,
          so <think></think> is always prepended regardless of original content.
        - Non-historical messages: prepend only if the message has no thinking.

        When a historical message has non-empty text content, the HF template
        produces "<think></think>\\ntext" (with a newline separator). This comes
        from c.split('</think>')[-1] preserving the \\n in _format_thinking_text's
        output. We add the \\n to the header suffix in that case.
        """
        is_historical = ctx.idx < ctx.last_user_index
        content = message.get("content", "")
        has_think = False
        if isinstance(content, list):
            has_think = any(p["type"] == "thinking" for p in content)
        elif isinstance(content, str):
            has_think = "<think>" in content
        # Non-historical with thinking: thinking will be in output, no prefix needed.
        if has_think and not is_historical:
            return ""
        # For historical messages with stripped thinking and non-empty text:
        # add \n separator to match HF template's c.split('</think>')[-1] behavior.
        if is_historical and has_think:
            has_nonempty_text = isinstance(content, list) and any(
                p["type"] == "text" and p.get("text", "") for p in content
            )
            if has_nonempty_text:
                return "<think></think>\n"
        return "<think></think>"

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a message, using idx < last_user_index for thinking stripping.

        Nemotron-3's HF template strips thinking only for messages BEFORE the
        last user message (truncate_history_thinking and loop.index0 < last_user_idx).
        The base Qwen3VLRenderer uses `not ctx.is_last`, which incorrectly strips
        thinking from assistant messages that follow tool responses.
        """
        if message["role"] == "assistant" and ctx.idx >= ctx.last_user_index and not ctx.is_last:
            # Prevent thinking from being stripped: patch is_last=True so that
            # the base class's `not ctx.is_last` condition evaluates to False.
            ctx = dataclasses.replace(ctx, is_last=True)
        return super().render_message(message, ctx)

    def _format_thinking_text(self, thinking: str) -> str:
        """Nemotron-3 uses a single separator newline after </think>."""
        return f"<think>\n{thinking}\n</think>\n"

    def _wrap_qwen_tool_response_chunks(
        self, chunks: list[ImagePart | TextPart]
    ) -> list[ImagePart | TextPart]:
        """Wrap tool response with Nemotron-3's format.

        Nemotron-3 HF template outputs '\\n</tool_response>\\n' (with trailing \\n),
        while the base class uses '\\n</tool_response>' (no trailing \\n).
        """
        return (
            [TextPart(type="text", text="<tool_response>\n")]
            + chunks
            + [TextPart(type="text", text="\n</tool_response>\n")]
        )

    def _format_tool_calls_chunks(self, message: Message) -> list[ImagePart | TextPart]:
        """Format tool_calls for Nemotron-3.

        Differences from Qwen3.5:
        - Single newline prefix (not two) before the first <tool_call>, unless the
          preceding content (thinking) already ends with \\n.
        - Trailing \\n after each </tool_call> (matching HF template line 156:
          '</function>\\n</tool_call>\\n').

        The prefix is omitted when the message has thinking-only content (no
        non-empty text parts), because _format_thinking_text already ends with \\n.
        """
        assert "tool_calls" in message
        content = message.get("content", "")
        has_thinking = isinstance(content, list) and any(p["type"] == "thinking" for p in content)
        has_nonempty_text = isinstance(content, list) and any(
            p["type"] == "text" and p.get("text", "") for p in content
        )
        # Thinking ends with \n; only add \n prefix if there's text after thinking
        # (which won't end with \n) or no thinking at all.
        prefix = "" if (has_thinking and not has_nonempty_text) else "\n"
        calls = "".join(self._format_tool_call_xml(tc) + "\n" for tc in message["tool_calls"])
        return [TextPart(type="text", text=prefix + calls)]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse response, stripping one separator newline after </think>.

        Nemotron-3's HF template inserts exactly one newline between </think>
        and the text content (Qwen3.5 uses two). This override calls the
        Qwen3_5Renderer parser (which handles <think>\\n prefill prepending and
        XML tool call conversion) and then corrects the separator.
        """
        message, success = super().parse_response(response)
        if not success:
            return message, success

        content = message.get("content")
        if isinstance(content, list):
            seen_thinking = False
            first_text_after_thinking: TextPart | None = None
            for p in content:
                if p["type"] == "thinking":
                    seen_thinking = True
                elif seen_thinking and p["type"] == "text":
                    first_text_after_thinking = p
                    break

            # Qwen3_5Renderer already stripped \n\n; put back one \n that was
            # over-stripped (Nemotron-3 has one separator newline, not two).
            # Only re-add if the Qwen3.5 parent stripped exactly two newlines.
            if first_text_after_thinking is not None and not first_text_after_thinking[
                "text"
            ].startswith("\n"):
                # Parent stripped \n\n; we need \n separator, so add one back.
                first_text_after_thinking["text"] = "\n" + first_text_after_thinking["text"]

        return message, success

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system message with Nemotron-3 XML tool specifications.

        Nemotron-3 uses structured XML for tool declarations (unlike Qwen3.5's
        JSON-per-line format). The system prompt also comes *before* the tools
        block, matching the HF template:

            <|im_start|>system
            {system_prompt}

            # Tools
            ...
        """
        tools_text = ""
        if tools:
            tool_declarations = "\n".join(
                _format_nemotron3_tool_declaration(tool) for tool in tools
            )
            tools_text = (
                "# Tools\n\n"
                "You have access to the following functions:\n\n"
                "<tools>\n"
                f"{tool_declarations}\n"
                "</tools>\n\n"
                "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n"
                "<function=example_function_name>\n"
                "<parameter=example_parameter_1>\n"
                "value_1\n"
                "</parameter>\n"
                "<parameter=example_parameter_2>\n"
                "This is the value for the second parameter\n"
                "that can span\n"
                "multiple lines\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n\n"
                "<IMPORTANT>\n"
                "Reminder:\n"
                "- Function calls MUST follow the specified format: "
                "an inner <function=...></function> block must be nested within "
                "<tool_call></tool_call> XML tags\n"
                "- Required parameters MUST be specified\n"
                "- You may provide optional reasoning for your function call in natural language "
                "BEFORE the function call, but NOT after\n"
                "- If there is no function call available, answer the question like normal with "
                "your current knowledge and do not tell the user about function calls\n"
                "</IMPORTANT>"
            )

        if tools_text:
            # Nemotron-3 puts system_prompt BEFORE tools (opposite of Qwen3.5)
            content = system_prompt + "\n\n" + tools_text if system_prompt else tools_text
        else:
            content = system_prompt

        return [Message(role="system", content=content)]


class Nemotron3DisableThinkingRenderer(Nemotron3Renderer):
    """Renderer for Nemotron-3 models with thinking disabled.

    Matches the Nemotron-3 HF template with enable_thinking=False. The only
    difference from Nemotron3Renderer is the generation suffix:
    <think></think> (no trailing newlines) instead of <think>\\n.
    """

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        maybe_newline = "\n" if ctx.idx > 0 else ""
        header_str = f"{maybe_newline}<|im_start|>{role}\n<think></think>"
        return self.tokenizer.encode(header_str, add_special_tokens=False)
