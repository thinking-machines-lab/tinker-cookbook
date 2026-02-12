"""AskSonnetRenderer hierarchy - Different modes for handling ask_sonnet interactions."""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from tinker_cookbook.recipes.taubench.components.types import AskSonnetMode

logger = logging.getLogger(__name__)

# Pattern to match and strip ASK_SONNET_INSTRUCTION from system prompts
_ASK_SONNET_INSTRUCTION_PATTERN = re.compile(
    r"\n\nIMPORTANT: You have access to a special tool called `ask_sonnet`.*?"
    r"for subsequent turns if needed\.",
    re.DOTALL
)


class AskSonnetRenderer(ABC):
    """
    Abstract base class for rendering ask_sonnet interactions.

    Handles both:
    - Preparing messages for advisor API call (render_for_advisor)
    - Processing advisor's response (format_sonnet_response_for_messages, get_tau2_action)
    """

    def render_for_advisor(
        self,
        messages: list[dict],
        tools: list[dict],
        base_system_prompt: str,
    ) -> list[dict]:
        # Strip ask_sonnet instructions from system prompt
        clean_system_prompt = _ASK_SONNET_INSTRUCTION_PATTERN.sub("", base_system_prompt)

        # Remove final ask_sonnet turn if present
        messages_to_render = list(messages)
        if messages_to_render:
            last_msg = messages_to_render[-1]
            if last_msg.get("role") == "assistant" and "ask_sonnet" in last_msg.get("content", ""):
                messages_to_render = messages_to_render[:-1]

        result = []

        for msg in messages_to_render:
            role = msg.get("role", "")

            if role == "system":
                result.append({
                    "role": "system",
                    "content": self._build_system_with_tools(clean_system_prompt, tools),
                })

            elif role == "tool":
                content = msg.get("content", "")
                result.append({
                    "role": "user",
                    "content": f"[Tool Result]: {content}" if content else "[Tool Result]: (empty)",
                })

            elif role == "assistant":
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                if tool_calls:
                    parts = [content] if content else []
                    for tc in tool_calls:
                        tc_json = self._format_tool_call(tc)
                        parts.append(f"<tool_call>\n{tc_json}\n</tool_call>")
                    content = "\n".join(parts)

                result.append({
                    "role": "assistant",
                    "content": content,
                })

            elif role == "user":
                result.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })

            else:
                logger.warning("Unknown message role: %s", role)
                result.append({
                    "role": role,
                    "content": msg.get("content", ""),
                })

        return result

    def _build_system_with_tools(self, base_prompt: str, tools: list[dict]) -> str:
        # Filter out ask_sonnet tool
        filtered_tools = [t for t in tools if t.get("function", {}).get("name") != "ask_sonnet"]

        if not filtered_tools:
            return base_prompt

        tool_descriptions = []
        for tool in filtered_tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            tool_descriptions.append(
                f"- {name}: {desc}\n  Parameters: {json.dumps(params, indent=2)}"
            )
        tools_text = "\n".join(tool_descriptions)

        return f"""{base_prompt}

# Available Tools

You have access to the following tools. To use a tool, respond with a JSON object in this exact format:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

{tools_text}

# Response Format

You MUST respond with EITHER:
1. A tool call using the <tool_call> format above, OR
2. A text message to send to the user

Do NOT respond with empty content. Always provide a response."""

    def _format_tool_call(self, tc: Any) -> str:
        if hasattr(tc, "function"):
            name = tc.function.name
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
        elif isinstance(tc, dict):
            func = tc.get("function", tc)
            name = func.get("name", "unknown")
            args = func.get("arguments", {})
        else:
            return str(tc)

        return json.dumps({"name": name, "arguments": args})

    @abstractmethod
    def format_sonnet_response_for_messages(self, content: str) -> dict:
        pass

    @abstractmethod
    def get_tau2_action(self, sonnet_response: str, qwen_followup: dict | None) -> str:
        pass

    @abstractmethod
    def should_return_early(self) -> bool:
        pass

    @abstractmethod
    def requires_followup(self) -> bool:
        pass

    def _extract_action_from_content(self, content: str) -> str:
        tool_call_match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            content,
            flags=re.DOTALL
        )
        if tool_call_match:
            return tool_call_match.group(1)

        raw_json_match = re.match(r'^\s*(\{.*\})\s*$', content, flags=re.DOTALL)
        if raw_json_match:
            try:
                parsed = json.loads(raw_json_match.group(1))
                if "name" in parsed:
                    return raw_json_match.group(1)
            except json.JSONDecodeError:
                pass

        return re.sub(
            r"<tool_call>.*?</tool_call>",
            "",
            content,
            flags=re.DOTALL
        ).strip()


class ConditioningRenderer(AskSonnetRenderer):
    """Conditioning mode: Sonnet's response is advice; policy decides what to do."""

    def format_sonnet_response_for_messages(self, content: str) -> dict:
        return {
            "role": "tool",
            "content": f"[Sonnet's Advice]:\n{content}",
            "tool_call_id": "ask_sonnet_call",
        }

    def get_tau2_action(self, sonnet_response: str, qwen_followup: dict | None) -> str:
        if qwen_followup is None:
            raise ValueError("Conditioning mode requires policy followup")
        content = qwen_followup.get("content", "")
        return self._extract_action_from_content(content)

    def should_return_early(self) -> bool:
        return True

    def requires_followup(self) -> bool:
        return True


class DirectRenderer(ConditioningRenderer):
    """Direct mode: Sonnet's response is sent directly to tau2."""

    def get_tau2_action(self, sonnet_response: str, qwen_followup: dict | None) -> str:
        return self._extract_action_from_content(sonnet_response)

    def should_return_early(self) -> bool:
        return False

    def requires_followup(self) -> bool:
        return False


def get_ask_sonnet_renderer(mode: AskSonnetMode) -> AskSonnetRenderer:
    if mode == AskSonnetMode.DIRECT_INJECTION:
        return DirectRenderer()
    elif mode == AskSonnetMode.CONDITIONING:
        return ConditioningRenderer()
    else:
        raise ValueError(f"Unknown ask_sonnet mode: {mode}")
