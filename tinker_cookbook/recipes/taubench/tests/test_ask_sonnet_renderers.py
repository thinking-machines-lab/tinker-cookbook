"""Tests for AskSonnetRenderer hierarchy — Direct and Conditioning modes."""

import json

import pytest

from tinker_cookbook.recipes.taubench.components.types import AskSonnetMode
from tinker_cookbook.recipes.taubench.components.ask_sonnet_renderers import (
    AskSonnetRenderer,
    DirectRenderer,
    ConditioningRenderer,
    get_ask_sonnet_renderer,
)


# ---------------------------------------------------------------------------
# DirectRenderer
# ---------------------------------------------------------------------------


class TestDirectRenderer:
    def test_should_return_early(self):
        r = DirectRenderer()
        assert r.should_return_early() is False

    def test_requires_followup(self):
        r = DirectRenderer()
        assert r.requires_followup() is False

    def test_get_tau2_action_uses_sonnet_response_tool_call_tags(self):
        r = DirectRenderer()
        action = r.get_tau2_action(
            sonnet_response='<tool_call>\n{"name":"get_order","arguments":{"id":"1"}}\n</tool_call>',
            qwen_followup=None,
        )
        parsed = json.loads(action)
        assert parsed["name"] == "get_order"
        assert parsed["arguments"] == {"id": "1"}

    def test_get_tau2_action_raw_json(self):
        """Raw JSON without tool_call tags (branch 2 of _extract_action_from_content)."""
        r = DirectRenderer()
        action = r.get_tau2_action(
            sonnet_response='{"name":"cancel_order","arguments":{"order_id":"99"}}',
            qwen_followup=None,
        )
        parsed = json.loads(action)
        assert parsed["name"] == "cancel_order"
        assert parsed["arguments"] == {"order_id": "99"}

    def test_get_tau2_action_plain_text(self):
        r = DirectRenderer()
        action = r.get_tau2_action(
            sonnet_response="Hello, how can I help you?",
            qwen_followup=None,
        )
        assert action == "Hello, how can I help you?"

    def test_format_sonnet_response_for_messages(self):
        r = DirectRenderer()
        msg = r.format_sonnet_response_for_messages("some advice")
        assert msg["role"] == "tool"
        assert "[Sonnet's Advice]:" in msg["content"]
        assert "some advice" in msg["content"]
        assert msg["tool_call_id"] == "ask_sonnet_call"


# ---------------------------------------------------------------------------
# ConditioningRenderer
# ---------------------------------------------------------------------------


class TestConditioningRenderer:
    def test_should_return_early(self):
        r = ConditioningRenderer()
        assert r.should_return_early() is True

    def test_requires_followup(self):
        r = ConditioningRenderer()
        assert r.requires_followup() is True

    def test_get_tau2_action_uses_policy_followup(self):
        r = ConditioningRenderer()
        followup = {"content": '<tool_call>\n{"name":"cancel","arguments":{}}\n</tool_call>'}
        action = r.get_tau2_action(sonnet_response="advice text", qwen_followup=followup)
        parsed = json.loads(action)
        assert parsed["name"] == "cancel"

    def test_get_tau2_action_plain_text_followup(self):
        r = ConditioningRenderer()
        followup = {"content": "I'll check that for you."}
        action = r.get_tau2_action(sonnet_response="advice", qwen_followup=followup)
        assert action == "I'll check that for you."

    def test_get_tau2_action_raises_without_followup(self):
        r = ConditioningRenderer()
        with pytest.raises(ValueError, match="requires policy followup"):
            r.get_tau2_action(sonnet_response="advice", qwen_followup=None)

    def test_format_sonnet_response_for_messages(self):
        r = ConditioningRenderer()
        msg = r.format_sonnet_response_for_messages("do X then Y")
        assert msg["role"] == "tool"
        assert "[Sonnet's Advice]:" in msg["content"]
        assert "do X then Y" in msg["content"]


# ---------------------------------------------------------------------------
# render_for_advisor — shared behavior
# ---------------------------------------------------------------------------


class TestRenderForAdvisor:
    def _make_renderer(self) -> AskSonnetRenderer:
        return DirectRenderer()

    def test_strips_ask_sonnet_instructions(self, sample_system_prompt, sample_tools):
        r = self._make_renderer()
        messages = [
            {"role": "system", "content": sample_system_prompt},
            {"role": "user", "content": "hi"},
        ]
        result = r.render_for_advisor(messages, sample_tools, sample_system_prompt)
        system_msg = result[0]["content"]
        assert "ask_sonnet" not in system_msg.split("# Available Tools")[0]

    def test_removes_final_ask_sonnet_turn(self, sample_tools):
        r = self._make_renderer()
        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "calling ask_sonnet now"},
        ]
        result = r.render_for_advisor(messages, sample_tools, "sys prompt")
        # Last message should be the user message, not the ask_sonnet assistant turn
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "help"

    def test_tool_messages_become_user_messages(self, sample_tools):
        r = self._make_renderer()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "using tool"},
            {"role": "tool", "content": "tool output here", "tool_call_id": "tc1"},
        ]
        result = r.render_for_advisor(messages, sample_tools, "sys")
        tool_as_user = [m for m in result if "[Tool Result]:" in m.get("content", "")]
        assert len(tool_as_user) == 1
        assert tool_as_user[0]["role"] == "user"
        assert "tool output here" in tool_as_user[0]["content"]

    def test_empty_tool_result_gets_placeholder(self, sample_tools):
        r = self._make_renderer()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "", "tool_call_id": "tc1"},
        ]
        result = r.render_for_advisor(messages, sample_tools, "sys")
        tool_msg = [m for m in result if "[Tool Result]:" in m.get("content", "")]
        assert len(tool_msg) == 1
        assert "(empty)" in tool_msg[0]["content"]

    def test_ask_sonnet_excluded_from_advisor_tools(self, sample_tools):
        r = self._make_renderer()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = r.render_for_advisor(messages, sample_tools, "sys")
        system_content = result[0]["content"]
        # The tools section should include get_order_details but NOT ask_sonnet
        assert "get_order_details" in system_content
        # ask_sonnet should not appear in the Available Tools section
        tools_section = system_content.split("# Available Tools")[-1] if "# Available Tools" in system_content else ""
        assert "ask_sonnet" not in tools_section

    def test_assistant_tool_calls_rendered_as_text(self, sample_tools):
        """Tool calls on assistant messages should be converted to <tool_call> text."""
        r = self._make_renderer()

        class FakeFunc:
            name = "get_order_details"
            arguments = '{"order_id": "123"}'

        class FakeToolCall:
            function = FakeFunc()

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "check my order"},
            {"role": "assistant", "content": "Let me look that up", "tool_calls": [FakeToolCall()]},
        ]
        result = r.render_for_advisor(messages, sample_tools, "sys")
        assistant_msg = result[2]
        assert assistant_msg["role"] == "assistant"
        assert "<tool_call>" in assistant_msg["content"]
        assert "get_order_details" in assistant_msg["content"]
        assert "Let me look that up" in assistant_msg["content"]

    def test_non_final_ask_sonnet_not_removed(self, sample_tools):
        """ask_sonnet in a non-final assistant message should NOT be removed."""
        r = self._make_renderer()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "calling ask_sonnet"},
            {"role": "user", "content": "thanks for the help"},
        ]
        result = r.render_for_advisor(messages, sample_tools, "sys")
        # The ask_sonnet assistant message should still be there (it's not the last message)
        assert result[1]["role"] == "assistant"
        assert "ask_sonnet" in result[1]["content"]
        assert result[-1]["role"] == "user"

    def test_only_ask_sonnet_tool_returns_plain_system_prompt(self):
        """When the only tool is ask_sonnet, advisor should get plain system prompt."""
        r = self._make_renderer()
        ask_sonnet_only = [
            {
                "type": "function",
                "function": {
                    "name": "ask_sonnet",
                    "description": "Delegate to Sonnet",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        result = r.render_for_advisor(messages, ask_sonnet_only, "You are helpful.")
        # No tools section should appear since the only tool was filtered out
        assert "# Available Tools" not in result[0]["content"]
        assert result[0]["content"] == "You are helpful."


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_direct_injection_returns_direct_renderer(self):
        r = get_ask_sonnet_renderer(AskSonnetMode.DIRECT_INJECTION)
        assert isinstance(r, DirectRenderer)

    def test_conditioning_returns_conditioning_renderer(self):
        r = get_ask_sonnet_renderer(AskSonnetMode.CONDITIONING)
        assert isinstance(r, ConditioningRenderer)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_ask_sonnet_renderer("invalid")  # type: ignore
