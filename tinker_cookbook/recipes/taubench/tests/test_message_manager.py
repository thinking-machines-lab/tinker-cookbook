"""Tests for MessageManager — message history management."""

from unittest.mock import MagicMock

from tinker_cookbook.recipes.taubench.components.message_manager import MessageManager


class TestMessageManagerInit:
    def test_initial_messages(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        assert len(mm.messages) == 2
        assert mm.messages[0] == {"role": "system", "content": "sys"}
        assert mm.messages[1] == {"role": "user", "content": "hi"}

    def test_system_prompt_stored(self):
        mm = MessageManager(system_prompt="sys prompt", initial_user_content="x")
        assert mm.system_prompt == "sys prompt"


class TestAddUser:
    def test_normal_content(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        mm.add_user("follow up")
        assert mm.messages[-1] == {"role": "user", "content": "follow up"}

    def test_empty_content_becomes_waiting(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        mm.add_user("")
        assert mm.messages[-1]["content"] == "(waiting)"

    def test_whitespace_is_not_replaced(self):
        """Non-empty whitespace string should NOT be replaced (only empty string is falsy)."""
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        mm.add_user(" ")
        assert mm.messages[-1]["content"] == " "


class TestAddToolResult:
    def test_normal_content(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        mm.add_tool_result("order found", tool_call_id="tc1")
        assert mm.messages[-1] == {
            "role": "tool",
            "content": "order found",
            "tool_call_id": "tc1",
        }

    def test_empty_content_becomes_placeholder(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        mm.add_tool_result("")
        assert mm.messages[-1]["content"] == "(empty result)"

    def test_default_tool_call_id(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        mm.add_tool_result("data")
        assert mm.messages[-1]["tool_call_id"] == "tool_call"


class TestAddAssistant:
    def test_add_string(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        mm.add_assistant("response text")
        assert mm.messages[-1] == {"role": "assistant", "content": "response text"}

    def test_add_message_dict(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        msg = {"role": "assistant", "content": "text", "tool_calls": []}
        mm.add_assistant_message_dict(msg)
        assert mm.messages[-1] is msg


class TestMessageOrdering:
    def test_mixed_adds_preserve_order(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hello")
        mm.add_assistant("greeting")
        mm.add_user("question")
        mm.add_assistant("answer with tool")
        mm.add_tool_result("result")
        mm.add_user("thanks")

        roles = [m["role"] for m in mm.messages]
        assert roles == ["system", "user", "assistant", "user", "assistant", "tool", "user"]


class TestAddSonnetResponse:
    def test_delegates_to_renderer(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        renderer = MagicMock()
        renderer.format_sonnet_response_for_messages.return_value = {
            "role": "tool",
            "content": "[Sonnet's Advice]:\nDo X",
            "tool_call_id": "ask_sonnet_call",
        }
        mm.add_sonnet_response("Do X", renderer)
        renderer.format_sonnet_response_for_messages.assert_called_once_with("Do X")
        assert mm.messages[-1]["content"] == "[Sonnet's Advice]:\nDo X"


class TestAddAskSonnetCall:
    def test_appends_message(self):
        mm = MessageManager(system_prompt="sys", initial_user_content="hi")
        call_msg = {"role": "assistant", "content": "", "tool_calls": [{"name": "ask_sonnet"}]}
        mm.add_ask_sonnet_call(call_msg)
        assert mm.messages[-1] is call_msg
