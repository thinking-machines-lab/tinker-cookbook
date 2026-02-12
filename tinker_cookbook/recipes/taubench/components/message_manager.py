"""MessageManager - Manages message history for the policy model."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinker_cookbook.recipes.taubench.components.ask_sonnet_renderers import AskSonnetRenderer

logger = logging.getLogger(__name__)


class MessageManager:
    """
    Manages message history for the policy model.

    Single message history is maintained. Use AdvisorRenderer to convert
    to advisor-compatible format when calling external LLM.
    """

    def __init__(
        self,
        system_prompt: str,
        initial_user_content: str,
    ):
        self.system_prompt = system_prompt
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_content},
        ]

    def add_assistant(self, content: str) -> None:
        msg = {"role": "assistant", "content": content}
        self.messages.append(msg)
        logger.debug("Added assistant message (messages=%d)", len(self.messages))

    def add_assistant_message_dict(self, message: dict) -> None:
        self.messages.append(message)
        logger.debug("Added assistant message dict (messages=%d)", len(self.messages))

    def add_user(self, content: str) -> None:
        if not content:
            content = "(waiting)"
        msg = {"role": "user", "content": content}
        self.messages.append(msg)
        logger.debug("Added user message (messages=%d)", len(self.messages))

    def add_tool_result(
        self,
        content: str,
        tool_call_id: str = "tool_call",
    ) -> None:
        if not content:
            content = "(empty result)"
        tool_msg = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }
        self.messages.append(tool_msg)
        logger.debug("Added tool result (messages=%d)", len(self.messages))

    def add_ask_sonnet_call(self, message: dict) -> None:
        self.messages.append(message)
        logger.debug("Added ask_sonnet call (messages=%d)", len(self.messages))

    def add_sonnet_response(
        self,
        content: str,
        renderer: "AskSonnetRenderer",
    ) -> None:
        policy_msg = renderer.format_sonnet_response_for_messages(content)
        self.messages.append(policy_msg)
        logger.debug("Added advisor response (messages=%d)", len(self.messages))
