"""Simple role:content format renderer."""

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ToolSpec,
    ensure_text,
)


class RoleColonRenderer(Renderer):
    """Simple role:content format renderer.

    Format::

        User: <content>

        Assistant: <content>

    This is basically the format used by DeepSeek R1-Zero, and similar to the format
    used by Anthropic, except that they use "Human" instead of "User".
    """

    @property
    def has_extension_property(self) -> bool:
        """RoleColon satisfies the extension property - no content is stripped from history."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a chat message into the ``Role: content`` format.

        Each message is formatted as ``{Role}: {content}\\n\\n`` with a stop_overlap
        of ``User:`` for assistant messages to support the ``\\n\\nUser:`` stop sequence.

        Args:
            message (Message): The chat message to render.
            ctx (RenderContext): Positional context for the message in the conversation.

        Returns:
            RenderedMessage: Header, output, and stop_overlap token chunks.
        """
        header_str = message["role"].capitalize() + ":"
        output_str = " " + ensure_text(message["content"]) + "\n\n"
        # stop_overlap completes the stop sequence "\n\nUser:" for assistant messages.
        # For non-assistant messages, we use a placeholder that's never actually concatenated.
        stop_overlap_str = "User:" if message["role"] == "assistant" else "<UNUSED>"
        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        stop_overlap = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(stop_overlap_str, add_special_tokens=False)
        )
        return RenderedMessage(header=header, output=output, stop_overlap=stop_overlap)

    def get_stop_sequences(self) -> list[str]:
        """Return stop sequences for RoleColon generation.

        Returns:
            list[str]: Single-element list containing the ``\\n\\nUser:`` string stop sequence.
        """
        return ["\n\nUser:"]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse sampled token IDs back into an assistant Message.

        Splits the decoded text on the ``\\n\\nUser:`` stop sequence. Handles EOS
        token stripping and multiple-delimiter edge cases.

        Args:
            response (list[int]): Raw token IDs from the sampler.

        Returns:
            tuple[Message, bool]: The parsed assistant message and whether the response
                terminated cleanly — either via the ``\\n\\nUser:`` stop sequence or
                via EOS. Malformed cases (no terminator at all, ``\\n\\nUser:``
                followed by EOS, or multiple ``\\n\\nUser:`` delimiters) return
                ``False``. Callers that need to distinguish stop-sequence
                termination from EOS termination (e.g. ``ProblemEnv`` with
                ``require_stop_sequence_for_format=True``) should check the EOS
                token themselves; this method intentionally treats both as a
                successful parse so that single-turn evals on base models grade
                normally (see issue #685).
        """
        import logging

        logger = logging.getLogger(__name__)

        # Strip EOS token from the end if present. Base models on single-turn
        # prompts commonly terminate with EOS instead of the "\n\nUser:" stop
        # sequence — that is still a clean end-of-turn signal, so treat it as a
        # successful parse. Without this, EnvFromMessageEnv short-circuits with
        # failed_parse_reward=0 and never grades the response (see issue #685).
        terminated_with_eos = False
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and response and response[-1] == eos_token_id:
            response = response[:-1]
            terminated_with_eos = True

        str_response = str(self.tokenizer.decode(response))
        splitted = str_response.split("\n\nUser:")
        if len(splitted) == 1:
            if terminated_with_eos:
                return Message(role="assistant", content=str_response.strip()), True
            logger.debug(f"Response is not a valid assistant response: {str_response}")
            return Message(role="assistant", content=str_response.strip()), False
        elif len(splitted) == 2:
            before, _after = splitted
            return Message(role="assistant", content=before.strip()), not terminated_with_eos
        else:
            logger.warning(
                "RoleColonRenderer.parse_response saw multiple stop delimiters "
                "(count=%d). Returning parse_success=False. Full response:\n%s",
                len(splitted) - 1,
                str_response,
            )
            return Message(role="assistant", content=splitted[0].strip()), False

    @property
    def _bos_tokens(self) -> list[int]:
        bos_token_str = self.tokenizer.bos_token
        if bos_token_str is None:
            return []
        assert isinstance(bos_token_str, str)
        return self.tokenizer.encode(bos_token_str, add_special_tokens=False)

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Not supported. RoleColon format has no tool calling convention.

        Args:
            tools (list[ToolSpec]): Tool specifications (unused).
            system_prompt (str): System prompt text (unused).

        Raises:
            NotImplementedError: Always raised; RoleColon does not support tool calling.
        """
        raise NotImplementedError("RoleColonRenderer does not support tool calling")
