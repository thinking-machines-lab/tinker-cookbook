"""Renderer for Llama 3 chat format."""

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    ensure_text,
    parse_response_for_stop_token,
)


class Llama3Renderer(Renderer):
    """Renderer for Llama 3 Instruct models.

    Format::

        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Note: We intentionally differ from HF's stock Llama template:

    - HF prepends "Cutting Knowledge Date..." to system messages; we don't
      (add manually if needed)

    Tool calling is NOT supported for Llama 3. The Llama 3 tool calling format
    uses bare JSON without delimiters, making it impossible to reliably distinguish
    tool calls from regular JSON content in model responses. Use a different model
    or develop your own renderer if you need tool calling.
    """

    @property
    def has_extension_property(self) -> bool:
        """Llama3 satisfies the extension property - no content is stripped from history."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a chat message into Llama 3 header and output token chunks.

        Formats each message as ``<|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>``.

        Args:
            message (Message): The chat message to render.
            ctx (RenderContext): Positional context for the message in the conversation.

        Returns:
            RenderedMessage: Header and output token chunks for the message.
        """
        role = message["role"]
        header_str = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        output_str = ensure_text(message["content"]) + "<|eot_id|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        (token,) = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        return token

    def get_stop_sequences(self) -> list[int]:
        """Return stop sequences for Llama 3 generation.

        Returns:
            list[int]: Single-element list containing the ``<|eot_id|>`` token ID.
        """
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse sampled token IDs back into an assistant Message.

        Strips the ``<|eot_id|>`` stop token if present and decodes the remaining
        tokens into text content.

        Args:
            response (list[int]): Raw token IDs from the sampler.

        Returns:
            tuple[Message, ParseTermination]: ``STOP_SEQUENCE`` if the
                ``<|eot_id|>`` stop token was found, ``MALFORMED`` otherwise.
        """
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)
