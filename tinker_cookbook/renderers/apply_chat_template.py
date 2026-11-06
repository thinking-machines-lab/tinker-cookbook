"""Model-agnostic renderer that delegates to ``tokenizer.apply_chat_template``."""

from typing import cast

import tinker
import torch

from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    TrainOnWhat,
    format_content_as_string,
    parse_response_for_stop_token,
)


def _apply_chat_template_ids(tokenizer, messages: list[Message], **kwargs) -> list[int]:
    """Call ``tokenizer.apply_chat_template`` and narrow the union return to ``list[int]``."""
    out = tokenizer.apply_chat_template(
        cast(list, messages), tokenize=True, return_dict=False, **kwargs
    )
    return cast(list[int], list(out))


class TitoRenderer(Renderer):
    """Generic Renderer that routes rendering through ``tokenizer.apply_chat_template``.

    The family-specific bits (special tokens, role markers, tool-call format) live
    in the model's Jinja chat template rather than in Python. Works for any
    chat-tuned model whose chat template is prefix-preserving when a tool message
    is appended — the condition the TITO blog post identifies, which is satisfied
    by most modern open-weights families out of the box.

    For families whose stock template breaks prefix preservation (Qwen3 thinking
    is the documented exception), use a patched chat template (e.g. via
    ``trl.chat_template_utils.get_training_chat_template``) before instantiating
    this renderer.
    """

    @property
    def has_extension_property(self) -> bool:
        """Whether successive renders extend the previous render byte-for-byte.

        Returns True under the assumption that the caller has selected a chat
        template that is prefix-preserving for tool messages. Verify with
        ``trl.chat_template_utils.is_chat_template_prefix_preserving`` if unsure.
        """
        return True

    def build_generation_prompt(
        self, messages: list[Message], role="assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        """Render messages to a prompt ready for sampling via ``apply_chat_template``.

        Args:
            messages (list[Message]): Conversation history to render.
            role (Role): Role of the partial message to be completed (unused; the
                template's ``add_generation_prompt`` controls which role opens).
            prefill (str | None): Optional text appended after the generation
                prompt to constrain the start of the model's output.

        Returns:
            tinker.ModelInput: Tokenized prompt.
        """
        ids = _apply_chat_template_ids(self.tokenizer, messages, add_generation_prompt=True)
        if prefill:
            ids.extend(self.tokenizer.encode(prefill, add_special_tokens=False))
        chunks: list[tinker.ModelInputChunk] = [tinker.types.EncodedTextChunk(tokens=ids)]
        return tinker.ModelInput(chunks=chunks)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build a supervised example from messages, with per-token loss weights.

        Computes per-message token spans by re-rendering each ``messages[:i]``
        through ``apply_chat_template`` and taking the deltas. Each span is
        weighted 1 if its message is selected by ``train_on_what`` (whole span
        including any template-added headers — matching what HF's
        ``{% generation %}`` markers produce) and 0 otherwise.

        Args:
            messages (list[Message]): Conversation history to render.
            train_on_what (TrainOnWhat): Which message outputs receive non-zero weight.

        Returns:
            tuple[tinker.ModelInput, torch.Tensor]: Tokenized example and its
                per-token loss weights as a float tensor.
        """
        tok = self.tokenizer

        def render(msgs: list[Message]) -> list[int]:
            return _apply_chat_template_ids(tok, msgs)

        full_ids = render(messages)
        prefix_lens: list[int] = [0]
        for i in range(1, len(messages) + 1):
            prefix_lens.append(len(render(messages[:i])))

        last_user_idx = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)

        weights: list[int] = [0] * len(full_ids)
        for i, msg in enumerate(messages):
            is_assistant = msg["role"] == "assistant"
            is_last_message = i == len(messages) - 1
            is_after_last_user = last_user_idx == -1 or i > last_user_idx

            if train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                has_weight = is_last_message and is_assistant
            elif train_on_what == TrainOnWhat.LAST_ASSISTANT_TURN:
                has_weight = is_assistant and is_after_last_user
            elif train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                has_weight = is_assistant
            elif train_on_what in (TrainOnWhat.ALL_MESSAGES, TrainOnWhat.ALL_TOKENS):
                has_weight = True
            elif train_on_what == TrainOnWhat.CUSTOMIZED:
                has_weight = bool(msg.get("trainable", False))
            else:
                has_weight = False

            if has_weight:
                for j in range(prefix_lens[i], prefix_lens[i + 1]):
                    weights[j] = 1

        chunks: list[tinker.ModelInputChunk] = [tinker.types.EncodedTextChunk(tokens=full_ids)]
        return tinker.ModelInput(chunks=chunks), torch.tensor(weights, dtype=torch.float32)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a single message as an output-only chunk (no header).

        Provided so external code that calls ``render_message`` directly does
        not crash. The result is not byte-faithful to the model's chat template;
        prefer ``build_generation_prompt`` / ``build_supervised_example``, which
        go through ``apply_chat_template`` for the full per-template format.

        Args:
            message (Message): The chat message to render.
            ctx (RenderContext): Positional context (unused).

        Returns:
            RenderedMessage: Empty header and the message content as a single
                output chunk.
        """
        content_text = format_content_as_string(message["content"])
        output_ids = self.tokenizer.encode(content_text, add_special_tokens=False)
        return RenderedMessage(
            header=tinker.types.EncodedTextChunk(tokens=[]),
            output=[tinker.types.EncodedTextChunk(tokens=output_ids)],
        )

    def get_stop_sequences(self) -> list[int]:
        """Return stop sequences for sampling.

        Returns:
            list[int]: Single-element list with the tokenizer's EOS token id.
        """
        return [cast(int, self.tokenizer.eos_token_id)]

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse sampled token IDs back into an assistant Message.

        Args:
            response (list[int]): Raw token IDs from the sampler.

        Returns:
            tuple[Message, ParseTermination]: ``STOP_SEQUENCE`` if the EOS token
                was found, ``MALFORMED`` otherwise.
        """
        return parse_response_for_stop_token(
            response, self.tokenizer, cast(int, self.tokenizer.eos_token_id)
        )
