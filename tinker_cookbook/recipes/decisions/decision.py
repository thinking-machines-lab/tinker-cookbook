"""Decisions: a question with a fixed set of choices, rendered so that a single answer token
selects a choice, and sampled by scoring that token for every choice at once.

A :class:`ChoiceRenderer` decides how answers are written; the default
:class:`PrefixedChoiceRenderer` writes ``choice#A``, ``choice#B``, ....
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import tinker
import torch

from tinker_cookbook.exceptions import ConfigurationError, DataValidationError, RendererError
from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer


async def sample_decision(
    sampling_client: tinker.SamplingClient, rendering: DecisionRendering
) -> dict[str, float]:
    """The probability of each label of a rendered decision, in choice order.

    One ``sample_async`` request with ``max_tokens=1``; ``asyncio.gather`` calls to batch. The
    probabilities are a softmax over the labels' tokens at the answer position.
    """
    labels = rendering.labels
    # Row i of target_prompt_logprobs scores prompt position i + 1.
    target_ids = torch.full((len(rendering.tokens) - 1, len(labels)), -1, dtype=torch.int64)
    target_ids[rendering.choice_pos - 1] = torch.tensor(
        [rendering.label_tokens[label] for label in labels], dtype=torch.int64
    )
    response = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(rendering.tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1),
        target_prompt_logprobs=tinker.TensorData.from_torch_sparse(target_ids, pad_value=-1),
    )
    if response.target_prompt_logprobs is None:
        raise RuntimeError("The sampling response did not include target_prompt_logprobs.")
    logprobs = response.target_prompt_logprobs.to_torch()[rendering.choice_pos - 1]
    probabilities = torch.softmax(logprobs.float(), dim=0).tolist()
    return dict(zip(labels, probabilities, strict=True))


@dataclass(frozen=True)
class Decision:
    """A question and its ordered ``(label, description)`` choices.

    Labels key the results; descriptions are shown to the model and may be empty. There must be
    at least two choices, with distinct labels.
    """

    text: str
    choices: Sequence[tuple[str, str]]

    def __post_init__(self) -> None:
        labels = self.labels
        if len(labels) < 2:
            raise DataValidationError(f"A decision needs at least two choices, got {labels}")
        if len(set(labels)) != len(labels):
            raise DataValidationError(f"Choice labels must be distinct, got {labels}")

    @property
    def labels(self) -> list[str]:
        """The choice labels, in choice order."""
        return [label for label, _ in self.choices]


@dataclass(frozen=True)
class DecisionRendering:
    """A rendered decision: ``tokens`` end at the answer position ``choice_pos``, where the
    token is a placeholder, and ``label_tokens`` gives the token that selects each label."""

    tokens: list[int]
    choice_pos: int
    label_tokens: dict[str, int]

    @property
    def labels(self) -> list[str]:
        return list(self.label_tokens)

    def with_label(self, label: str) -> list[int]:
        """``tokens`` with ``label``'s token at the answer position."""
        tokens = list(self.tokens)
        tokens[self.choice_pos] = self.label_tokens[label]
        return tokens


SYSTEM_PROMPT_TEMPLATE = """\
Answer the user's question by picking exactly one of the choices they supply. Reply with the id of your choice, written as `{answer_format}`, and nothing else."""

USER_PROMPT_TEMPLATE = """\
{decision}

Choices (your answer must be exactly one of these):
{choices}"""


class DecisionRenderer:
    """Render a :class:`Decision` into the tokens that :func:`sample_decision` scores.

    Pass the model's cookbook renderer (``model_info.get_recommended_renderer_name``) and,
    optionally, a :class:`ChoiceRenderer` deciding how answers are written; the default writes
    ``choice#A``, ``choice#B``, .... ``effort`` is the thinking effort for Inkling's ``tml_v0``
    renderer only.
    """

    def __init__(
        self,
        renderer: Renderer,
        choice_renderer: ChoiceRenderer | None = None,
        *,
        effort: float = 0.0,
    ):
        self.renderer = renderer
        self.tokenizer = renderer.tokenizer
        self.effort = effort
        self.choice_renderer: ChoiceRenderer = (
            choice_renderer if choice_renderer is not None else PrefixedChoiceRenderer(renderer)
        )

    def render(self, decision: Decision) -> DecisionRendering:
        """The decision as a chat conversation, cut at the answer token.

        The conversation is a system message with the answer format, a user message with the
        question and choices, and the answer format as the assistant reply. Raises
        :class:`RendererError` if the choice renderer and the chat template disagree.
        """
        choice_rendering = self._render_choices(decision)
        answer_tokens = choice_rendering.answer_tokens
        tokens = self._render_conversation(
            [
                *self._build_messages(decision, choice_rendering),
                Message(role="assistant", content=str(self.tokenizer.decode(answer_tokens))),
            ]
        )
        n = len(answer_tokens)
        start = next(
            (i for i in range(len(tokens) - n, -1, -1) if tokens[i : i + n] == answer_tokens),
            None,
        )
        if start is None:
            raise RendererError(
                f"The reply tokens {answer_tokens} do not appear in the rendered conversation; "
                "the chat template tokenized the reply differently from the choice renderer."
            )
        choice_pos = start + choice_rendering.answer_position
        return DecisionRendering(
            tokens=tokens[: choice_pos + 1],
            choice_pos=choice_pos,
            label_tokens={label: choice_rendering.label_tokens[label] for label in decision.labels},
        )

    def _render_choices(self, decision: Decision) -> ChoiceRendering:
        labels = decision.labels
        choice_rendering = self.choice_renderer.render_choices(labels)
        if set(choice_rendering.label_tokens) != set(labels):
            raise RendererError(
                f"The choice renderer returned label_tokens keyed by "
                f"{sorted(choice_rendering.label_tokens)}, expected the decision's labels "
                f"{labels}."
            )
        return choice_rendering

    def _build_messages(
        self, decision: Decision, choice_rendering: ChoiceRendering
    ) -> list[Message]:
        answer_format = str(self.tokenizer.decode(choice_rendering.answer_tokens))
        lines = []
        for label, description in decision.choices:
            reply = str(self.tokenizer.decode(choice_rendering.answer_tokens_for(label)))
            choice_id = str(self.tokenizer.decode([choice_rendering.label_tokens[label]]))
            line = reply if choice_id == label else f'{reply} = "{label}"'
            lines.append(f"{line}: {description}" if description else line)
        return [
            Message(
                role="system", content=SYSTEM_PROMPT_TEMPLATE.format(answer_format=answer_format)
            ),
            Message(
                role="user",
                content=USER_PROMPT_TEMPLATE.format(
                    decision=decision.text, choices="\n".join(lines)
                ),
            ),
        ]

    def _render_conversation(self, messages: Sequence[Message]) -> list[int]:
        conversation = list(messages)
        if isinstance(self.renderer, TmlV0Renderer):
            model_input, _ = self.renderer.build_supervised_example(
                conversation, effort=self.effort
            )
        else:
            model_input, _ = self.renderer.build_supervised_example(conversation)
        return model_input.to_ints()


@dataclass(frozen=True)
class ChoiceRendering:
    """The reply the model is told to write, with a placeholder token at ``answer_position``
    (e.g. ``choice#?``), and the token that replaces it for each label."""

    answer_tokens: list[int]
    answer_position: int
    label_tokens: dict[str, int]

    def answer_tokens_for(self, label: str) -> list[int]:
        """The reply tokens that select ``label``."""
        tokens = list(self.answer_tokens)
        tokens[self.answer_position] = self.label_tokens[label]
        return tokens


class ChoiceRenderer(Protocol):
    """Decides how a set of labels is answered; ``label_tokens`` must be keyed by ``labels``."""

    def render_choices(self, labels: Sequence[str]) -> ChoiceRendering: ...


class PrefixedChoiceRenderer(ChoiceRenderer):
    """Answer with ``<prefix><id>`` for the ids ``A``, ``B``, ``C``, ... in choice order.

    The placeholder shown to the model is ``<prefix><placeholder>``. Every id and the
    placeholder must be exactly one token after the prefix, else :class:`ConfigurationError`.
    Up to 26 choices.
    """

    IDS = string.ascii_uppercase

    def __init__(self, renderer: Renderer, prefix: str = "choice#", placeholder: str = "?"):
        self.tokenizer = renderer.tokenizer
        self.prefix = prefix
        self.prefix_tokens: list[int] = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.answer_tokens: list[int] = self.tokenizer.encode(
            prefix + placeholder, add_special_tokens=False
        )
        n = len(self.prefix_tokens)
        if self.answer_tokens[:n] != self.prefix_tokens or len(self.answer_tokens) != n + 1:
            raise ConfigurationError(
                f"The placeholder {placeholder!r} must be exactly one token after the prefix "
                f"{prefix!r}, got {self.answer_tokens} vs prefix {self.prefix_tokens}"
            )

    def render_choices(self, labels: Sequence[str]) -> ChoiceRendering:
        if len(labels) > len(self.IDS):
            raise ConfigurationError(
                f"At most {len(self.IDS)} choices are supported, got {len(labels)}"
            )

        n = len(self.prefix_tokens)
        label_tokens: dict[str, int] = {}
        problems: list[str] = []
        for label, choice_id in zip(labels, self.IDS, strict=False):
            tokens = self.tokenizer.encode(self.prefix + choice_id, add_special_tokens=False)
            if tokens[:n] == self.prefix_tokens and len(tokens) == n + 1:
                label_tokens[label] = tokens[n]
            else:
                pieces = [self.tokenizer.decode([t]) for t in tokens[n:]] or [
                    "<merged into prefix>"
                ]
                problems.append(f"{choice_id!r} -> {pieces}")
        if problems:
            raise ConfigurationError(
                f"Every choice id must be exactly one token after the prefix "
                f"{self.prefix!r}; these are not: {', '.join(problems)}. Pick a prefix that "
                "ends at a token boundary."
            )
        if len(set(label_tokens.values())) != len(label_tokens):
            raise ConfigurationError(
                f"Choice ids must tokenize to distinct tokens after the prefix "
                f"{self.prefix!r}, got {label_tokens}"
            )
        return ChoiceRendering(
            answer_tokens=self.answer_tokens, answer_position=n, label_tokens=label_tokens
        )
