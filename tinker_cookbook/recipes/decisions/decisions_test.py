"""Unit tests for the decision model recipe; real tokenizers, no API key.

Rendering tests run once per model in ``MODELS``, which also holds each chat format's user
and reply headers.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import tinker
import torch

from tinker_cookbook.exceptions import ConfigurationError, DataValidationError, RendererError
from tinker_cookbook.recipes.decisions import (
    ChoiceRendering,
    Decision,
    DecisionRenderer,
    PrefixedChoiceRenderer,
    decision_choice_datums,
    decision_dist_datums,
    probability_loss,
    sample_decision,
)
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer


@dataclass(frozen=True)
class ModelCase:
    name: str
    renderer_name: str
    reply_header: str
    """What the chat format writes between the user turn and the assistant's reply text."""
    user_header: str
    """What the chat format writes at the start of the user turn."""


MODELS = {
    "inkling": ModelCase(
        "thinkingmachines/Inkling-Small",
        "tml_v0",
        reply_header="<|message_model|><|content_text|>",
        user_header="<|message_user|>",
    ),
    "qwen": ModelCase(
        "Qwen/Qwen3.5-4B",
        "qwen3_5",
        reply_header="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        user_header="<|im_start|>user\n",
    ),
}

PREFIX = "choice#"
PLACEHOLDER = "?"
IDS = "ABC"

MESSAGE = "My running shoes arrived in the wrong size. Can I swap them for a size 10?"

DEPARTMENT_DECISION = Decision(
    f"Customer message: {MESSAGE}\n\nWhich team should handle this?",
    [
        ("returns", "Exchanges, wrong or damaged items"),
        ("shipping", "Delivery status, delays, lost packages"),
        ("billing", "Charges, invoices, payment problems"),
    ],
)
MULTIWORD_DECISION = Decision(
    f"Customer message: {MESSAGE}\n\nHow does the customer sound?",
    [("Calm", "No frustration"), ("Very angry", "Strong frustration")],
)
FRUSTRATION_DECISION = Decision(
    f"Customer message: {MESSAGE}\n\nHow frustrated is the customer, from 0 (calm) to 2 (angry)?",
    [("0", ""), ("1", ""), ("2", "")],
)
EXCHANGE_DECISION = Decision(
    f"Customer message: {MESSAGE}\n\nDoes the message request an exchange?",
    [("yes", "It asks for an exchange"), ("no", "It does not")],
)
ALL_DECISIONS = [DEPARTMENT_DECISION, FRUSTRATION_DECISION, EXCHANGE_DECISION]
DECISION_IDS = ["department", "frustration", "exchange"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=list(MODELS), ids=list(MODELS))
def model(request: pytest.FixtureRequest) -> ModelCase:
    return MODELS[request.param]


@pytest.fixture(scope="module")
def tokenizer(model: ModelCase) -> Tokenizer:
    return get_tokenizer(model.name)


@pytest.fixture(scope="module")
def renderer(model: ModelCase, tokenizer: Tokenizer) -> Renderer:
    return get_renderer(model.renderer_name, tokenizer)


@pytest.fixture(scope="module")
def decision_renderer(renderer: Renderer) -> DecisionRenderer:
    return DecisionRenderer(renderer, PrefixedChoiceRenderer(renderer, PREFIX))


@pytest.fixture(scope="module")
def inkling_tokenizer() -> Tokenizer:
    return get_tokenizer(MODELS["inkling"].name)


@pytest.fixture(scope="module")
def inkling_renderer(inkling_tokenizer: Tokenizer) -> Renderer:
    return get_renderer("tml_v0", inkling_tokenizer)


@pytest.fixture(scope="module")
def inkling_decision_renderer(inkling_renderer: Renderer) -> DecisionRenderer:
    return DecisionRenderer(inkling_renderer)


def _mock_sampling_client(logprobs_by_label: list[float]) -> MagicMock:
    """A sampling client whose ``sample_async`` returns the given logprobs at the answer row."""

    async def sample_async(
        *,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        target_prompt_logprobs: tinker.TensorData,
    ) -> MagicMock:
        assert num_samples == 1
        assert sampling_params.max_tokens == 1
        target = target_prompt_logprobs.to_torch(pad_value=-1)
        assert list(target.shape) == [prompt.length - 1, len(logprobs_by_label)]
        (row,) = torch.nonzero(target[:, 0] != -1).flatten().tolist()
        dense = torch.zeros_like(target, dtype=torch.float32)
        dense[row, :] = torch.tensor(logprobs_by_label)
        return MagicMock(target_prompt_logprobs=tinker.TensorData.from_torch(dense))

    client = MagicMock()
    client.sample_async = AsyncMock(side_effect=sample_async)
    return client


def _split_system_user(text: str, model: ModelCase) -> tuple[str, str]:
    system_end = text.index(model.user_header)
    return text[:system_end], text[system_end:]


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def test_decision_labels_follow_choice_order() -> None:
    assert DEPARTMENT_DECISION.labels == ["returns", "shipping", "billing"]
    assert Decision("q", [("a", ""), ("b", "")]) == Decision("q", [("a", ""), ("b", "")])
    with pytest.raises(DataValidationError):
        Decision("q", [("a", ""), ("a", "again")])
    with pytest.raises(DataValidationError):
        Decision("q", [("a", "")])


# ---------------------------------------------------------------------------
# Choice rendering
# ---------------------------------------------------------------------------


class TestPrefixedChoiceRenderer:
    @pytest.mark.parametrize(
        "labels",
        [
            ["returns", "shipping", "billing"],
            ["0", "1", "2"],
            ["yes", "no"],
            ["Calm", "Very angry"],
        ],
    )
    def test_labels_get_letter_ids_in_order(
        self, renderer: Renderer, tokenizer: Tokenizer, labels: Sequence[str]
    ) -> None:
        rendering = PrefixedChoiceRenderer(renderer, PREFIX).render_choices(labels)
        prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
        assert rendering.answer_position == len(prefix_tokens)
        assert tokenizer.decode(rendering.answer_tokens) == PREFIX + PLACEHOLDER
        assert rendering.answer_tokens[-1] not in rendering.label_tokens.values()
        assert list(rendering.label_tokens) == labels
        for label, letter in zip(labels, IDS, strict=False):
            assert rendering.answer_tokens_for(label) == tokenizer.encode(
                PREFIX + letter, add_special_tokens=False
            )
            assert tokenizer.decode(rendering.answer_tokens_for(label)) == PREFIX + letter

    def test_all_26_ids_are_distinct_single_tokens(
        self, renderer: Renderer, tokenizer: Tokenizer
    ) -> None:
        labels = [f"option {i}" for i in range(26)]
        rendering = PrefixedChoiceRenderer(renderer, PREFIX).render_choices(labels)
        assert len(set(rendering.label_tokens.values())) == 26
        assert tokenizer.decode(rendering.answer_tokens_for("option 25")) == PREFIX + "Z"

    @pytest.mark.parametrize(
        "prefix, labels",
        [
            ("answer:", ["x", "y"]),  # ":A" merges into one token
            (PREFIX, [str(i) for i in range(27)]),
        ],
        ids=["merging-prefix", "too-many"],
    )
    def test_rejects_unusable_inputs(
        self, renderer: Renderer, prefix: str, labels: Sequence[str]
    ) -> None:
        with pytest.raises(ConfigurationError):
            PrefixedChoiceRenderer(renderer, prefix=prefix).render_choices(labels)

    def test_placeholder_may_be_a_choice_id(self, renderer: Renderer) -> None:
        rendering = PrefixedChoiceRenderer(renderer, PREFIX, placeholder="A").render_choices(
            ["x", "y"]
        )
        pos = rendering.answer_position
        assert rendering.answer_tokens[pos] == rendering.label_tokens["x"]
        assert rendering.answer_tokens_for("y")[pos] == rendering.label_tokens["y"]


# ---------------------------------------------------------------------------
# Decision rendering
# ---------------------------------------------------------------------------


class TestDecisionRenderer:
    def test_prompt_layout(
        self, model: ModelCase, decision_renderer: DecisionRenderer, tokenizer: Tokenizer
    ) -> None:
        rendering = decision_renderer.render(DEPARTMENT_DECISION)
        text = cast(str, tokenizer.decode(rendering.tokens))
        system, user = _split_system_user(text, model)

        # The system message carries the answer format and none of the choices.
        assert PREFIX + PLACEHOLDER in system
        assert not any(PREFIX + letter in system for letter in IDS)
        assert not any(label in system for label in DEPARTMENT_DECISION.labels)

        # The user message carries the question, then every choice's reply, label and
        # description, in order.
        assert DEPARTMENT_DECISION.text in user
        positions = []
        for (label, description), letter in zip(DEPARTMENT_DECISION.choices, IDS, strict=True):
            reply = PREFIX + letter
            assert reply in user and label in user and description in user
            positions.append(user.index(reply))
        assert user.index(DEPARTMENT_DECISION.text) < positions[0]
        assert positions == sorted(positions)

        # The sequence ends with the reply header and the answer format, and the placeholder
        # token is the last one.
        assert text.endswith(model.reply_header + PREFIX + PLACEHOLDER)
        assert rendering.choice_pos == len(rendering.tokens) - 1
        assert rendering.tokens[rendering.choice_pos] not in rendering.label_tokens.values()
        assert rendering.labels == DEPARTMENT_DECISION.labels
        assert rendering.with_label("returns")[-1] == rendering.label_tokens["returns"]

    def test_choices_without_descriptions_still_list_id_and_label(
        self, model: ModelCase, decision_renderer: DecisionRenderer, tokenizer: Tokenizer
    ) -> None:
        rendering = decision_renderer.render(FRUSTRATION_DECISION)
        _, user = _split_system_user(cast(str, tokenizer.decode(rendering.tokens)), model)
        for label, letter in zip(FRUSTRATION_DECISION.labels, IDS, strict=True):
            assert PREFIX + letter in user
            assert label in user[user.index(PREFIX + letter) :]

    def test_multi_word_labels_render(
        self, model: ModelCase, decision_renderer: DecisionRenderer, tokenizer: Tokenizer
    ) -> None:
        rendering = decision_renderer.render(MULTIWORD_DECISION)
        _, user = _split_system_user(cast(str, tokenizer.decode(rendering.tokens)), model)
        assert rendering.labels == MULTIWORD_DECISION.labels
        for label, description in MULTIWORD_DECISION.choices:
            assert label in user and description in user

    @pytest.mark.parametrize("decision", ALL_DECISIONS, ids=DECISION_IDS)
    def test_rendering_is_a_prefix_of_the_full_conversation(
        self, decision_renderer: DecisionRenderer, decision: Decision
    ) -> None:
        """Substituting any label's token at ``choice_pos`` yields a prefix of the chat format's
        own render of that reply."""
        rendering = decision_renderer.render(decision)
        choice_rendering = decision_renderer.choice_renderer.render_choices(decision.labels)
        for label in decision.labels:
            reply = cast(
                str, decision_renderer.tokenizer.decode(choice_rendering.answer_tokens_for(label))
            )
            full = decision_renderer._render_conversation(
                [
                    *decision_renderer._build_messages(decision, choice_rendering),
                    {"role": "assistant", "content": reply},
                ]
            )
            with_label = rendering.with_label(label)
            assert full[: len(with_label)] == with_label

    @pytest.mark.parametrize("effort", [0.0, 0.2])
    def test_inkling_effort_is_passed_to_the_renderer(
        self, inkling_renderer: Renderer, effort: float
    ) -> None:
        assert isinstance(inkling_renderer, TmlV0Renderer)
        decision_renderer = DecisionRenderer(inkling_renderer, effort=effort)
        rendering = decision_renderer.render(EXCHANGE_DECISION)
        choice_rendering = decision_renderer.choice_renderer.render_choices(
            EXCHANGE_DECISION.labels
        )
        messages = [
            *decision_renderer._build_messages(EXCHANGE_DECISION, choice_rendering),
            {
                "role": "assistant",
                "content": decision_renderer.tokenizer.decode(choice_rendering.answer_tokens),
            },
        ]
        model_input, _ = inkling_renderer.build_supervised_example(messages, effort=effort)
        assert model_input.to_ints()[: len(rendering.tokens)] == rendering.tokens
        other_effort, _ = inkling_renderer.build_supervised_example(messages, effort=0.5)
        assert other_effort.to_ints()[: len(rendering.tokens)] != rendering.tokens

    def test_default_choice_renderer_is_prefixed(
        self, inkling_decision_renderer: DecisionRenderer
    ) -> None:
        assert isinstance(inkling_decision_renderer.choice_renderer, PrefixedChoiceRenderer)
        assert inkling_decision_renderer.effort == 0.0

    def test_custom_choice_renderer_is_used(
        self, inkling_renderer: Renderer, inkling_tokenizer: Tokenizer
    ) -> None:
        hash_prefix = "#"

        class HashChoiceRenderer:
            """Writes ``#A`` / ``#B`` / ``#C``."""

            def render_choices(self, labels: Sequence[str]) -> ChoiceRendering:
                return ChoiceRendering(
                    answer_tokens=inkling_tokenizer.encode(hash_prefix + IDS[0]),
                    answer_position=1,
                    label_tokens={
                        label: inkling_tokenizer.encode(hash_prefix + letter)[-1]
                        for label, letter in zip(labels, IDS, strict=False)
                    },
                )

        rendering = DecisionRenderer(inkling_renderer, HashChoiceRenderer()).render(
            EXCHANGE_DECISION
        )
        text = cast(str, inkling_tokenizer.decode(rendering.tokens))
        assert text.endswith(MODELS["inkling"].reply_header + hash_prefix + IDS[0])
        assert PREFIX not in text
        assert rendering.choice_pos == len(rendering.tokens) - 1
        for label, letter in zip(EXCHANGE_DECISION.labels, IDS, strict=False):
            assert hash_prefix + letter in text
            assert (
                rendering.label_tokens[label] == inkling_tokenizer.encode(hash_prefix + letter)[-1]
            )

    def test_reply_not_found_raises(self, inkling_renderer: Renderer) -> None:
        class BrokenChoiceRenderer:
            def render_choices(self, labels: Sequence[str]) -> ChoiceRendering:
                return ChoiceRendering(
                    answer_tokens=[1, 2, 3],  # never appears in the render
                    answer_position=2,
                    label_tokens=dict.fromkeys(labels, 1),
                )

        with pytest.raises(RendererError):
            DecisionRenderer(inkling_renderer, BrokenChoiceRenderer()).render(EXCHANGE_DECISION)

    def test_label_tokens_keyed_by_other_labels_raise(self, inkling_renderer: Renderer) -> None:
        class MiskeyedChoiceRenderer:
            def render_choices(self, labels: Sequence[str]) -> ChoiceRendering:
                return PrefixedChoiceRenderer(inkling_renderer).render_choices(
                    [label.upper() for label in labels]
                )

        with pytest.raises(RendererError):
            DecisionRenderer(inkling_renderer, MiskeyedChoiceRenderer()).render(EXCHANGE_DECISION)


def test_kimi_k26_choice_rendering() -> None:
    kimi_tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
    kimi_renderer = get_renderer("kimi_k26", kimi_tokenizer)
    rendering = PrefixedChoiceRenderer(kimi_renderer, PREFIX).render_choices(
        DEPARTMENT_DECISION.labels
    )
    for label, letter in zip(DEPARTMENT_DECISION.labels, IDS, strict=True):
        assert rendering.answer_tokens_for(label) == list(kimi_tokenizer.encode(PREFIX + letter))
    rendering = DecisionRenderer(kimi_renderer).render(DEPARTMENT_DECISION)
    assert str(kimi_tokenizer.decode(rendering.tokens)).endswith(PREFIX + PLACEHOLDER)


# ---------------------------------------------------------------------------
# Deciding
# ---------------------------------------------------------------------------


class TestSampleDecision:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "decision, logprobs, expected",
        [
            # Unnormalized logprobs: the softmax is over the choice tokens only.
            (
                DEPARTMENT_DECISION,
                [math.log(0.02), math.log(0.07), math.log(0.01)],
                {"returns": 0.2, "shipping": 0.7, "billing": 0.1},
            ),
            (
                FRUSTRATION_DECISION,
                [math.log(1e-9), math.log(0.57), math.log(0.43)],
                {"0": 0.0, "1": 0.57, "2": 0.43},
            ),
            (EXCHANGE_DECISION, [math.log(0.93), math.log(0.07)], {"yes": 0.93, "no": 0.07}),
        ],
        ids=DECISION_IDS,
    )
    async def test_probabilities(
        self,
        inkling_decision_renderer: DecisionRenderer,
        decision: Decision,
        logprobs: list[float],
        expected: dict[str, float],
    ) -> None:
        probabilities = await sample_decision(
            _mock_sampling_client(logprobs), inkling_decision_renderer.render(decision)
        )
        assert list(probabilities) == decision.labels
        assert probabilities == pytest.approx(expected, abs=1e-6)

    @pytest.mark.asyncio
    async def test_request_scores_label_tokens_at_choice_pos(
        self, inkling_decision_renderer: DecisionRenderer
    ) -> None:
        client = _mock_sampling_client([0.0, 0.0])
        rendering = inkling_decision_renderer.render(EXCHANGE_DECISION)
        await sample_decision(client, rendering)
        kwargs = client.sample_async.await_args.kwargs
        assert kwargs["prompt"].to_ints() == rendering.tokens
        assert kwargs["sampling_params"].max_tokens == 1
        # Row i scores prompt position i + 1; only the answer row requests anything.
        target = kwargs["target_prompt_logprobs"]
        assert target.dtype == "int64"
        dense = target.to_torch(pad_value=-1)
        assert list(dense.shape) == [len(rendering.tokens) - 1, len(rendering.label_tokens)]
        assert dense[rendering.choice_pos - 1].tolist() == list(rendering.label_tokens.values())
        assert (dense[: rendering.choice_pos - 1] == -1).all()
        assert (dense[rendering.choice_pos :] == -1).all()

    @pytest.mark.asyncio
    async def test_missing_target_logprobs_raises(
        self, inkling_decision_renderer: DecisionRenderer
    ) -> None:
        client = MagicMock()
        client.sample_async = AsyncMock(return_value=MagicMock(target_prompt_logprobs=None))
        with pytest.raises(RuntimeError):
            await sample_decision(client, inkling_decision_renderer.render(EXCHANGE_DECISION))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decision, target",
    [(DEPARTMENT_DECISION, "billing"), (FRUSTRATION_DECISION, "2"), (EXCHANGE_DECISION, "yes")],
    ids=DECISION_IDS,
)
def test_decision_choice_datums_weight_only_the_answer_token(
    decision_renderer: DecisionRenderer, decision: Decision, target: str
) -> None:
    (datum,) = decision_choice_datums(decision_renderer, [(decision, target)])
    rendering = decision_renderer.render(decision)
    tokens, choice_pos, label_tokens = (
        rendering.tokens,
        rendering.choice_pos,
        rendering.label_tokens,
    )
    weights = datum.loss_fn_inputs["weights"].to_torch()
    targets = datum.loss_fn_inputs["target_tokens"].to_torch()
    # Right-shifted: the input is everything before the answer token, and the answer is the
    # only weighted target.
    assert datum.model_input.to_ints() == tokens[:choice_pos]
    assert weights.tolist() == [0.0] * (choice_pos - 1) + [1.0]
    assert targets[-1].item() == label_tokens[target]


def test_datums_are_built_in_order(inkling_decision_renderer: DecisionRenderer) -> None:
    examples = [(DEPARTMENT_DECISION, "billing"), (EXCHANGE_DECISION, "no")]
    datums = decision_choice_datums(inkling_decision_renderer, examples)
    assert len(datums) == 2
    for datum, (decision, target) in zip(datums, examples, strict=True):
        rendering = inkling_decision_renderer.render(decision)
        assert datum.model_input.to_ints() == rendering.tokens[: rendering.choice_pos]
        assert (
            datum.loss_fn_inputs["target_tokens"].to_torch()[-1].item()
            == (rendering.label_tokens[target])
        )
    dist_datums = decision_dist_datums(
        inkling_decision_renderer, [(decision, {target: 1.0}) for decision, target in examples]
    )
    assert [d.model_input.to_ints() for d in dist_datums] == [
        d.model_input.to_ints() for d in datums
    ]


def test_datums_reject_targets_outside_the_choices(
    inkling_decision_renderer: DecisionRenderer,
) -> None:
    with pytest.raises(DataValidationError):
        decision_choice_datums(inkling_decision_renderer, [(EXCHANGE_DECISION, "maybe")])
    with pytest.raises(DataValidationError):
        decision_dist_datums(inkling_decision_renderer, [(EXCHANGE_DECISION, {"maybe": 1.0})])
    with pytest.raises(DataValidationError):
        decision_dist_datums(
            inkling_decision_renderer, [(EXCHANGE_DECISION, {"yes": 1.5, "no": -0.5})]
        )


@pytest.mark.parametrize(
    "decision, target",
    [(DEPARTMENT_DECISION, "billing"), (FRUSTRATION_DECISION, "2"), (EXCHANGE_DECISION, "yes")],
    ids=DECISION_IDS,
)
def test_decision_dist_datums_target_every_choice_at_the_answer_row(
    decision_renderer: DecisionRenderer, decision: Decision, target: str
) -> None:
    (datum,) = decision_dist_datums(decision_renderer, [(decision, {target: 1.0})])
    rendering = decision_renderer.render(decision)
    tokens, choice_pos, label_tokens = (
        rendering.tokens,
        rendering.choice_pos,
        rendering.label_tokens,
    )
    labels = decision.labels
    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
    weights = datum.loss_fn_inputs["weights"].to_torch()

    assert datum.model_input.to_ints() == tokens[:choice_pos]
    assert target_tokens.shape == weights.shape == (choice_pos, len(labels))
    assert target_tokens[-1].tolist() == [label_tokens[la] for la in labels]
    assert (target_tokens[:-1] == 0).all() and (weights[:-1] == 0).all()
    assert weights[-1].tolist() == [1.0 if la == target else 0.0 for la in labels]


def test_decision_dist_datums_store_the_target_distribution_in_choice_order(
    inkling_decision_renderer: DecisionRenderer,
) -> None:
    (datum,) = decision_dist_datums(
        inkling_decision_renderer, [(DEPARTMENT_DECISION, {"billing": 0.25, "shipping": 0.75})]
    )
    weights = datum.loss_fn_inputs["weights"].to_torch()
    # choices are [returns, shipping, billing]; omitted "returns" gets 0
    assert weights[-1].tolist() == pytest.approx([0.0, 0.75, 0.25])
    assert (weights[:-1] == 0).all()


def _fake_logprobs(n_positions: int, answer_row_logprobs: list[float]) -> torch.Tensor:
    logprobs = torch.full((n_positions, len(answer_row_logprobs)), -5.0)
    logprobs[-1] = torch.tensor(answer_row_logprobs)
    return logprobs.requires_grad_(True)


def test_probability_loss_matches_cross_entropy_over_choices(
    inkling_decision_renderer: DecisionRenderer,
) -> None:
    datums = decision_dist_datums(
        inkling_decision_renderer,
        [
            (DEPARTMENT_DECISION, {"shipping": 1.0}),
            (EXCHANGE_DECISION, {"no": 1.0}),
        ],
    )
    logprobs = [
        _fake_logprobs(datums[0].model_input.length, [math.log(0.2), math.log(0.7), math.log(0.1)]),
        _fake_logprobs(datums[1].model_input.length, [math.log(0.25), math.log(0.75)]),
    ]
    loss, metrics = probability_loss(lambda probs, target: -(target * torch.log(probs)).sum())(
        datums, logprobs
    )

    assert loss.item() == pytest.approx((-math.log(0.7) - math.log(0.75)) / 2)
    assert metrics["target_prob:mean"] == pytest.approx((0.7 + 0.75) / 2)
    assert metrics["accuracy"] == 1.0
    loss.backward()
    # Gradient flows only into the answer row, and pushes the labelled choice up.
    for lp, column in zip(logprobs, [1, 1], strict=True):
        assert lp.grad is not None
        assert (lp.grad[:-1] == 0).all()
        assert lp.grad[-1, column] < 0
        assert lp.grad[-1].sum().item() == pytest.approx(0.0, abs=1e-6)


def test_probability_loss_with_a_squared_error_loss(
    inkling_decision_renderer: DecisionRenderer,
) -> None:
    def squared_error(choice_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        return ((choice_probs - target_probs) ** 2).sum()

    datums = decision_dist_datums(inkling_decision_renderer, [(FRUSTRATION_DECISION, {"0": 1.0})])
    probs = [0.1, 0.6, 0.3]
    logprobs = [_fake_logprobs(datums[0].model_input.length, [math.log(p) for p in probs])]
    loss, metrics = probability_loss(squared_error)(datums, logprobs)
    assert loss.item() == pytest.approx((1 - 0.1) ** 2 + 0.6**2 + 0.3**2)
    assert metrics["target_prob:mean"] == pytest.approx(0.1)
    assert metrics["accuracy"] == 0.0  # "1" is the argmax, "0" is labelled
