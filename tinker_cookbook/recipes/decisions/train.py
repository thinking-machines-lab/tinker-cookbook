"""Training datums for decisions, rendered by the same :class:`DecisionRenderer` used to sample.

Tensor suffixes: T = token positions, K = choices, D = datums.

Cross-entropy on the correct choice::

    datums = decision_choice_datums(decision_renderer, [(decision, "billing"), ...])
    fwd_bwd_future = await training_client.forward_backward_async(datums, loss_fn="cross_entropy")
    optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
    await fwd_bwd_future.result_async()
    await optim_future.result_async()

A custom loss on the probability distribution over choices::

    datums = decision_dist_datums(decision_renderer, [(decision, {"billing": 1.0}), ...])
    fwd_bwd_future = await training_client.forward_backward_custom_async(
        datums, probability_loss(lambda probs, target: ((probs - target) ** 2).sum())
    )
    optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
    await fwd_bwd_future.result_async()
    await optim_future.result_async()
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import tinker
import torch

from tinker_cookbook.exceptions import DataValidationError
from tinker_cookbook.recipes.decisions.decision import Decision, DecisionRenderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.utils.misc_utils import safezip

ProbabilityLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Scalar loss from ``(choice_probs, target_probs)``, both shape ``[K]`` in choice order; must
be differentiable in torch."""

CustomLossFn = Callable[
    [list[tinker.Datum], list[torch.Tensor]], tuple[torch.Tensor, dict[str, float]]
]
"""The signature ``TrainingClient.forward_backward_custom`` expects."""


def _check_label(decision: Decision, label: str) -> None:
    if label not in decision.labels:
        raise DataValidationError(f"target must be one of {decision.labels}, got {label!r}")


def decision_choice_datums(
    decision_renderer: DecisionRenderer, examples: Sequence[tuple[Decision, str]]
) -> list[tinker.Datum]:
    """One ``cross_entropy`` datum per ``(decision, correct label)`` pair, weighting only the
    answer token. Raises :class:`DataValidationError` for a label outside the choices."""
    datums = []
    for decision, target in examples:
        _check_label(decision, target)
        rendering = decision_renderer.render(decision)
        tokens_T = rendering.with_label(target)
        weights_T = torch.zeros(len(tokens_T), dtype=torch.float32)
        weights_T[rendering.choice_pos] = 1.0
        datums.append(
            datum_from_model_input_weights(tinker.ModelInput.from_ints(tokens_T), weights_T)
        )
    return datums


def decision_dist_datums(
    decision_renderer: DecisionRenderer,
    examples: Sequence[tuple[Decision, Mapping[str, float]]],
) -> list[tinker.Datum]:
    """One datum for :func:`probability_loss` per ``(decision, target_probs)`` pair.

    ``target_probs`` maps labels to non-negative target probabilities; ``{label: 1.0}`` is a
    single correct answer and omitted labels get 0. Raises :class:`DataValidationError` for an
    unknown label or a negative probability.
    """
    datums = []
    for decision, target_probs in examples:
        for label, prob in target_probs.items():
            _check_label(decision, label)
            if prob < 0:
                raise DataValidationError(f"target probability for {label!r} is negative: {prob}")
        rendering = decision_renderer.render(decision)
        labels = decision.labels
        # The input stops before the answer position; the last row of each (T, K) tensor holds
        # the K choice tokens and their target probabilities.
        n_positions = rendering.choice_pos
        target_tokens_T_K = torch.zeros((n_positions, len(labels)), dtype=torch.int64)
        target_tokens_T_K[-1] = torch.tensor([rendering.label_tokens[label] for label in labels])
        weights_T_K = torch.zeros((n_positions, len(labels)), dtype=torch.float32)
        weights_T_K[-1] = torch.tensor([target_probs.get(label, 0.0) for label in labels])
        datums.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(rendering.tokens[:n_positions]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_torch(target_tokens_T_K),
                    "weights": tinker.TensorData.from_torch(weights_T_K),
                },
            )
        )
    return datums


def probability_loss(loss_fn: ProbabilityLossFn) -> CustomLossFn:
    """A ``forward_backward_custom`` loss from a per-example loss on the choice distribution.

    ``loss_fn(choice_probs, target_probs)`` gets the model's probabilities over the choices and
    the datum's targets, both in choice order; the batch loss is the mean. Metrics: ``loss``,
    ``target_prob:mean`` and ``accuracy`` (argmax agreement).
    """

    def custom_loss(
        data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses_D = []
        target_probs_D = []
        correct = 0
        for datum, logprobs_T_K in safezip(data, logprobs_list):
            target_probs_K = datum.loss_fn_inputs["weights"].to_torch()[-1].float()
            choice_probs_K = torch.softmax(logprobs_T_K[-1].float(), dim=-1)
            losses_D.append(loss_fn(choice_probs_K, target_probs_K))
            target_probs_D.append((choice_probs_K * target_probs_K).sum().detach())
            correct += int(choice_probs_K.argmax().item() == target_probs_K.argmax().item())
        loss = torch.stack(losses_D).mean()
        return loss, {
            "loss": float(loss.item()),
            "target_prob:mean": float(torch.stack(target_probs_D).mean().item()),
            "accuracy": correct / len(data),
        }

    return custom_loss
