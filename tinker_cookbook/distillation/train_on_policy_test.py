import asyncio
import math
from concurrent.futures import Future
from types import SimpleNamespace
from typing import TypeVar, cast

import chz
import pytest
import tinker
import torch
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.distillation import sdft, train_on_policy
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.rl.types import (
    PARSE_ERROR_MASKED_METRIC_KEY,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

T = TypeVar("T")

# Raw teacher logprobs for the three entries returned at every position. They are not
# normalized, so the trainer must renormalize them over the returned set.
TEACHER_RAW_LPS = [-0.2, -1.0, -3.0]
VOCAB_SIZE = 10_000


def _done(value: T) -> AwaitableConcurrentFuture[T]:
    future: Future[T] = Future()
    future.set_result(value)
    return AwaitableConcurrentFuture(future)


class FakeTeacher:
    """Returns, at each position, the token actually there plus two others (+1000, +2000)."""

    def __init__(self, short_rows: tuple[int, ...] = (), missing_rows: tuple[int, ...] = ()):
        self.short_rows = short_rows
        self.missing_rows = missing_rows
        self.sample_calls: list[list[int]] = []
        self.logprob_calls: list[list[int]] = []

    def sample(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
    ) -> AwaitableConcurrentFuture[SimpleNamespace]:
        assert num_samples == 1 and sampling_params.max_tokens == 1 and include_prompt_logprobs
        tokens = prompt.to_ints()
        self.sample_calls.append(tokens)
        rows: list[list[tuple[int, float]] | None] = [None]
        for row, token in enumerate(tokens[1:], start=1):
            if row in self.missing_rows:
                rows.append(None)
                continue
            entries = [(token + 1000 * j, lp) for j, lp in enumerate(TEACHER_RAW_LPS)]
            rows.append(entries[:2] if row in self.short_rows else entries[:topk_prompt_logprobs])
        return _done(SimpleNamespace(topk_prompt_logprobs=rows))

    def compute_logprobs(self, prompt: tinker.ModelInput) -> AwaitableConcurrentFuture[list]:
        tokens = prompt.to_ints()
        self.logprob_calls.append(tokens)
        return _done([None, *[-0.5] * (len(tokens) - 1)])


class FakeTokenizer:
    def __len__(self) -> int:
        return VOCAB_SIZE

    def decode(self, tokens: list[int]) -> str:
        return " ".join(map(str, tokens))


def _transition(ob: list[int], ac: list[int], done: bool, masked: bool = False) -> Transition:
    return Transition(
        ob=tinker.ModelInput.from_ints(ob),
        ac=TokensWithLogprobs(tokens=ac, maybe_logprobs=[-1.0] * len(ac)),
        reward=0.0,
        episode_done=done,
        metrics={PARSE_ERROR_MASKED_METRIC_KEY: 1.0} if masked else {},
    )


def _single_turn(prompt: list[int], completion: list[int]) -> Trajectory:
    return Trajectory(
        transitions=[_transition(prompt, completion, True)], final_ob=tinker.ModelInput.empty()
    )


def _group(*trajectories: Trajectory) -> TrajectoryGroup:
    n = len(trajectories)
    return TrajectoryGroup(list(trajectories), [0.0] * n, [{} for _ in range(n)])


def _builders(n: int) -> list[EnvGroupBuilder]:
    return [cast(EnvGroupBuilder, SimpleNamespace(logging_tags=lambda: ["test"]))] * n


def _prepare(
    groups: list[TrajectoryGroup],
    dataset_indices_P: list[int],
    teachers: list[FakeTeacher],
    teacher_topk: int,
) -> tuple[list[tinker.Datum], dict]:
    return asyncio.run(
        train_on_policy.prepare_minibatch(
            _builders(len(groups)),
            groups,
            cast(Tokenizer, FakeTokenizer()),
            dataset_indices_P,
            [cast(tinker.SamplingClient, teacher) for teacher in teachers],
            kl_penalty_coef=1.0,
            kl_discount_factor=0.0,
            teacher_topk=teacher_topk,
        )
    )


def _expected_row(token: int, k: int) -> tuple[list[int], torch.Tensor]:
    lps = torch.tensor(TEACHER_RAW_LPS[:k])
    return [token + 1000 * j for j in range(k)], torch.softmax(lps, dim=0)


def test_topk_targets_follow_the_sampled_sequence_and_route_per_teacher():
    groups = [
        _group(_single_turn([1, 2, 3], [10, 11, 12]), _single_turn([1, 2, 3], [13, 14])),
        _group(_single_turn([4, 5], [20, 21, 22])),
    ]
    teachers = [FakeTeacher(), FakeTeacher()]
    data_D, metrics = _prepare(groups, [0, 1], teachers, teacher_topk=3)

    assert teachers[0].sample_calls == [[1, 2, 3, 10, 11, 12], [1, 2, 3, 13, 14]]
    assert teachers[1].sample_calls == [[4, 5, 20, 21, 22]]
    assert all(not teacher.logprob_calls for teacher in teachers)

    expected_completions = [[10, 11, 12], [13, 14], [20, 21, 22]]
    for datum, completion in zip(data_D, expected_completions, strict=True):
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        weights = datum.loss_fn_inputs["weights"].to_torch()
        n = datum.model_input.length
        prompt_rows = n - len(completion)
        assert targets.shape == weights.shape == (n, 3)
        assert set(datum.loss_fn_inputs) == {"target_tokens", "weights"}
        assert weights[:prompt_rows].abs().sum().item() == 0.0
        for row, token in enumerate(completion, start=prompt_rows):
            expected_tokens, expected_weights = _expected_row(token, 3)
            assert targets[row].tolist() == expected_tokens
            assert torch.allclose(weights[row], expected_weights)
            assert weights[row].sum().item() == pytest.approx(1.0)

    assert metrics["opd/topk"] == 3.0
    assert metrics["opd/topk_positions"] == 8.0
    expected_entropy = -(_expected_row(0, 3)[1] * _expected_row(0, 3)[1].log()).sum().item()
    assert metrics["opd/mean_teacher_entropy"] == pytest.approx(expected_entropy)
    assert "teacher_kl" not in metrics


def test_fewer_than_k_teacher_entries_leave_unused_slots_at_zero():
    teacher = FakeTeacher(short_rows=(4,))
    (datum,), _ = _prepare([_group(_single_turn([1, 2, 3], [10, 11]))], [0], [teacher], 3)

    weights = datum.loss_fn_inputs["weights"].to_torch()
    targets = datum.loss_fn_inputs["target_tokens"].to_torch()
    # Row 3 predicts the token at teacher row 4, which only has two entries.
    _, expected = _expected_row(11, 2)
    assert targets[3, :2].tolist() == [11, 1011]
    assert torch.allclose(weights[3, :2], expected)
    assert weights[3, 2].item() == 0.0
    assert weights[2].sum().item() == pytest.approx(1.0)


def test_split_multi_turn_trajectory_uses_each_datums_own_prefix():
    # The second observation does not extend the first turn, so the trajectory becomes two
    # datums; each must be scored on its own sequence, not the first datum's prompt.
    trajectory = Trajectory(
        transitions=[_transition([1, 2], [10, 11], False), _transition([5, 6], [20], True)],
        final_ob=tinker.ModelInput.empty(),
    )
    teacher = FakeTeacher()
    data_D, _ = _prepare([_group(trajectory)], [0], [teacher], teacher_topk=2)

    assert len(data_D) == 2
    assert teacher.sample_calls == [[1, 2, 10, 11], [5, 6, 20]]
    second_targets = data_D[1].loss_fn_inputs["target_tokens"].to_torch()
    second_weights = data_D[1].loss_fn_inputs["weights"].to_torch()
    assert second_weights[0].sum().item() == 0.0
    assert second_targets[1].tolist() == [20, 1020]


def test_observation_gaps_missing_rows_and_masked_datums_get_zero_weight():
    # One datum: prompt [1, 2], action [10, 11], observation [20], action [30, 31].
    multi_turn = Trajectory(
        transitions=[
            _transition([1, 2], [10, 11], False),
            _transition([1, 2, 10, 11, 20], [30, 31], True),
        ],
        final_ob=tinker.ModelInput.empty(),
    )
    masked = Trajectory(
        transitions=[_transition([1, 2], [40, 41], True, masked=True)],
        final_ob=tinker.ModelInput.empty(),
    )
    # Teacher row 6 predicts the final token 31.
    teacher = FakeTeacher(missing_rows=(6,))
    (datum, masked_datum), metrics = _prepare(
        [_group(multi_turn, masked)], [0], [teacher], teacher_topk=2
    )

    assert teacher.sample_calls[0] == [1, 2, 10, 11, 20, 30, 31]
    targets = datum.loss_fn_inputs["target_tokens"].to_torch()
    weights = datum.loss_fn_inputs["weights"].to_torch()
    row_sums = weights.sum(dim=-1)
    assert row_sums.tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    assert [targets[row, 0].item() for row in (1, 2, 4)] == [10, 11, 30]
    assert masked_datum.loss_fn_inputs["weights"].to_torch().abs().sum().item() == 0.0
    assert metrics["opd/topk_positions"] == 3.0


def test_teacher_topk_zero_keeps_sampled_token_kl_advantages():
    teacher = FakeTeacher()
    (datum,), metrics = _prepare([_group(_single_turn([1, 2, 3], [10, 11]))], [0], [teacher], 0)

    assert teacher.logprob_calls == [[1, 2, 3, 10, 11]]
    assert teacher.sample_calls == []
    # advantage = -(student_lp - teacher_lp) = -(-1.0 - -0.5) on trained tokens
    assert datum.loss_fn_inputs["advantages"].to_torch().tolist() == [0.0, 0.0, 0.5, 0.5]
    assert metrics["teacher_kl"] == pytest.approx(-0.5)
    assert not any(key.startswith("opd/") for key in metrics)


def _custom_loss_grad(
    logits_NV: torch.Tensor, targets_NK: torch.Tensor, weights_NK: torch.Tensor
) -> torch.Tensor:
    logits = logits_NV.clone().requires_grad_(True)
    student_logp_NK = torch.log_softmax(logits, dim=-1).gather(-1, targets_NK)
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(len(targets_NK)))),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(targets_NK),
            "weights": tinker.TensorData.from_torch(weights_NK),
        },
    )
    loss, _ = sdft.reverse_kl_custom_loss([datum], [student_logp_NK])
    loss.backward()
    assert logits.grad is not None
    return logits.grad


def _renormalized_kl_grad(
    logits_NV: torch.Tensor, support_NK: torch.Tensor, teacher_NK: torch.Tensor
) -> torch.Tensor:
    logits = logits_NV.clone().requires_grad_(True)
    log_p = torch.log_softmax(torch.log_softmax(logits, dim=-1).gather(-1, support_NK), dim=-1)
    log_q = teacher_NK.log()
    (log_p.exp() * (log_p - log_q)).sum().backward()
    assert logits.grad is not None
    return logits.grad


def test_full_support_gradient_matches_true_reverse_kl():
    torch.manual_seed(0)
    n, vocab = 3, 6
    logits = torch.randn(n, vocab)
    teacher = torch.softmax(torch.randn(n, vocab), dim=-1)
    targets = torch.stack([torch.randperm(vocab) for _ in range(n)])
    weights = teacher.gather(-1, targets)

    grad = _custom_loss_grad(logits, targets, weights)

    true_logits = logits.clone().requires_grad_(True)
    log_p = torch.log_softmax(true_logits, dim=-1)
    (log_p.exp() * (log_p - teacher.log())).sum().backward()
    assert true_logits.grad is not None
    assert torch.allclose(grad, true_logits.grad, atol=1e-6)


def test_two_token_gradient_matches_renormalized_kl_and_masks_rows():
    torch.manual_seed(1)
    logits = torch.randn(3, 5)
    targets = torch.tensor([[0, 0], [3, 1], [4, 2]])
    teacher = torch.tensor([[0.0, 0.0], [0.7, 0.3], [0.4, 0.6]])

    grad = _custom_loss_grad(logits, targets, teacher)

    assert grad[0].abs().sum().item() == 0.0
    expected = _renormalized_kl_grad(logits[1:], targets[1:], teacher[1:])
    assert torch.allclose(grad[1:], expected, atol=1e-6)
    assert grad[1:].abs().sum().item() > 0


BASE_CONFIG = train_on_policy.Config(
    learning_rate=1e-4,
    dataset_configs=[],
    model_name="student",
    recipe_name="test",
    max_tokens=8,
    log_path="unused",
)


@pytest.mark.parametrize(
    "overrides",
    [
        {"teacher_topk": -1},
        {"teacher_topk": 1},
        {"teacher_topk": 20, "kl_penalty_coef": 0.5},
        {"teacher_topk": 20, "kl_discount_factor": 0.9},
        {"teacher_topk": 20, "loss_fn": "ppo"},
        {"teacher_topk": 20, "loss_fn_config": {"clip_low_threshold": 0.8}},
        {"teacher_topk": 20, "compute_post_kl": True},
    ],
)
def test_config_rejects_settings_topk_would_ignore(overrides: dict[str, object]):
    with pytest.raises(ConfigurationError):
        chz.replace(BASE_CONFIG, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"teacher_topk": 20},
        {"teacher_topk": 0, "kl_penalty_coef": 0.5, "kl_discount_factor": 0.9},
        {"teacher_topk": 0, "loss_fn": "ppo", "compute_post_kl": True},
    ],
)
def test_config_accepts_used_settings(overrides: dict[str, object]):
    chz.replace(BASE_CONFIG, **overrides)


class FakeTrainingClient:
    """Runs the custom loss on a uniform student, like the SDK does client-side."""

    def __init__(self) -> None:
        self.custom_batches: list[list[tinker.Datum]] = []
        self.optim_steps = 0
        self.saves = 0

    def forward_backward_custom(self, data, loss_fn):
        self.custom_batches.append(data)
        logprobs = [
            torch.full(
                datum.loss_fn_inputs["target_tokens"].to_torch().shape, -math.log(VOCAB_SIZE)
            )
            for datum in data
        ]
        _, metrics = loss_fn(data, logprobs)
        return _done(SimpleNamespace(metrics=metrics))

    def optim_step(self, adam_params: tinker.AdamParams):
        self.optim_steps += 1
        return _done(SimpleNamespace(metrics={}))

    def save_weights_and_get_sampling_client(self, record_stability_info: bool, kind: str):
        self.saves += 1
        return _done("new-sampling-client")


def test_train_step_with_topk_uses_custom_reverse_kl_loss():
    training_client = FakeTrainingClient()
    client = cast(tinker.TrainingClient, training_client)
    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=client,
        service_client=cast(tinker.ServiceClient, None),
        log_path="unused",
    )
    teachers = [FakeTeacher(), FakeTeacher()]

    sampling_client, metrics = asyncio.run(
        train_on_policy.do_train_step_and_get_sampling_client(
            chz.replace(BASE_CONFIG, teacher_topk=3),
            0,
            client,
            checkpoint_mgr,
            cast(tinker.ServiceClient, None),
            cast(Tokenizer, FakeTokenizer()),
            _builders(2),
            [_group(_single_turn([1, 2], [10, 11])), _group(_single_turn([3, 4], [12]))],
            [0, 1],
            [cast(tinker.SamplingClient, teacher) for teacher in teachers],
        )
    )

    assert sampling_client == "new-sampling-client"
    assert training_client.saves == 1
    assert training_client.optim_steps == 1
    assert [len(batch) for batch in training_client.custom_batches] == [2]
    assert [len(teacher.sample_calls) for teacher in teachers] == [1, 1]

    # A uniform student renormalized over the 3 teacher tokens has p = 1/3 everywhere.
    teacher_probs = _expected_row(0, 3)[1]
    expected_kl = (1 / 3 * (math.log(1 / 3) - teacher_probs.log())).sum().item()
    assert metrics["opd/reverse_kl_mean"] == pytest.approx(expected_kl, rel=1e-5)
    assert metrics["opd/reverse_kl_positions"] == 3.0
    assert not any(key.startswith("sdft/") for key in metrics)
    assert "teacher_kl" not in metrics
