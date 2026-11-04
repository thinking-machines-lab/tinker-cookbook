import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, cast

import pytest
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.distillation import sdft
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import (
    PARSE_ERROR_MASKED_METRIC_KEY,
    Trajectory,
    Transition,
)

Mode = Literal["scalar", "forward", "reverse"]


@dataclass(frozen=True)
class Scenario:
    name: str
    actions: list[list[int]]
    observations: list[list[int]]
    teacher_suffix: list[int]
    masked_turns: tuple[int, ...] = ()


SCENARIOS = [
    Scenario("contiguous", [[10, 11, 12, 13, 14, 15]], [[]], [10, 11, 12, 13, 14, 15]),
    Scenario("gap", [[10, 11, 12, 13], [30, 31]], [[], [20]], [10, 11, 12, 13, 20, 30, 31]),
    Scenario("wide-gap", [[10, 11], [30, 31]], [[], [20, 21, 22]], [10, 11, 20, 21, 22, 30, 31]),
    Scenario(
        "multiple-gaps", [[10], [30], [50, 51]], [[], [20, 21], [40]], [10, 20, 21, 30, 40, 50, 51]
    ),
    Scenario(
        "masked-action", [[10], [30, 31], [50]], [[], [20], [40]], [10, 20, 30, 31, 40, 50], (1,)
    ),
    Scenario("leading-mask", [[10, 11], [30, 31]], [[], [20]], [30, 31], (0,)),
    Scenario("trailing-mask", [[10, 11], [30]], [[], [20]], [10, 11, 20, 30], (1,)),
    Scenario("all-masked", [[10, 11]], [[]], [], (0,)),
]
GAP_BEFORE_SKIP = Scenario(
    "gap-before-skip",
    [[10, 11], [30, 31, 32, 33]],
    [[], [20, 21]],
    [10, 11, 20, 21, 30, 31, 32, 33],
)


def _datum(scenario: Scenario) -> tinker.Datum:
    history = [1, 2]
    transitions = []
    for i, (action, observation) in enumerate(
        zip(scenario.actions, scenario.observations, strict=True)
    ):
        history = [*history, *observation]
        transitions.append(
            Transition(
                ob=tinker.ModelInput.from_ints(history),
                ac=TokensWithLogprobs(tokens=action, maybe_logprobs=[-1.0] * len(action)),
                reward=0.0,
                episode_done=i == len(scenario.actions) - 1,
                metrics={PARSE_ERROR_MASKED_METRIC_KEY: 1.0} if i in scenario.masked_turns else {},
            )
        )
        history = [*history, *action]
    (datum,) = trajectory_to_data(
        Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty()), traj_advantage=0.0
    )
    return datum


class RecordingTeacher:
    def __init__(self, *, response_limit: int | None = None, missing_row: int | None = None):
        self.response_limit = response_limit
        self.missing_row = missing_row
        self.calls: list[list[int]] = []
        self.widths: list[int] = []

    async def compute_logprobs_async(self, prompt: tinker.ModelInput) -> list[float | None]:
        tokens = prompt.to_ints()
        self.calls.append(tokens)
        rows: list[float | None] = [None, *[-float(token) for token in tokens[1:]]]
        if self.missing_row is not None:
            rows[self.missing_row] = None
        return rows if self.response_limit is None else rows[: self.response_limit]

    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
    ) -> SimpleNamespace:
        tokens = prompt.to_ints()
        self.calls.append(tokens)
        self.widths.append(topk_prompt_logprobs)
        assert num_samples == 1 and sampling_params.max_tokens == 1 and include_prompt_logprobs
        rows: list[list[tuple[int, float]] | None] = [
            None,
            *[[(token + 1000, 0.0)] for token in tokens[1:]],
        ]
        if self.missing_row is not None:
            rows[self.missing_row] = None
        return SimpleNamespace(
            topk_prompt_logprobs=rows
            if self.response_limit is None
            else rows[: self.response_limit]
        )


def _score(
    mode: Mode,
    datum: tinker.Datum,
    teacher: RecordingTeacher,
    *,
    limit: int = 100,
    skip: int = 0,
    use_defaults: bool = False,
) -> tuple[tinker.Datum, dict[str, float]]:
    prompt = tinker.ModelInput.from_ints([90, 91])
    client = cast(tinker.SamplingClient, teacher)
    if mode == "scalar":
        metrics = asyncio.run(
            sdft.compute_sdft_advantages(
                [datum], [{"group_idx": 0}], client, [prompt], max_context_length=limit + 2
            )
        )
        return datum, metrics
    builder = (
        sdft.build_topk_distillation_datums if mode == "forward" else sdft.build_reverse_kl_datums
    )
    if use_defaults:
        datums, metrics = asyncio.run(builder([datum], [{"group_idx": 0}], client, [prompt]))
    else:
        datums, metrics = asyncio.run(
            builder(
                [datum],
                [{"group_idx": 0}],
                client,
                [prompt],
                topk=2,
                max_context_length=limit + 2,
                skip_first_n_tokens=skip,
            )
        )
    return datums[0], metrics


def _assert_scores(
    mode: Mode,
    original: tinker.Datum,
    scored: tinker.Datum,
    teacher: RecordingTeacher,
    suffix: list[int],
    *,
    limit: int = 100,
    skip: int = 0,
    width: int = 2,
) -> None:
    retained = suffix[:limit]
    full_teacher_input = [90, 91, *retained]
    assert teacher.calls == [full_teacher_input]
    mask = original.loss_fn_inputs["mask"].to_torch().tolist()
    tokens = original.loss_fn_inputs["target_tokens"].to_torch().tolist()
    advantages: list[float] = []
    targets = scored.loss_fn_inputs["target_tokens"].to_torch()
    weights = None
    if mode == "scalar":
        advantages = scored.loss_fn_inputs["advantages"].to_torch().tolist()
    else:
        weights = scored.loss_fn_inputs["weights"].to_torch()
        assert list(targets.shape) == list(weights.shape) == [len(tokens), width]
        assert teacher.widths == [width]
    ordinal = 0
    for i, (token, learned) in enumerate(zip(tokens, mask, strict=True)):
        active = learned > 0 and token in retained
        skipped = mode != "scalar" and ordinal < skip
        if learned > 0:
            ordinal += 1
        row = full_teacher_input.index(token) if active else -1
        absent = teacher.response_limit is not None and row >= teacher.response_limit
        missing = row == teacher.missing_row
        if mode == "scalar":
            expected = (1.0 if missing else 1.0 - token) if active and not absent else 0.0
            assert advantages[i] == expected
        elif active and not skipped and not absent and not missing:
            assert weights is not None
            assert targets[i, 0].item() == token + 1000
            assert weights[i, 0].item() == 1.0
            assert weights[i, 1:].sum().item() == 0.0
        else:
            assert weights is not None
            assert weights[i].sum().item() == 0.0
    assert scored.model_input.to_ints() == original.model_input.to_ints()


@pytest.mark.parametrize("mode", ["scalar", "forward", "reverse"])
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.name for scenario in SCENARIOS])
def test_teacher_rows_follow_physical_token_positions(mode: Mode, scenario: Scenario):
    datum = _datum(scenario)
    teacher = RecordingTeacher()
    scored, _ = _score(mode, datum, teacher)
    _assert_scores(mode, datum, scored, teacher, scenario.teacher_suffix)


@pytest.mark.parametrize("mode", ["scalar", "forward", "reverse"])
@pytest.mark.parametrize("limit", range(8))
def test_context_cutoff_does_not_train_actions_past_an_observation(mode: Mode, limit: int):
    scenario = SCENARIOS[1]
    datum = _datum(scenario)
    teacher = RecordingTeacher()
    scored, metrics = _score(mode, datum, teacher, limit=limit)
    _assert_scores(mode, datum, scored, teacher, scenario.teacher_suffix, limit=limit)
    included = sum(
        token in scenario.teacher_suffix[:limit] for action in scenario.actions for token in action
    )
    if mode == "forward" and included:
        assert metrics["sdft/total_completion_tokens"] == included
    if mode == "reverse" and included:
        assert metrics["sdft/total_completion_positions"] == included


@pytest.mark.parametrize("limit", [5, 7])
def test_scalar_metrics_use_selected_action_rows_without_changing_denominator(limit: int):
    scenario = SCENARIOS[1]
    datum = _datum(scenario)
    teacher = RecordingTeacher()
    _, metrics = _score("scalar", datum, teacher, limit=limit)
    selected = [
        token
        for action in scenario.actions
        for token in action
        if token in scenario.teacher_suffix[:limit]
    ]
    denominator = sum(len(action) for action in scenario.actions)
    assert metrics["sdft/mean_teacher_lp"] == pytest.approx(-sum(selected) / denominator)
    assert metrics["sdft/mean_student_lp"] == pytest.approx(-len(selected) / denominator)
    assert metrics["sdft/mean_advantage"] == pytest.approx(
        sum(1 - token for token in selected) / denominator
    )


@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize(
    "scenario",
    [SCENARIOS[0], SCENARIOS[1], GAP_BEFORE_SKIP],
    ids=["contiguous", "gap", "gap-before-skip"],
)
def test_default_width_and_skip_preserve_action_ordinal(mode: Mode, scenario: Scenario):
    datum = _datum(scenario)
    teacher = RecordingTeacher()
    scored, _ = _score(mode, datum, teacher, use_defaults=True)
    _assert_scores(mode, datum, scored, teacher, scenario.teacher_suffix, skip=3, width=20)


@pytest.mark.parametrize("mode", ["scalar", "forward", "reverse"])
@pytest.mark.parametrize("missing", [False, True], ids=["short-response", "missing-row"])
def test_unavailable_teacher_rows_do_not_shift_later_actions(mode: Mode, missing: bool):
    scenario = SCENARIOS[1]
    teacher = RecordingTeacher(missing_row=7) if missing else RecordingTeacher(response_limit=7)
    datum = _datum(scenario)
    scored, _ = _score(mode, datum, teacher)
    _assert_scores(mode, datum, scored, teacher, scenario.teacher_suffix)
