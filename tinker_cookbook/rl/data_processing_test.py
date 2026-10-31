"""Tests for the provenance spans that trajectory_to_data records."""

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import PARSE_ERROR_MASKED_METRIC_KEY, Trajectory, Transition

Prompt = tinker.PromptProvenanceSpan
Sampled = tinker.SampledProvenanceSpan


def _transition(
    ob_tokens: list[int],
    ac_tokens: list[int],
    sequence_id: str | None,
    metrics: dict[str, float] | None = None,
) -> Transition:
    return Transition(
        ob=tinker.ModelInput.from_ints(ob_tokens),
        ac=TokensWithLogprobs(
            tokens=ac_tokens, maybe_logprobs=[-1.0] * len(ac_tokens), sequence_id=sequence_id
        ),
        reward=0.0,
        episode_done=False,
        metrics=metrics or {},
    )


def _data(*transitions: Transition, include_provenance: bool = True) -> list[tinker.Datum]:
    return trajectory_to_data(
        Trajectory(transitions=list(transitions), final_ob=tinker.ModelInput.empty()),
        traj_advantage=1.0,
        include_provenance=include_provenance,
    )


class TestProvenanceSpans:
    def test_single_turn(self):
        # Prompt [1, 2, 3], sampled [4, 5]: the full sequence is 1 2 3 4 5, the
        # model input is 1 2 3 4 and the targets are 2 3 4 5.
        (datum,) = _data(_transition([1, 2, 3], [4, 5], "a"))
        assert datum.model_input_spans == [
            Prompt(sequence_id="a", offset=0, length=3),
            Sampled(sequence_id="a", offset=0, length=1),
        ]
        assert datum.loss_fn_input_spans == [
            Prompt(sequence_id="a", offset=1, length=2),
            Sampled(sequence_id="a", offset=0, length=2),
        ]

    def test_multi_turn_prefix_extension_merges_into_one_datum(self):
        # Turn 2's prompt is turn 1's prompt + action + 2 new tokens, so only the
        # new tokens (positions 5-6 of turn 2's prompt) are attributed to turn 2.
        (datum,) = _data(
            _transition([1, 2, 3], [4, 5], "a"),
            _transition([1, 2, 3, 4, 5, 6, 7], [8, 9, 10], "b"),
        )
        assert datum.model_input_spans == [
            Prompt(sequence_id="a", offset=0, length=3),
            Sampled(sequence_id="a", offset=0, length=2),
            Prompt(sequence_id="b", offset=5, length=2),
            Sampled(sequence_id="b", offset=0, length=2),
        ]
        assert datum.loss_fn_input_spans == [
            Prompt(sequence_id="a", offset=1, length=2),
            Sampled(sequence_id="a", offset=0, length=2),
            Prompt(sequence_id="b", offset=5, length=2),
            Sampled(sequence_id="b", offset=0, length=3),
        ]

    def test_non_prefix_turn_starts_a_new_datum(self):
        first, second = _data(
            _transition([1, 2], [3], "a"),
            _transition([9, 9, 9], [4, 5], "b"),
        )
        assert first.loss_fn_input_spans == [
            Prompt(sequence_id="a", offset=1, length=1),
            Sampled(sequence_id="a", offset=0, length=1),
        ]
        assert second.model_input_spans == [
            Prompt(sequence_id="b", offset=0, length=3),
            Sampled(sequence_id="b", offset=0, length=1),
        ]

    def test_one_token_prompt_and_action(self):
        # Dropping the first/last position removes a length-1 span entirely.
        (datum,) = _data(_transition([7], [8], "a"))
        assert datum.model_input_spans == [Prompt(sequence_id="a", offset=0, length=1)]
        assert datum.loss_fn_input_spans == [Sampled(sequence_id="a", offset=0, length=1)]

    def test_turn_with_no_new_observation_tokens(self):
        # Turn 2's prompt is exactly turn 1's prompt + action: no prompt span for turn 2.
        (datum,) = _data(
            _transition([1, 2], [3], "a"),
            _transition([1, 2, 3], [4, 5], "b"),
        )
        assert datum.loss_fn_input_spans == [
            Prompt(sequence_id="a", offset=1, length=1),
            Sampled(sequence_id="a", offset=0, length=1),
            Sampled(sequence_id="b", offset=0, length=2),
        ]

    def test_masked_parse_error_turn_is_still_covered(self):
        (datum,) = _data(
            _transition([1, 2], [3], "a", metrics={PARSE_ERROR_MASKED_METRIC_KEY: 1.0}),
            _transition([1, 2, 3, 4], [5], "b"),
        )
        assert datum.loss_fn_input_spans == [
            Prompt(sequence_id="a", offset=1, length=1),
            Sampled(sequence_id="a", offset=0, length=1),
            Prompt(sequence_id="b", offset=3, length=1),
            Sampled(sequence_id="b", offset=0, length=1),
        ]
        assert datum.loss_fn_inputs["mask"].to_torch().tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_empty_transition_needs_no_sequence_id(self):
        # Rollouts that end before their first turn (e.g. the prompt overflows the
        # context window) record an empty transition with no sampled sequence.
        empty = Transition(
            ob=tinker.ModelInput.empty(),
            ac=TokensWithLogprobs(tokens=[], maybe_logprobs=[]),
            reward=0.0,
            episode_done=True,
        )
        assert _data(empty) == []
        (datum,) = _data(_transition([1, 2], [3], "a"), empty)
        assert datum.loss_fn_input_spans == [
            Prompt(sequence_id="a", offset=1, length=1),
            Sampled(sequence_id="a", offset=0, length=1),
        ]

    def test_missing_sequence_id_omits_provenance(self):
        # Spans for only some turns can't tile the datum, so it gets none.
        for transitions in [
            [_transition([1, 2], [3], None)],
            [_transition([1, 2], [3], "a"), _transition([1, 2, 3, 4], [5], None)],
        ]:
            (datum,) = _data(*transitions)
            assert datum.model_input_spans is None
            assert datum.loss_fn_input_spans is None

    def test_missing_sequence_id_only_affects_its_datum(self):
        with_id, without_id = _data(_transition([1, 2], [3], "a"), _transition([9], [8], None))
        assert with_id.loss_fn_input_spans is not None
        assert without_id.loss_fn_input_spans is None

    def test_no_provenance_without_a_sampled_target(self):
        # The backend rejects provenance whose targets include no sampled token.
        (datum,) = _data(_transition([1, 2], [], "a"))
        assert datum.loss_fn_input_spans is None

    def test_no_provenance_when_disabled(self):
        (datum,) = _data(_transition([1, 2], [3], "a"), include_provenance=False)
        assert datum.model_input_spans is None
        assert datum.loss_fn_input_spans is None
