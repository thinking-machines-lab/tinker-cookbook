from __future__ import annotations

from tinker_cookbook.recipes.golf_forecasting.build_dataset import _split_tournaments
from tinker_cookbook.recipes.golf_forecasting.data import DatasetManifest, GolfForecastExample, PlayerSnapshot
from tinker_cookbook.recipes.golf_forecasting.env import parse_forecast_response, score_forecast


def _make_example(
    tournament_id: str,
    example_id: str,
    target_winner: str,
) -> GolfForecastExample:
    return GolfForecastExample(
        example_id=example_id,
        tournament_id=tournament_id,
        tournament_name=tournament_id,
        course_name=None,
        round_number=4,
        event_day="Sunday",
        snapshot_timestamp="2026-04-11T00:00:00Z",
        players=(
            PlayerSnapshot(
                name="Player A",
                position="1",
                score_to_par=-10,
                strokes_behind=0.0,
            ),
            PlayerSnapshot(
                name="Player B",
                position="2",
                score_to_par=-9,
                strokes_behind=1.0,
            ),
        ),
        target_winner=target_winner,
    )


def test_parse_forecast_response_normalizes_and_routes_unknown_mass() -> None:
    forecast, diagnostics = parse_forecast_response(
        '{"winner_probs": {"Player A": 0.6, "Mystery Golfer": 0.4}}',
        allowed_labels=["Player A", "Player B", "other"],
    )
    assert forecast["Player A"] == 0.6
    assert forecast["Player B"] == 0.0
    assert forecast["other"] == 0.0
    assert diagnostics["raw_total_probability"] == 1.0
    assert diagnostics["unknown_probability_mass"] == 0.4


def test_score_forecast_prefers_higher_target_probability() -> None:
    better = score_forecast(
        {"Player A": 0.7, "Player B": 0.2, "other": 0.1},
        target_label="Player A",
    )
    worse = score_forecast(
        {"Player A": 0.2, "Player B": 0.7, "other": 0.1},
        target_label="Player A",
    )
    assert better["log_loss"] < worse["log_loss"]
    assert better["brier"] < worse["brier"]
    assert better["top1_correct"] == 1.0
    assert worse["top1_correct"] == 0.0


def test_split_tournaments_preserves_existing_heldout_assignments() -> None:
    examples = [
        _make_example("masters", "masters-1", "Player A"),
        _make_example("open", "open-1", "Player A"),
        _make_example("players", "players-1", "Player A"),
    ]
    existing_manifest = DatasetManifest(
        dataset_version="v1",
        created_at="2026-04-11T00:00:00Z",
        output_dir="/tmp/golf",
        train_path="/tmp/golf/train.jsonl",
        val_path="/tmp/golf/val.jsonl",
        heldout_path="/tmp/golf/heldout.jsonl",
        heldout_locked=True,
        split_tournament_ids={"train": ["masters"], "val": ["players"], "heldout": ["open"]},
    )
    splits = _split_tournaments(
        examples,
        val_fraction=0.2,
        heldout_fraction=0.2,
        seed=0,
        existing_manifest=existing_manifest,
    )
    assert [example.tournament_id for example in splits["heldout"]] == ["open"]
    assert [example.tournament_id for example in splits["val"]] == ["players"]
    assert [example.tournament_id for example in splits["train"]] == ["masters"]
