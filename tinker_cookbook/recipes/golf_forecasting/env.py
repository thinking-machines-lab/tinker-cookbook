from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import tinker
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.golf_forecasting.data import (
    GolfForecastExample,
    candidate_labels,
    leaderboard_table,
    normalize_player_name,
)
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

logger = logging.getLogger(__name__)

FORECAST_SYSTEM_PROMPT = (
    "You are a calibrated golf forecasting assistant. "
    "Read the live leaderboard snapshot and produce a probability distribution over likely winners. "
    "Return JSON only."
)


class WinnerForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner_probs: dict[str, float]

    @field_validator("winner_probs")
    @classmethod
    def validate_probs(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("winner_probs must not be empty")
        if any(prob < 0.0 for prob in value.values()):
            raise ValueError("winner_probs cannot contain negative probabilities")
        return value


def build_messages(example: GolfForecastExample, *, include_other_bucket: bool = True) -> list[renderers.Message]:
    candidates = candidate_labels(example) if include_other_bucket else example.candidate_names
    weather = example.system_context.get("weather_summary")
    course = example.system_context.get("course_difficulty")
    field_strength = example.system_context.get("field_strength")
    extras = []
    if weather:
        extras.append(f"- Weather: {weather}")
    if course:
        extras.append(f"- Course difficulty: {course}")
    if field_strength:
        extras.append(f"- Field strength: {field_strength}")
    extra_context = "\n".join(extras) if extras else "- No extra context provided."
    instructions = (
        f"Tournament: {example.tournament_name}\n"
        f"Course: {example.course_name or 'Unknown'}\n"
        f"Round: {example.round_number}\n"
        f"Event day: {example.event_day or 'Unknown'}\n"
        f"Snapshot time: {example.snapshot_timestamp}\n\n"
        "Leaderboard snapshot:\n"
        f"{leaderboard_table(example)}\n\n"
        "Extra context:\n"
        f"{extra_context}\n\n"
        "Return a JSON object with a single key `winner_probs`. "
        "Each key must be one of the candidate labels below and the probabilities must sum to 1.\n"
        f"Candidate labels: {', '.join(candidates)}\n"
        "Do not include explanations, markdown fences, or extra keys.\n"
        'Example: {"winner_probs": {"Player A": 0.45, "other": 0.55}}'
    )
    return [
        {"role": "system", "content": FORECAST_SYSTEM_PROMPT},
        {"role": "user", "content": instructions},
    ]


def _extract_json_blob(text: str) -> str:
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in model output")
    return text[start : end + 1]


def parse_forecast_response(
    text: str,
    *,
    allowed_labels: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    raw = WinnerForecast.model_validate_json(_extract_json_blob(text))
    allowed_lookup = {normalize_player_name(label): label for label in allowed_labels}
    output: dict[str, float] = {label: 0.0 for label in allowed_labels}
    unknown_mass = 0.0
    for label, prob in raw.winner_probs.items():
        canonical = allowed_lookup.get(normalize_player_name(label))
        if canonical is None:
            unknown_mass += prob
            continue
        output[canonical] += prob

    total = sum(output.values()) + unknown_mass
    if total <= 0:
        raise ValueError("Total forecast probability must be positive")
    normalized = {label: prob / total for label, prob in output.items()}
    diagnostics = {
        "raw_total_probability": total,
        "unknown_probability_mass": unknown_mass,
    }
    return normalized, diagnostics


def compute_multiclass_brier(
    forecast: dict[str, float],
    *,
    target_label: str,
) -> float:
    total = 0.0
    for label, prob in forecast.items():
        target = 1.0 if label == target_label else 0.0
        total += (prob - target) ** 2
    return total


def compute_log_loss(
    forecast: dict[str, float],
    *,
    target_label: str,
    floor: float = 1e-6,
) -> float:
    target_prob = max(forecast.get(target_label, 0.0), floor)
    return -math.log(target_prob)


def score_forecast(
    forecast: dict[str, float],
    *,
    target_label: str,
) -> dict[str, float]:
    ordered = sorted(forecast.items(), key=lambda item: item[1], reverse=True)
    top_labels = [label for label, _ in ordered[:3]]
    brier = compute_multiclass_brier(forecast, target_label=target_label)
    log_loss = compute_log_loss(forecast, target_label=target_label)
    # Multiclass Brier ranges from 0 to 2, so divide by 2 for a [0, 1] reward.
    brier_reward = 1.0 - min(brier / 2.0, 1.0)
    return {
        "brier": brier,
        "log_loss": log_loss,
        "brier_reward": brier_reward,
        "target_prob": forecast.get(target_label, 0.0),
        "top1_correct": float(ordered[0][0] == target_label),
        "top3_contains_target": float(target_label in top_labels),
    }


class GolfForecastEnv(Env):
    def __init__(
        self,
        *,
        example: GolfForecastExample,
        renderer: renderers.Renderer,
        include_other_bucket: bool = True,
        format_coef: float = 0.1,
    ):
        self.example = example
        self.renderer = renderer
        self.include_other_bucket = include_other_bucket
        self.format_coef = format_coef
        self.messages = build_messages(example, include_other_bucket=include_other_bucket)
        self.allowed_labels = candidate_labels(example) if include_other_bucket else example.candidate_names

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        base_metrics: dict[str, float] = {"parse_success": float(parse_success)}
        if not parse_success:
            return self._invalid_result(
                message=message,
                metrics=base_metrics,
                reason="renderer_parse_failed",
                content=content,
            )

        try:
            forecast, diagnostics = parse_forecast_response(content, allowed_labels=self.allowed_labels)
        except (ValidationError, ValueError) as exc:
            logger.debug("Forecast parse failed for %s: %s", self.example.example_id, exc)
            return self._invalid_result(
                message=message,
                metrics=base_metrics,
                reason=f"forecast_parse_failed:{exc}",
                content=content,
            )

        scores = score_forecast(forecast, target_label=self.example.target_label)
        reward = self.format_coef * (1.0 - 1.0) + scores["brier_reward"]
        metrics = {
            **base_metrics,
            **scores,
            **diagnostics,
            "format_valid": 1.0,
        }
        self._log_attempt(message=message, content=content, forecast=forecast, metrics=metrics)
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    def _invalid_result(
        self,
        *,
        message: renderers.Message,
        metrics: dict[str, float],
        reason: str,
        content: str,
    ) -> StepResult:
        invalid_metrics = {
            **metrics,
            "format_valid": 0.0,
            "brier": 2.0,
            "log_loss": compute_log_loss({}, target_label=self.example.target_label),
            "brier_reward": 0.0,
            "target_prob": 0.0,
            "top1_correct": 0.0,
            "top3_contains_target": 0.0,
            "raw_total_probability": 0.0,
            "unknown_probability_mass": 0.0,
        }
        self._log_attempt(
            message=message,
            content=content,
            forecast=None,
            metrics=invalid_metrics,
            error_reason=reason,
        )
        return StepResult(
            reward=-self.format_coef,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=invalid_metrics,
        )

    def _log_attempt(
        self,
        *,
        message: renderers.Message,
        content: str,
        metrics: dict[str, float],
        forecast: dict[str, float] | None,
        error_reason: str | None = None,
    ) -> None:
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=self.messages))
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Forecast Reward"):
            summary = {
                "example_id": self.example.example_id,
                "target_winner": self.example.target_winner,
                "target_label": self.example.target_label,
                "format_valid": bool(metrics["format_valid"]),
                "target_prob": f"{metrics['target_prob']:.4f}",
                "brier": f"{metrics['brier']:.4f}",
                "log_loss": f"{metrics['log_loss']:.4f}",
                "reward": f"{metrics['brier_reward']:.4f}",
            }
            if error_reason is not None:
                summary["error_reason"] = error_reason
            logtree.table_from_dict(summary, caption="Forecast summary")
            if forecast is not None:
                logtree.table_from_dict(
                    {label: f"{prob:.4f}" for label, prob in sorted(forecast.items())},
                    caption="Normalized forecast probabilities",
                )
            else:
                logtree.log_text(content[:1000])


@dataclass(frozen=True)
class GolfForecastGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], GolfForecastEnv]
    num_envs: int
    dataset_name: str = "golf_forecasting"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]

