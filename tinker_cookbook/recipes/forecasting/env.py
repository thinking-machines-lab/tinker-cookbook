"""Prophet Arena prompt, Brier reward, and Tinker RL dataset."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import chz

from tinker_cookbook import renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .data import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_REVISION,
    DEFAULT_MAX_TRAIN_QUESTIONS,
    DEFAULT_MAX_VALIDATION_QUESTIONS,
    DEFAULT_SPLIT_DATE,
    ForecastExample,
    fetch_prophet_arena,
    load_prophet_arena_split,
    parse_utc_datetime,
)

logger = logging.getLogger(__name__)

_FINAL_FORECAST_RE = re.compile(
    r"""
    \A\s*
    (?:\*{1,2}|_{1,2})?
    (?:(?:(?:final\s+)?(?:answer|forecast)|probability(?:\s+of\s+yes)?)\s*[:=]\s*)?
    (?:\*{1,2}|_{1,2})?
    (?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    \s*(?P<percent>%?)
    (?:\*{1,2}|_{1,2})?
    [.!]?
    (?:\*{1,2}|_{1,2})?
    \s*\Z
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_forecast(response: str) -> float | None:
    """Parse a probability from the response's final non-empty line."""
    lines = [line for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    match = _FINAL_FORECAST_RE.fullmatch(lines[-1])
    if match is None:
        return None
    probability = float(match.group("value"))
    if match.group("percent"):
        probability /= 100.0
    return probability if 0.0 <= probability <= 1.0 else None


def brier_reward(probability: float, outcome: int) -> float:
    """Return ``1 - (p - y)^2``, an affine form of the binary Brier score."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be finite and in [0, 1], got {probability}")
    if isinstance(outcome, bool) or outcome not in (0, 1):
        raise ValueError(f"outcome must be 0 or 1, got {outcome}")
    return 1.0 - (probability - outcome) ** 2


def _score(probability: float | None, outcome: int) -> dict[str, float]:
    if probability is None:
        return {
            "brier_reward": 0.0,
            "accuracy": 0.0,
            "format_valid": 0.0,
        }

    correctness = 0.5 if probability == 0.5 else float((probability > 0.5) == bool(outcome))
    return {
        "brier_reward": brier_reward(probability, outcome),
        "accuracy": correctness,
        "format_valid": 1.0,
    }


def render_prompt(example: ForecastExample) -> str:
    """Render the forecasting fields attached to the recorded snapshot."""
    # Do not add the resolved outcome or the market prices stored elsewhere in
    # the row.
    return f"""Forecast whether this market will resolve YES using information available through {example.snapshot_time.isoformat()}.

Event:
{example.event_title}

Market:
{example.market}

Reference material:
{example.reference_material}

Resolution criteria:
{example.resolution_criteria}

Market close time: {example.close_time.isoformat()}

Output only the probability of YES as a number between 0 and 1."""


class ForecastEnv(MessageEnv):
    """A single-turn binary forecast scored from its resolved outcome."""

    def __init__(self, example: ForecastExample):
        self.example = example
        self.example_id = f"{example.submission_id}:{example.market}"

    async def initial_observation(self) -> list[renderers.Message]:
        return [{"role": "user", "content": render_prompt(self.example)}]

    async def step(self, message: renderers.Message) -> MessageStepResult:
        probability = parse_forecast(get_text_content(message))
        scores = _score(probability, self.example.outcome)
        return MessageStepResult(
            reward=scores["brier_reward"],
            episode_done=True,
            next_messages=[],
            logs={
                "question_id": self.example_id,
                "forecast": "invalid" if probability is None else f"{probability:.6f}",
                "outcome": self.example.outcome,
                **scores,
            },
        )


@dataclass(frozen=True)
class ForecastGroupBuilder(EnvGroupBuilder):
    """Create one policy-gradient group from repeated forecasts of one market."""

    example: ForecastExample
    renderer: renderers.Renderer
    group_size: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            EnvFromMessageEnv(
                renderer=self.renderer,
                message_env=ForecastEnv(self.example),
                failed_parse_reward=0.0,
                context_overflow_reward=0.0,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        """Report each metric for valid, invalid, and truncated rollouts."""
        del env_group
        rewards_and_metrics: list[tuple[float, Metrics]] = []
        for trajectory in trajectory_group:
            logged_scores = next(
                (
                    transition.logs
                    for transition in trajectory.transitions
                    if "brier_reward" in transition.logs
                ),
                None,
            )
            scores = (
                _score(None, self.example.outcome)
                if logged_scores is None
                else {
                    key: float(logged_scores[key])
                    for key in ("brier_reward", "accuracy", "format_valid")
                }
            )
            rewards_and_metrics.append((0.0, scores))
        return rewards_and_metrics

    def logging_tags(self) -> list[str]:
        category = self.example.category.casefold().replace(" ", "-")
        return ["prophet-arena", category]


class ForecastRLDataset(RLDataset):
    """Batches of unique questions repeated for a configured number of epochs."""

    def __init__(
        self,
        examples: Sequence[ForecastExample],
        *,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        epochs: int = 1,
    ):
        if batch_size <= 0 or group_size <= 0 or epochs <= 0:
            raise ValueError("batch_size, group_size, and epochs must be positive")
        self.examples = tuple(examples)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.epochs = epochs

    def __len__(self) -> int:
        return self.epochs * math.ceil(len(self.examples) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if not 0 <= index < len(self):
            raise IndexError(f"batch index {index} outside [0, {len(self)})")
        batches_per_epoch = math.ceil(len(self.examples) / self.batch_size)
        start = (index % batches_per_epoch) * self.batch_size
        return [
            ForecastGroupBuilder(
                example=example,
                renderer=self.renderer,
                group_size=self.group_size,
            )
            for example in self.examples[start : start + self.batch_size]
        ]


@chz.chz
class ProphetArenaRLDatasetBuilder(RLDatasetBuilder):
    """Serializable config that downloads, splits, and renders Prophet Arena."""

    model_name_for_tokenizer: str
    renderer_name: str
    groups_per_batch: int
    group_size: int
    validation_group_size: int = 8
    train_epochs: int = 3
    data_path: str | None = None
    data_cache_dir: str = DEFAULT_CACHE_DIR
    dataset_revision: str = DEFAULT_DATASET_REVISION
    split_date: str = DEFAULT_SPLIT_DATE
    max_train_questions: int | None = DEFAULT_MAX_TRAIN_QUESTIONS
    max_validation_questions: int | None = DEFAULT_MAX_VALIDATION_QUESTIONS
    seed: int = 0

    async def __call__(self) -> tuple[ForecastRLDataset, ForecastRLDataset]:
        csv_path = (
            fetch_prophet_arena(self.dataset_revision, self.data_cache_dir)
            if self.data_path is None
            else self.data_path
        )
        split = load_prophet_arena_split(
            csv_path,
            split_time=parse_utc_datetime(self.split_date, "split_date"),
            max_train_questions=self.max_train_questions,
            max_validation_questions=self.max_validation_questions,
            seed=self.seed,
        )
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        logger.info(
            "Prophet Arena split: %d train, %d validation, %d crossing events excluded",
            len(split.train),
            len(split.validation),
            split.excluded_crossing_events,
        )
        return (
            ForecastRLDataset(
                split.train,
                batch_size=self.groups_per_batch,
                group_size=self.group_size,
                renderer=renderer,
                epochs=self.train_epochs,
            ),
            ForecastRLDataset(
                split.validation,
                batch_size=self.groups_per_batch,
                group_size=self.validation_group_size,
                renderer=renderer,
            ),
        )
