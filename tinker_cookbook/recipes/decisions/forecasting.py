"""Binary forecasting on Prophet Arena with the decision model.

Same data, temporal split and validation metrics as :mod:`tinker_cookbook.recipes.forecasting`,
without any reasoning or rollouts: each market is a ``yes`` / ``no`` decision written as
``choice#Y`` / ``choice#N`` by :class:`YesNoChoiceRenderer`, P(YES) is the model's probability
of ``yes``, and training is supervised with a Brier loss on the resolved outcome.

    # 1024 train / 256 validation markets, as in the forecasting recipe
    uv run -m tinker_cookbook.recipes.decisions.forecasting

    # Quick run on a subset
    uv run -m tinker_cookbook.recipes.decisions.forecasting --model_name=Qwen/Qwen3.5-4B \\
        --max_train_questions=50 --max_validation_questions=50 --batch_size=10 --max_steps=5

Pass ``--log_path`` to make a run resumable: an interrupted run restarted with the same
``log_path`` continues from its last checkpoint (``save_every`` controls how often one is
written).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import chz
import tinker
import torch

from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.recipes.decisions.decision import (
    ChoiceRenderer,
    ChoiceRendering,
    Decision,
    DecisionRenderer,
    sample_decision,
)
from tinker_cookbook.recipes.decisions.train import decision_dist_datums, probability_loss
from tinker_cookbook.recipes.forecasting.data import (
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
from tinker_cookbook.recipes.forecasting.env import brier_reward, roc_auc
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.git_rev import recipe_user_metadata

logger = logging.getLogger(__name__)


def brier_score_loss(choice_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Brier score against the target; ``2 * (p_yes - outcome)^2`` for a one-hot yes / no."""
    return ((choice_probs - target_probs) ** 2).sum()


YES = "yes"
NO = "no"
REPLIES: Mapping[str, str] = {YES: "choice#Y", NO: "choice#N"}
ANSWER_FORMAT = "choice#?"


class YesNoChoiceRenderer(ChoiceRenderer):
    """Write ``yes`` / ``no`` as ``choice#Y`` / ``choice#N``, shown to the model as ``choice#?``.

    The replies and the format share every token but the last, so that is the answer
    position. The result is keyed by ``yes`` / ``no``.
    """

    def __init__(self, renderer: renderers.Renderer):
        self.tokenizer = renderer.tokenizer

    def render_choices(self, labels: Sequence[str]) -> ChoiceRendering:
        if set(labels) != set(REPLIES):
            raise ValueError(f"expected the labels {list(REPLIES)}, got {list(labels)}")
        reply_tokens = {
            label: self.tokenizer.encode(REPLIES[label], add_special_tokens=False)
            for label in labels
        }
        format_tokens = self.tokenizer.encode(ANSWER_FORMAT, add_special_tokens=False)
        last_tokens = {tokens[-1] for tokens in (*reply_tokens.values(), format_tokens)}
        if (
            any(tokens[:-1] != format_tokens[:-1] for tokens in reply_tokens.values())
            or len(last_tokens) != 3
        ):
            raise ConfigurationError(
                f"{REPLIES[YES]!r}, {REPLIES[NO]!r} and {ANSWER_FORMAT!r} must tokenize "
                f"identically except for a distinct final token, got {reply_tokens} and "
                f"{format_tokens}"
            )
        return ChoiceRendering(
            answer_tokens=format_tokens,
            answer_position=len(format_tokens) - 1,
            label_tokens={label: tokens[-1] for label, tokens in reply_tokens.items()},
        )


DECISION_TEMPLATE = """\
Will this market resolve YES? Use only information available through {snapshot_time}.

Event:
{event_title}

Market:
{market}

Reference material:
{reference_material}

Resolution criteria:
{resolution_criteria}

Market close time: {close_time}"""


def forecast_decision(example: ForecastExample) -> Decision:
    """A market as a yes / no question; the outcome and market prices are deliberately absent."""
    return Decision(
        DECISION_TEMPLATE.format(
            snapshot_time=example.snapshot_time.isoformat(),
            event_title=example.event_title,
            market=example.market,
            reference_material=example.reference_material,
            resolution_criteria=example.resolution_criteria,
            close_time=example.close_time.isoformat(),
        ),
        [(YES, "The market resolves YES."), (NO, "The market resolves NO.")],
    )


async def evaluate_forecasts(
    sampling_client: tinker.SamplingClient,
    decision_renderer: DecisionRenderer,
    examples: Sequence[ForecastExample],
) -> dict[str, float]:
    """Decide every validation market concurrently and score the YES probabilities with the
    Brier reward, accuracy and AUC, as the forecasting recipe reports them."""
    probabilities = await asyncio.gather(
        *(
            sample_decision(sampling_client, decision_renderer.render(forecast_decision(e)))
            for e in examples
        )
    )
    pairs = [(p[YES], e.outcome) for e, p in zip(examples, probabilities, strict=True)]
    metrics = {
        "brier_reward": sum(brier_reward(p, y) for p, y in pairs) / len(pairs),
        "accuracy": sum(0.5 if p == 0.5 else float((p > 0.5) == bool(y)) for p, y in pairs)
        / len(pairs),
    }
    auc = roc_auc(pairs)
    if auc is not None:
        metrics["auc"] = auc
    return metrics


def default_renderer_name(model_name: str) -> str:
    """The recommended renderer, preferring its low-reasoning-effort variant.

    The decision model never reasons (the reply is written by us and only scored), so a
    renderer that tells the model to think at length asks for something it never gets to do.
    Qwen3.8's and GLM-5.3's recommended renderers default to the highest effort; their
    ``*_low_reasoning`` variants are picked instead, as in the forecasting recipe.
    """
    recommended = model_info.get_recommended_renderer_names(model_name)
    return next((name for name in recommended if name.endswith("_low_reasoning")), recommended[0])


@chz.chz
class Config:
    # Model
    model_name: str = "Qwen/Qwen3.8-27B"
    renderer_name: str | None = None  # default: default_renderer_name(model_name)
    lora_rank: int = 32

    # Data (same defaults and split as tinker_cookbook.recipes.forecasting)
    data_path: str | None = None
    data_cache_dir: str = DEFAULT_CACHE_DIR
    dataset_revision: str = DEFAULT_DATASET_REVISION
    split_date: str = DEFAULT_SPLIT_DATE
    max_train_questions: int | None = DEFAULT_MAX_TRAIN_QUESTIONS
    max_validation_questions: int | None = DEFAULT_MAX_VALIDATION_QUESTIONS
    seed: int = 0  # data subsampling and order, and LoRA initialization

    # Optimization
    batch_size: int = 32
    learning_rate: float = 1e-4  # as in recipes/forecasting
    epochs: int = 1
    max_steps: int | None = None

    # Evaluation and logging
    eval_every: int = 16  # 0 disables periodic evaluation; the start and end are always scored
    save_every: int = 0  # 0 disables intermediate checkpoints; a final one is always saved
    log_path: str | None = None  # set this to make the run resumable
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    base_url: str | None = None


async def train(
    cfg: Config,
    *,
    training_client: tinker.TrainingClient,
    decision_renderer: DecisionRenderer,
    train_examples: Sequence[tuple[Decision, str]],
    validation: Sequence[ForecastExample],
    ml_logger: ml_log.Logger,
    log_path: str,
    start_step: int = 0,
) -> None:
    """Brier-loss SFT on ``(decision, outcome label)`` pairs, scoring ``validation`` from the
    trained weights at every evaluation."""
    steps_per_epoch = math.ceil(len(train_examples) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    logger.info(
        "training for %d steps of %d examples, starting at step %d",
        total_steps,
        cfg.batch_size,
        start_step,
    )
    custom_loss = probability_loss(brier_score_loss)

    async def evaluate(step: int) -> None:
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        metrics = await evaluate_forecasts(sampling_client, decision_renderer, validation)
        ml_logger.log_metrics({f"test/{k}": v for k, v in metrics.items()}, step=step)

    if start_step == 0:
        await evaluate(step=0)
    for step in range(start_step, total_steps):
        start_time = time.time()
        offset = (step % steps_per_epoch) * cfg.batch_size
        batch = train_examples[offset : offset + cfg.batch_size]

        datums = decision_dist_datums(
            decision_renderer, [(decision, {target: 1.0}) for decision, target in batch]
        )
        fwd_bwd_future = await training_client.forward_backward_custom_async(datums, custom_loss)
        optim_future = await training_client.optim_step_async(
            tinker.AdamParams(learning_rate=cfg.learning_rate)
        )
        fwd_bwd_result = await fwd_bwd_future.result_async()
        await optim_future.result_async()

        metrics = {
            f"train/{key}": value
            for key, value in fwd_bwd_result.metrics.items()
            if key in ("loss", "target_prob:mean", "accuracy")
        }
        metrics.update(
            learning_rate=cfg.learning_rate,
            num_examples=len(batch),
            progress=(step + 1) / total_steps,
            time_total=time.time() - start_time,
        )
        ml_logger.log_metrics(metrics, step=step + 1)

        is_last = step + 1 == total_steps
        if is_last or (cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0):
            await evaluate(step=step + 1)
        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0 and not is_last:
            await checkpoint_utils.save_checkpoint_async(
                training_client,
                name=f"{step + 1:06d}",
                log_path=log_path,
                loop_state={"step": step + 1},
                kind="state",
            )

    await checkpoint_utils.save_checkpoint_async(
        training_client,
        name="final",
        log_path=log_path,
        loop_state={"step": total_steps, "final": True},
        kind="both",
    )


async def main(cfg: Config) -> None:
    run_name = (
        f"decisions-forecasting-{cfg.model_name.lower().replace('/', '-')}-bs{cfg.batch_size}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cfg.log_path or cli_utils.runs_path("decisions_forecasting", run_name)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    # Resume from the last checkpoint if the log directory was kept ("resume" above).
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or run_name,
        config=cfg,
    )

    csv_path = (
        Path(cfg.data_path).expanduser()
        if cfg.data_path is not None
        else fetch_prophet_arena(cfg.dataset_revision, cfg.data_cache_dir)
    )
    split = load_prophet_arena_split(
        csv_path,
        split_time=parse_utc_datetime(cfg.split_date, "split_date"),
        max_train_questions=cfg.max_train_questions,
        max_validation_questions=cfg.max_validation_questions,
        seed=cfg.seed,
    )
    train_examples = [
        (forecast_decision(example), YES if example.outcome else NO) for example in split.train
    ]
    logger.info(
        "train: %d markets, validation: %d markets", len(split.train), len(split.validation)
    )

    renderer_name = cfg.renderer_name or default_renderer_name(cfg.model_name)
    model_info.warn_if_renderer_not_recommended(cfg.model_name, cfg.renderer_name)
    renderer = renderers.get_renderer(renderer_name, get_tokenizer(cfg.model_name))
    decision_renderer = DecisionRenderer(renderer, YesNoChoiceRenderer(renderer))

    service_client = tinker.ServiceClient(
        base_url=cfg.base_url,
        user_metadata=recipe_user_metadata("recipe_decisions_forecasting"),
    )
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, renderer_name)
    if resume_info is not None:
        assert resume_info.state_path is not None
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_info.state_path, renderer_name
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path, user_metadata=user_metadata
            )
        )
        start_step = int(resume_info.get("step", 0))
        logger.info("Resumed training from %s at step %d", resume_info.state_path, start_step)
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=cfg.model_name,
            rank=cfg.lora_rank,
            seed=cfg.seed,
            user_metadata=user_metadata,
        )
        start_step = 0

    await train(
        cfg,
        training_client=training_client,
        decision_renderer=decision_renderer,
        train_examples=train_examples,
        validation=split.validation,
        ml_logger=ml_logger,
        log_path=log_path,
        start_step=start_step,
    )
    ml_logger.close()


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config, allow_hyphens=True)))
