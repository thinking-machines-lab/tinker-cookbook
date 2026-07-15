"""RL fine-tuning of an audio model on Expresso: emotion tag + transcription.

The policy hears an expressive speech clip and must answer with
``[<style>] <transcription>``. Rewards combine emotion-classification
accuracy and word error rate (see ``env.py``); the RL loop, checkpointing,
and eval cadence come from ``rl.train``. The dev-split evaluator reports
``dev/emotion_accuracy`` / ``dev/wer`` during training (the step-0 eval is
the pre-fine-tuning baseline); run ``evaluate.py`` for the final
before/after comparison on the test split.

Smoke run (2 batches of 4 clips x 4 rollouts, a few minutes):

    uv run python -m tinker_cookbook.recipes.audio.emotion.rl_train \
        groups_per_batch=4 group_size=4 n_eval=8 max_steps=2 eval_every=1

Reference run, warm-started from a Stage-1 SFT checkpoint (see ``sl_train.py``;
this trains on rl_train, clip-disjoint from the SFT data):

    uv run python -m tinker_cookbook.recipes.audio.emotion.rl_train \
        load_checkpoint_path="tinker://<sft-run-id>/weights/<step>"
"""

from __future__ import annotations

import asyncio

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.audio.emotion.env import (
    DEFAULT_DATA_DIR,
    ExpressoRLDatasetBuilder,
)
from tinker_cookbook.recipes.audio.emotion.evaluate import ExpressoEvaluatorBuilder
from tinker_cookbook.rl import train


@chz.chz
class Config:
    log_path: str = "/tmp/tinker-examples/audio-rl"
    model_name: str = "thinkingmachines/Inkling"
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    # Stage-1 SFT checkpoint to warm-start from (sl_train.py); None = RL from base.
    load_checkpoint_path: str | None = None

    # Data prepared by prepare_data.py; RL trains on the rl_train split
    # (clip-disjoint from the SFT stage's sft_train split).
    data_dir: str = DEFAULT_DATA_DIR
    n_train: int | None = None  # None = the full rl_train split (800 clips)
    # Dev clips for the in-training evaluator (None = all, 0 = disable).
    n_eval: int | None = 64
    seed: int = 0

    # Reward: emotion_coef * 1[style correct] + wer_coef * max(0, 1 - WER).
    # No format term -- the base model already formats reliably (see README).
    emotion_coef: float = 0.5
    wer_coef: float = 0.5

    # Training. hyperparam_utils.get_lr has no TML entry, so the LR is
    # hand-picked (LoRA wants ~10x the full-FT LR).
    group_size: int = 8  # rollouts per clip (advantages centered per group)
    groups_per_batch: int = 16  # clips per training batch
    learning_rate: float = 1e-5
    lora_rank: int = 32
    # None = one pass over rl_train (50 steps); dev metrics plateau by ~step 40.
    max_steps: int | None = None
    # A large budget lets the not-yet-finetuned model finish thinking and emit
    # a parseable answer; without it, early rollouts are all format failures.
    max_tokens: int = 8192
    temperature: float = 1.0

    # Evaluation (dev split) and checkpointing
    eval_every: int = 10  # 0 = disabled; step 0 gives the baseline
    save_every: int = 20

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None


def cli_main(cfg: Config) -> None:
    # check_log_dir may prompt via input(); run it before entering asyncio.
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    dataset_builder = ExpressoRLDatasetBuilder(
        model_name=cfg.model_name,
        data_dir=cfg.data_dir,
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        n_train=cfg.n_train,
        seed=cfg.seed,
        emotion_coef=cfg.emotion_coef,
        wer_coef=cfg.wer_coef,
    )
    evaluator_builder = ExpressoEvaluatorBuilder(
        model_name=cfg.model_name,
        log_path=cfg.log_path,
        data_dir=cfg.data_dir,
        split="dev",
        n_eval=cfg.n_eval,
        seed=cfg.seed,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    train_config = train.Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        recipe_name="recipe_audio_emotion_rl",
        renderer_name=model_info.get_recommended_renderer_name(cfg.model_name),
        dataset_builder=dataset_builder,
        evaluator_builders=[evaluator_builder] if cfg.eval_every and cfg.n_eval != 0 else [],
        load_checkpoint_path=cfg.load_checkpoint_path,
        learning_rate=cfg.learning_rate,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        max_steps=cfg.max_steps,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        base_url=cfg.base_url,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
