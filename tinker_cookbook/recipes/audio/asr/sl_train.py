"""Toy audio-ASR SFT recipe (LibriSpeech) via the tml_v0 renderer.

    HF openslr/librispeech_asr (other/train.500)
      -> local WAV cache -> chat.Message(AudioPointer) (DMel-encoded at render time)
      -> renderer.build_supervised_example -> cookbook Datum
      -> supervised.train.main   (NLL down, WER down)

Smoke run (defaults: 64 clips, 5 epochs, a few minutes):

    TINKER_API_KEY=<key> uv run python -m tinker_cookbook.recipes.audio.asr.sl_train \
        wandb_project=<project>

Reference-scale run (512 clips / 201 steps, ~30 min):

    TINKER_API_KEY=<key> uv run python -m tinker_cookbook.recipes.audio.asr.sl_train \
        wandb_project=<project> \
        n_train=512 n_eval=64 num_epochs=7 max_steps=201 eval_every=25 \
        batch_size=16 lora_rank=32

Data, datasets, and the WER evaluator live in env.py; the loop, checkpointing,
resume, and eval cadence come from supervised.train. Eval samples with a large
max_tokens budget so the untrained base model can finish thinking and emit a
parseable transcript for test/wer (a rollout with no parseable message scores
WER 1.0). Note Inkling already transcribes clean LibriSpeech well (test/wer
~0.06 zero-shot), so gains on this pairing are small -- the recipe demonstrates
the audio SFT pipeline end to end; point it at a harder distribution (see
medical_asr/) for visible curves.
"""

from __future__ import annotations

import asyncio

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.audio.asr.env import (
    AudioASRDatasetBuilder,
    WEREvaluatorBuilder,
)
from tinker_cookbook.supervised import train
from tinker_cookbook.utils.lr_scheduling import LRSchedule


@chz.chz
class Config:
    log_path: str = "/tmp/tinker-examples/audio-asr"
    model_name: str = "thinkingmachines/Inkling"
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Data. The WAV cache lives outside log_path so clearing the log dir
    # between runs doesn't force a re-download.
    n_train: int = 64
    n_eval: int = 32
    shuffle_buffer_size: int = 1_000  # buffered rows carry decoded audio; keep modest
    audio_cache_dir: str = "/tmp/tinker-examples/audio-asr-clips"
    max_length: int = 8192
    seed: int = 0  # data selection + epoch shuffling

    # Training. hyperparam_utils.get_lr has no TML entry, so the peak LR is
    # hand-picked (LoRA wants ~10x the full-FT LR).
    num_epochs: int = 5
    max_steps: int | None = None
    batch_size: int = 8
    learning_rate: float = 2e-4
    lr_schedule: LRSchedule = "linear"
    lora_rank: int = 16

    # Evaluation
    eval_every: int = 20  # 0 = disabled
    max_tokens: int = 8192
    temperature: float = 1.0
    max_parallel_tasks: int = 128

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None


def cli_main(cfg: Config) -> None:
    # check_log_dir may prompt via input(); run it before entering asyncio.
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    dataset_builder = AudioASRDatasetBuilder(
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        n_train=cfg.n_train,
        n_eval=cfg.n_eval,
        max_length=cfg.max_length,
        seed=cfg.seed,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        audio_cache_dir=cfg.audio_cache_dir,
    )
    wer_evaluator_builder = WEREvaluatorBuilder(
        model_name=cfg.model_name,
        log_path=cfg.log_path,
        n_eval=cfg.n_eval,
        seed=cfg.seed,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        audio_cache_dir=cfg.audio_cache_dir,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        max_parallel_tasks=cfg.max_parallel_tasks,
    )
    train_config = train.Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        recipe_name="recipe_audio_asr_sl",
        renderer_name=model_info.get_recommended_renderer_name(cfg.model_name),
        dataset_builder=dataset_builder,
        evaluator_builders=[wer_evaluator_builder] if cfg.n_eval else [],
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        lora_rank=cfg.lora_rank,
        base_url=cfg.base_url,
        eval_every=cfg.eval_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
