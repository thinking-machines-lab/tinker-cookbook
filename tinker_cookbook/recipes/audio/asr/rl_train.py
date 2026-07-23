"""Toy audio-ASR RL recipe (LibriSpeech) via the tml_v0 renderer.

    HF openslr/librispeech_asr (other/train.500)
      -> local WAV cache -> chat.Message(AudioPointer) prompts
      -> group_size transcripts sampled per clip
      -> per-rollout reward = -WER  ->  rl.train.main
         (group-centered advantages, GRPO-style)

Smoke run (defaults: 64 clips, 8 rollouts x 8 clips per batch, one 8-iteration pass):

    TINKER_API_KEY=<key> uv run python -m tinker_cookbook.recipes.audio.asr.rl_train \
        wandb_project=<project>

Reference-scale run (2048 clips, 64 iterations, ~2 h):

    TINKER_API_KEY=<key> uv run python -m tinker_cookbook.recipes.audio.asr.rl_train \
        wandb_project=<project> \
        n_train=2048 n_eval=64 groups_per_batch=32 learning_rate=2e-5 \
        eval_every=4 save_every=8

Envs, datasets, and the WER evaluator live in env.py. Rollouts sample at T=1.0
(within-group WER variance is the gradient signal); rewards cap at -1 so one
unparseable ramble (WER >> 1) can't dominate its group. Start from an audio
sl_train checkpoint via load_checkpoint_path, or from the base model to
exercise format acquisition. Note Inkling starts near WER 0.06 on clean
LibriSpeech, so most GRPO groups are constant-reward on this pairing -- the
recipe demonstrates the audio RL pipeline; a harder distribution gives a
visible reward curve.
"""

from __future__ import annotations

import asyncio

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.audio.asr.env import (
    AudioASRRLDatasetBuilder,
    WEREvaluatorBuilder,
)
from tinker_cookbook.rl import train


@chz.chz
class Config:
    log_path: str = "/tmp/tinker-examples/audio-asr-rl"
    model_name: str = "thinkingmachines/Inkling"
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    # Start from an audio sl_train checkpoint to skip format acquisition.
    load_checkpoint_path: str | None = None

    # Data (same LibriSpeech cache as the SFT recipe).
    n_train: int = 64
    n_eval: int = 32
    shuffle_buffer_size: int = 1_000
    audio_cache_dir: str = "/tmp/tinker-examples/audio-asr-clips"
    seed: int = 0  # data selection

    # RL: one pass of n_train clips, groups_per_batch clips per iteration,
    # group_size rollouts per clip. Size with n_train; cut short with max_steps.
    group_size: int = 8
    groups_per_batch: int = 8
    max_steps: int | None = None
    learning_rate: float = 1e-5
    lora_rank: int = 32
    max_tokens: int = 8192

    # Evaluation (env.py's WER evaluator on held-out clips).
    eval_every: int = 8  # 0 = disabled
    save_every: int = 8
    max_parallel_tasks: int = 128

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None


def cli_main(cfg: Config) -> None:
    # check_log_dir may prompt via input(); run it before entering asyncio.
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    dataset_builder = AudioASRRLDatasetBuilder(
        model_name=cfg.model_name,
        groups_per_batch=cfg.groups_per_batch,
        group_size=cfg.group_size,
        n_train=cfg.n_train,
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
        max_parallel_tasks=cfg.max_parallel_tasks,
    )
    train_config = train.Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        recipe_name="recipe_audio_asr_rl",
        renderer_name=model_info.get_recommended_renderer_name(cfg.model_name),
        dataset_builder=dataset_builder,
        evaluator_builders=[wer_evaluator_builder] if cfg.n_eval else [],
        learning_rate=cfg.learning_rate,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        max_steps=cfg.max_steps,
        load_checkpoint_path=cfg.load_checkpoint_path,
        base_url=cfg.base_url,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
