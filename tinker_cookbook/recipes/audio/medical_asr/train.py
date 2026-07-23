"""Full medical-ASR SFT (EkaCare) via the tml_v0 renderer.

    ekacare/eka-medical-asr-evaluation-dataset (en)  [seeded random 80/20 split]
      -> local WAV cache -> chat.Message(AudioPointer) (DMel-encoded at render time)
      -> renderer.build_supervised_example -> cookbook Datum
      -> supervised.train.main   (test/wer down, test/entity_cer down, test/nll down)

Full run (defaults below): all ~2,895 train clips, 12 epochs (~2,172 steps at
batch_size 16), LoRA rank 32, lr 1e-3 (linear decay). Eval is the full ~724-clip
held-out set every 100 steps -- corpus WER (test/wer), a medical-entity character
error rate + hit-rate over annotated drug/finding phrases (test/entity_cer,
test/entity_hit_rate), and teacher-forced transcript NLL (test/nll, auto-added
from the eval dataset). Checkpoints are
saved every 100 steps (aligned with eval), so the best-eval checkpoint can be
picked post-hoc from the WER curve.

    TINKER_API_KEY=<key> WANDB_API_KEY=<key> \
      uv run python -m tinker_cookbook.recipes.audio.medical_asr.train \
        wandb_project=<project>

Split note: the random split puts each speaker on both sides, so test/wer is
speaker-adapted, not unseen-speaker generalization; ~8% duplicate transcripts
make it optimistic. It is the in-distribution hill-climbing signal.
"""

from __future__ import annotations

import asyncio

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.audio.medical_asr.env import (
    EkacareASRDatasetBuilder,
    EkacareWEREvaluatorBuilder,
)
from tinker_cookbook.supervised import train
from tinker_cookbook.utils.lr_scheduling import LRSchedule


@chz.chz
class Config:
    log_path: str = "/tmp/tinker-examples/medical-asr"
    model_name: str = "thinkingmachines/Inkling"
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Data. WAV cache lives outside log_path so clearing logs doesn't re-download.
    # split_tag documents the split design in the run config; changing it does
    # nothing on its own (the split is a seeded random 80/20 in env.py).
    audio_cache_dir: str = "/tmp/tinker-examples/medical-asr-clips"
    max_length: int = 8192
    seed: int = 0
    split_tag: str = "random_speaker_adapted"

    # Training (full run). LoRA wants ~10x the full-FT LR; 1e-3 is hand-picked.
    num_epochs: int = 12
    max_steps: int | None = None
    batch_size: int = 16
    learning_rate: float = 1e-3
    lr_schedule: LRSchedule = "linear"
    lora_rank: int = 32

    # Evaluation / checkpointing (aligned so each eval has a checkpoint).
    eval_every: int = 100
    save_every: int = 100
    max_tokens: int = 8192
    temperature: float = 1.0
    max_parallel_tasks: int = 128

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None


def cli_main(cfg: Config) -> None:
    # check_log_dir may prompt via input(); run it before entering asyncio.
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    dataset_builder = EkacareASRDatasetBuilder(
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        seed=cfg.seed,
        audio_cache_dir=cfg.audio_cache_dir,
    )
    wer_evaluator_builder = EkacareWEREvaluatorBuilder(
        model_name=cfg.model_name,
        log_path=cfg.log_path,
        seed=cfg.seed,
        audio_cache_dir=cfg.audio_cache_dir,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        max_parallel_tasks=cfg.max_parallel_tasks,
    )
    train_config = train.Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        recipe_name="recipe_audio_medical_asr_sl",
        renderer_name=model_info.get_recommended_renderer_name(cfg.model_name),
        dataset_builder=dataset_builder,
        evaluator_builders=[wer_evaluator_builder],
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        lora_rank=cfg.lora_rank,
        base_url=cfg.base_url,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
