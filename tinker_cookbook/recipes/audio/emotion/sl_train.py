"""SFT on Expresso: emotion tag + transcription, as a warm start for RL.

Supervises the same task as the RL recipe -- ``[<style>] <transcription>``
per clip -- with plain cross-entropy on the standard cookbook SFT stack.
The motivation: RL from the base model
stalls on emotion because most rollout groups sample the same (wrong) style
and get zero advantage; SFT supervises the style tag on every clip
unconditionally, then RL (``rl_train.py`` with ``load_checkpoint_path``) can
refine from a policy whose groups actually have contrast.

Training uses the ``sft_train`` split (1,600 clips: one style-balanced
rendition per distinct sentence -- Expresso repeats each sentence ~4.5x
across styles and speakers), and runs the same dev-split evaluator as RL
training (``dev/emotion_accuracy``, ``dev/emotion_macro_f1``, ``dev/wer``,
``dev/format_valid``), so SFT and RL curves are directly comparable.

Reference run (one epoch over sft_train = 100 steps):

    uv run python -m tinker_cookbook.recipes.audio.emotion.sl_train
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

import chz
import tinker

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.audio.data import audio_renderer
from tinker_cookbook.recipes.audio.emotion.env import (
    DEFAULT_DATA_DIR,
    Clip,
    load_clips,
    prompt_messages,
)
from tinker_cookbook.recipes.audio.emotion.evaluate import ExpressoEvaluatorBuilder
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.utils.lr_scheduling import LRSchedule

if TYPE_CHECKING:
    from tml_renderers import chat  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


def sft_messages(clip: Clip) -> list[chat.Message]:
    """The full conversation: prompt + supervised ``[<style>] <transcription>``
    turn. ``ModelEndSampling`` closes the model turn and is a loss target, so
    the model learns to stop instead of thinking on; the leading space
    matches the audio training data convention."""
    from tml_renderers import chat  # pyright: ignore[reportMissingImports]

    model = chat.Author(chat.AuthorKind.Model)
    target = f"[{clip['emotion']}] {clip['text'].strip()}"
    return prompt_messages(clip) + [
        chat.Message(content=chat.Text(" " + target), author=model),
        chat.Message(content=chat.ModelEndSampling(), author=model),
    ]


def _clip_datum(renderer: TmlV0Renderer, clip: Clip, max_length: int) -> tinker.Datum:
    model_input, weights = renderer.build_supervised_example(sft_messages(clip))
    return datum_from_model_input_weights(model_input, weights, max_length=max_length)


class ExpressoSFTDataset(SupervisedDataset):
    """A fixed list of datums served in per-epoch reshuffled batches."""

    def __init__(self, datums: list[tinker.Datum], batch_size: int):
        self.datums = datums
        self.batch_size = batch_size
        self.order = list(range(len(datums)))

    def __len__(self) -> int:
        return len(self.datums) // self.batch_size  # drop-last

    def set_epoch(self, seed: int = 0):
        random.Random(seed).shuffle(self.order)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        lo = index * self.batch_size
        return [self.datums[i] for i in self.order[lo : lo + self.batch_size]]


@chz.chz
class ExpressoSFTDatasetBuilder(SupervisedDatasetBuilder):
    model_name: str
    data_dir: str
    batch_size: int
    n_train: int | None  # first n of the shuffled sft_train split (None = all)
    n_eval: int | None  # dev clips for the auto-added NLL evaluator (None = all, 0 = disable)
    max_length: int
    seed: int

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        renderer = audio_renderer(self.model_name)
        train_clips = load_clips(self.data_dir, "sft_train", self.n_train, self.seed)
        train_ds = ExpressoSFTDataset(
            [_clip_datum(renderer, c, self.max_length) for c in train_clips], self.batch_size
        )
        if self.n_eval == 0:
            return train_ds, None
        eval_clips = load_clips(self.data_dir, "dev", self.n_eval, self.seed)
        eval_ds = ExpressoSFTDataset(
            [_clip_datum(renderer, c, self.max_length) for c in eval_clips], self.batch_size
        )
        return train_ds, eval_ds


@chz.chz
class Config:
    log_path: str = "/tmp/tinker-examples/audio-sft"
    model_name: str = "thinkingmachines/Inkling"
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Data: the sft_train split built by prepare_data.py -- 1,600 clips, one
    # style-balanced rendition per distinct sentence, clip-disjoint from the
    # RL stage's rl_train split.
    data_dir: str = DEFAULT_DATA_DIR
    n_train: int | None = None
    # Dev clips for both the NLL and sampling evaluators (None = all, 0 = disable).
    n_eval: int | None = 128
    max_length: int = 8192
    seed: int = 0

    # Training. Constant schedule so eval curves show saturation, not LR decay.
    # One epoch over sft_train = 100 steps at batch 16.
    num_epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 16
    learning_rate: float = 2e-4
    lr_schedule: LRSchedule = "constant"
    lora_rank: int = 32

    # Evaluation (dev split) and checkpointing
    eval_every: int = 25  # 0 = disabled; step 0 gives the baseline
    save_every: int = 50
    max_tokens: int = 8192
    temperature: float = 1.0

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None


def cli_main(cfg: Config) -> None:
    # check_log_dir may prompt via input(); run it before entering asyncio.
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    dataset_builder = ExpressoSFTDatasetBuilder(
        model_name=cfg.model_name,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        n_train=cfg.n_train,
        n_eval=cfg.n_eval,
        max_length=cfg.max_length,
        seed=cfg.seed,
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
        recipe_name="recipe_audio_emotion_sl",
        renderer_name=model_info.get_recommended_renderer_name(cfg.model_name),
        dataset_builder=dataset_builder,
        evaluator_builders=[evaluator_builder] if cfg.eval_every and cfg.n_eval != 0 else [],
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
