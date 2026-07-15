"""Emotion accuracy + WER evaluation for the Expresso audio RL recipe.

``ExpressoEvaluator`` samples one response per clip and reports:

- ``<split>/emotion_accuracy``: exact match on the ``[<style>]`` tag
  (an unparseable response counts as wrong),
- ``<split>/emotion_macro_f1``: F1 per style, macro-averaged -- unlike
  accuracy, this is not dominated by the most frequent styles,
- ``<split>/wer``: corpus-level WER of the parsed transcriptions, over
  format-valid responses only -- a single unparseable rollout is a wall of
  raw text (thousands of insertions) that would dominate the corpus metric,
  and ``format_valid`` already accounts for those failures,
- ``<split>/format_valid``: fraction of responses parsing as
  ``[<style>] <transcription>``.

Per-clip results are written to ``<log_path>/eval_rollouts_<split>_NNN.jsonl``.

Used two ways: as an in-training evaluator (see ``rl_train.py``), and as a
standalone CLI for the before/after fine-tuning comparison:

    # Before: the base model.
    uv run python -m tinker_cookbook.recipes.audio.emotion.evaluate

    # After: a fine-tuned sampler checkpoint from rl_train.py.
    uv run python -m tinker_cookbook.recipes.audio.emotion.evaluate \
        model_path="tinker://<run-id>/sampler_weights/<step>"
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import cast

import chz
import tinker

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.audio.emotion.env import (
    DEFAULT_DATA_DIR,
    STYLES,
    Split,
    load_clips,
    prompt_messages,
    score_response,
)
from tinker_cookbook.recipes.audio.grading import corpus_wer, parse_response_text
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Macro-averaged F1 over (true, predicted) style pairs.

    Styles never seen as either truth or prediction are excluded from the
    average; a prediction outside ``STYLES`` (or empty) only counts against
    the true style's recall.
    """
    from collections import Counter

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for true, pred in pairs:
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    f1s = [2 * tp[s] / (2 * tp[s] + fp[s] + fn[s]) for s in STYLES if tp[s] + fp[s] + fn[s] > 0]
    return sum(f1s) / len(f1s) if f1s else 0.0


Record = dict[str, str | float | bool | None]


def aggregate_metrics(records: list[Record], prefix: str) -> dict[str, float]:
    """The four headline metrics over per-clip records (as produced by
    ``env.score_response`` plus clip identity). Corpus WER covers
    format-valid records only; ``format_valid`` accounts for the rest."""
    valid = [r for r in records if r["format_valid"]]
    return {
        f"{prefix}/emotion_accuracy": sum(bool(r["emotion_correct"]) for r in records)
        / len(records),
        f"{prefix}/emotion_macro_f1": macro_f1(
            [(str(r["emotion"]), str(r["pred_emotion"])) for r in records]
        ),
        f"{prefix}/wer": corpus_wer([str(r["ref"]) for r in valid], [str(r["hyp"]) for r in valid]),
        f"{prefix}/format_valid": len(valid) / len(records),
    }


class ExpressoEvaluator(SamplingClientEvaluator):
    """Samples one response per clip and scores emotion accuracy and WER."""

    def __init__(self, config: ExpressoEvaluatorBuilder):
        self.config = config
        if not model_info.get_model_attributes(config.model_name).is_audio_in:
            raise ValueError(f"Audio input is not supported by {config.model_name!r}; use Inkling.")
        self.renderer = get_renderer(
            model_info.get_recommended_renderer_name(config.model_name),
            get_tokenizer(config.model_name),
        )
        self.eval_clips = list(
            load_clips(config.data_dir, config.split, config.n_eval, config.seed)
        )
        # Pre-render eval prompts once; DMel encoding is CPU work.
        self.eval_prompts = [
            self.renderer.build_generation_prompt(prompt_messages(c)) for c in self.eval_clips
        ]
        self._n_calls = 0

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        params = tinker.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=self.renderer.get_stop_sequences(),
        )
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def sample_one(prompt: tinker.ModelInput) -> str:
            async with semaphore:
                resp = await sampling_client.sample_async(
                    prompt=prompt, num_samples=1, sampling_params=params
                )
            response, _termination = parse_response_text(
                self.renderer, list(resp.sequences[0].tokens)
            )
            return response

        responses = await asyncio.gather(*[sample_one(p) for p in self.eval_prompts])

        rollouts: list[Record] = [
            cast(
                "Record",
                {
                    "id": clip["id"],
                    "emotion": clip["emotion"],
                    "ref": clip["text"],
                    "raw": response,
                    **score_response(clip, response),
                },
            )
            for clip, response in zip(self.eval_clips, responses, strict=True)
        ]
        self._save_rollouts(rollouts)
        return aggregate_metrics(rollouts, self.config.split)

    def _save_rollouts(self, rollouts: list[Record]) -> None:
        log_dir = Path(self.config.log_path).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        # Never overwrite earlier rollout files: a resumed run rebuilds the
        # evaluator (resetting the counter), but e.g. the step-0 baseline
        # rollouts must survive.
        while (
            path := log_dir / f"eval_rollouts_{self.config.split}_{self._n_calls:03d}.jsonl"
        ).exists():
            self._n_calls += 1
        self._n_calls += 1
        path.write_text("\n".join(json.dumps(r) for r in rollouts) + "\n")
        logger.info(f"Wrote {len(rollouts)} eval rollouts -> {path}")


@chz.chz
class ExpressoEvaluatorBuilder:
    model_name: str
    log_path: str
    data_dir: str = DEFAULT_DATA_DIR
    split: Split = "dev"
    n_eval: int | None = 64  # None = the full split
    seed: int = 0
    max_tokens: int = 8192
    temperature: float = 1.0
    max_parallel_tasks: int = 128

    def __call__(self) -> ExpressoEvaluator:
        return ExpressoEvaluator(self)


@chz.chz
class Config:
    """Standalone before/after evaluation on the held-out test split."""

    model_name: str = "thinkingmachines/Inkling"
    model_path: str | None = None  # tinker:// sampler checkpoint; None = base model
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/audio-rl-eval"
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    data_dir: str = DEFAULT_DATA_DIR
    split: Split = "test"
    n_eval: int | None = None  # None = the full split (588 test clips)
    seed: int = 0

    max_tokens: int = 8192
    temperature: float = 1.0
    max_parallel_tasks: int = 128


async def async_main(cfg: Config) -> dict[str, float]:
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    if cfg.model_path is not None:
        sampling_client = service_client.create_sampling_client(model_path=cfg.model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)

    evaluator = ExpressoEvaluatorBuilder(
        model_name=cfg.model_name,
        log_path=cfg.log_path,
        data_dir=cfg.data_dir,
        split=cfg.split,
        n_eval=cfg.n_eval,
        seed=cfg.seed,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        max_parallel_tasks=cfg.max_parallel_tasks,
    )()
    metrics = await evaluator(sampling_client)

    metrics_path = Path(cfg.log_path).expanduser() / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"model_path": cfg.model_path or cfg.model_name, **metrics}
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return metrics


def cli_main(cfg: Config) -> None:
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
