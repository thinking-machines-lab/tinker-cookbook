from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
import numpy as np
import tinker
from tinker import types

from tinker_cookbook import model_info, renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.golf_forecasting.data import (
    GolfForecastExample,
    candidate_labels,
    load_dataset_manifest,
    load_examples,
)
from tinker_cookbook.recipes.golf_forecasting.env import (
    build_messages,
    compute_log_loss,
    parse_forecast_response,
    score_forecast,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


@chz.chz
class GolfForecastEvalConfig:
    model_name: str
    renderer_name: str | None = None
    dataset_manifest_path: str = "tinker_cookbook/example_data/golf_forecasting/dataset_manifest.json"
    heldout_jsonl_path: str | None = None
    checkpoint_url: str | None = None
    base_url: str | None = None
    output_path: str = "tinker_cookbook/recipes/golf_forecasting/results"
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0
    top_k: int = -1
    max_parallel_tasks: int = 32
    n_eval: int | None = None
    include_other_bucket: bool = True


@dataclass(frozen=True)
class ExampleEvalResult:
    example_id: str
    tournament_id: str
    target_winner: str
    target_label: str
    forecast: dict[str, float]
    brier: float
    log_loss: float
    target_prob: float
    top1_correct: float
    top3_contains_target: float
    format_valid: float
    raw_total_probability: float
    unknown_probability_mass: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "tournament_id": self.tournament_id,
            "target_winner": self.target_winner,
            "target_label": self.target_label,
            "forecast": self.forecast,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "target_prob": self.target_prob,
            "top1_correct": self.top1_correct,
            "top3_contains_target": self.top3_contains_target,
            "format_valid": self.format_valid,
            "raw_total_probability": self.raw_total_probability,
            "unknown_probability_mass": self.unknown_probability_mass,
        }


class GolfForecastEvaluator(SamplingClientEvaluator):
    def __init__(self, config: GolfForecastEvalConfig):
        self.config = config
        renderer_name = self.config.renderer_name or model_info.get_recommended_renderer_name(
            self.config.model_name
        )
        tokenizer = get_tokenizer(self.config.model_name)
        self.renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
        if self.config.heldout_jsonl_path is not None:
            self.examples = load_examples(self.config.heldout_jsonl_path)
        else:
            manifest = load_dataset_manifest(self.config.dataset_manifest_path)
            self.examples = load_examples(manifest.heldout_path)
        if self.config.n_eval is not None:
            self.examples = self.examples[: self.config.n_eval]

    def _build_prompt(self, example: GolfForecastExample) -> tinker.ModelInput:
        messages = build_messages(example, include_other_bucket=self.config.include_other_bucket)
        return self.renderer.build_generation_prompt(messages)

    async def _evaluate_one(
        self,
        example: GolfForecastExample,
        sampling_client: tinker.SamplingClient,
        sampling_params: types.SamplingParams,
    ) -> ExampleEvalResult:
        response = await sampling_client.sample_async(
            prompt=self._build_prompt(example),
            num_samples=1,
            sampling_params=sampling_params,
        )
        text = renderers.get_text_content(self.renderer.parse_response(response.sequences[0].tokens)[0])
        allowed = candidate_labels(example) if self.config.include_other_bucket else example.candidate_names
        try:
            forecast, diagnostics = parse_forecast_response(text, allowed_labels=allowed)
            scores = score_forecast(forecast, target_label=example.target_label)
            return ExampleEvalResult(
                example_id=example.example_id,
                tournament_id=example.tournament_id,
                target_winner=example.target_winner,
                target_label=example.target_label,
                forecast=forecast,
                brier=scores["brier"],
                log_loss=scores["log_loss"],
                target_prob=scores["target_prob"],
                top1_correct=scores["top1_correct"],
                top3_contains_target=scores["top3_contains_target"],
                format_valid=1.0,
                raw_total_probability=diagnostics["raw_total_probability"],
                unknown_probability_mass=diagnostics["unknown_probability_mass"],
            )
        except Exception:
            invalid_forecast = {label: 0.0 for label in allowed}
            return ExampleEvalResult(
                example_id=example.example_id,
                tournament_id=example.tournament_id,
                target_winner=example.target_winner,
                target_label=example.target_label,
                forecast=invalid_forecast,
                brier=2.0,
                log_loss=compute_log_loss(invalid_forecast, target_label=example.target_label),
                target_prob=0.0,
                top1_correct=0.0,
                top3_contains_target=0.0,
                format_valid=0.0,
                raw_total_probability=0.0,
                unknown_probability_mass=0.0,
            )

    async def evaluate_examples(
        self, sampling_client: tinker.SamplingClient
    ) -> tuple[list[ExampleEvalResult], dict[str, float]]:
        sampling_params = types.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            stop=self.renderer.get_stop_sequences(),
        )
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def bounded_eval(example: GolfForecastExample) -> ExampleEvalResult:
            async with semaphore:
                return await self._evaluate_one(example, sampling_client, sampling_params)

        results = await asyncio.gather(*(bounded_eval(example) for example in self.examples))
        metrics = {
            "eval/log_loss": float(np.mean([result.log_loss for result in results])),
            "eval/brier": float(np.mean([result.brier for result in results])),
            "eval/target_prob": float(np.mean([result.target_prob for result in results])),
            "eval/top1_accuracy": float(np.mean([result.top1_correct for result in results])),
            "eval/top3_recall": float(np.mean([result.top3_contains_target for result in results])),
            "eval/format_valid_rate": float(np.mean([result.format_valid for result in results])),
        }
        return results, metrics

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        _, metrics = await self.evaluate_examples(sampling_client)
        return metrics


async def run_eval(config: GolfForecastEvalConfig) -> dict[str, float]:
    service_client = tinker.ServiceClient(base_url=config.base_url)
    if config.checkpoint_url:
        sampling_client = service_client.create_sampling_client(
            model_path=config.checkpoint_url,
            base_model=config.model_name,
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    evaluator = GolfForecastEvaluator(config)
    results, metrics = await evaluator.evaluate_examples(sampling_client)

    output_dir = Path(config.output_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    with (output_dir / "predictions.jsonl").open("w") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True))
            handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    cli_config = chz.entrypoint(GolfForecastEvalConfig)
    asyncio.run(run_eval(cli_config))
