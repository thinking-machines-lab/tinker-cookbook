"""
Evaluation functions for SDFT recipe.

Implements the exact evaluation methodology from the SDFT paper
("Self-Distillation Enables Continual Learning", arxiv 2601.19897)
for SciKnowEval and ToolAlpaca benchmarks.
"""

import asyncio
import contextlib
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Science (SciKnowEval) evaluation
# ---------------------------------------------------------------------------


def extract_xml_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> XML tags.

    Matches the paper's eval_science.py:extract_xml_answer exactly.
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def evaluate_science_correctness(responses: list[str], answers: list[str]) -> list[int]:
    """Evaluate science responses via exact match on extracted XML answer."""
    results = []
    for response, answer in zip(responses, answers):
        extracted = extract_xml_answer(response)
        results.append(1 if extracted == answer else 0)
    return results


# ---------------------------------------------------------------------------
# Tool Use (ToolAlpaca) evaluation
# ---------------------------------------------------------------------------


def extract_actions(text: str) -> list[str]:
    """Extract all Action: fields from model response."""
    return re.findall(r"Action:\s*(\w+)", text)


def extract_action_inputs(text: str) -> dict[str, str]:
    """Extract and merge all Action Input: JSON blocks from model response."""
    json_blocks = re.findall(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    combined: dict[str, str] = {}
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            combined.update(parsed)
        except json.JSONDecodeError:
            continue
    return combined


def evaluate_tooluse_correctness(
    responses: list[str],
    golden_answers: list[list[dict[str, str]]],
) -> list[int]:
    """Evaluate tool-use responses via exact match on actions and inputs.

    Matches the paper's eval_tooluse.py:evaluate_correctness exactly.
    """
    results = []
    for response, golden_answer in zip(responses, golden_answers):
        pred_actions = extract_actions(response)
        pred_inputs = extract_action_inputs(response)

        gt_actions = [item["Action"] for item in golden_answer]
        gt_inputs: dict[str, str] = {}
        for item in golden_answer:
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                gt_inputs.update(json.loads(item["Action_Input"]))

        actions_match = Counter(pred_actions) == Counter(gt_actions)
        inputs_match = pred_inputs == gt_inputs
        results.append(1 if (actions_match and inputs_match) else 0)
    return results


# ---------------------------------------------------------------------------
# Evaluator classes for integration with SDFT training loop
# ---------------------------------------------------------------------------


class SciKnowEvalEvaluator(SamplingClientEvaluator):
    """Evaluates SciKnowEval accuracy during training.

    Generates greedy completions for each prompt and checks exact match
    on the extracted <answer> tag, matching the paper's eval methodology.
    """

    def __init__(
        self,
        prompts: Sequence[list[dict[str, str]]],
        answers: list[str],
        renderer: renderers.Renderer,
        max_tokens: int = 2048,
    ):
        self.prompts = prompts
        self.answers = answers
        self.renderer = renderer
        self.max_tokens = max_tokens

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        stop_condition = self.renderer.get_stop_sequences()
        sampling_params = tinker.SamplingParams(
            stop=stop_condition,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

        tasks = []
        for prompt_messages in self.prompts:
            model_input = self.renderer.build_generation_prompt(prompt_messages)  # type: ignore[arg-type]
            tasks.append(
                sampling_client.sample_async(
                    prompt=model_input, num_samples=1, sampling_params=sampling_params
                )
            )

        results = await asyncio.gather(*tasks)
        tokenizer = sampling_client.get_tokenizer()
        responses = [tokenizer.decode(r.sequences[0].tokens) for r in results]

        scores = evaluate_science_correctness(responses, self.answers)
        accuracy = sum(scores) / len(scores) if scores else 0.0

        logger.info(f"SciKnowEval eval: {sum(scores)}/{len(scores)} correct ({accuracy:.2%})")
        return {
            "sciknoweval/accuracy": accuracy,
            "sciknoweval/num_correct": float(sum(scores)),
            "sciknoweval/num_total": float(len(scores)),
        }


class ToolUseEvaluator(SamplingClientEvaluator):
    """Evaluates ToolAlpaca accuracy during training.

    Generates greedy completions and checks exact match on extracted
    Action/Action Input fields, matching the paper's eval methodology.
    """

    def __init__(
        self,
        prompts: list[str],
        golden_answers: list[list[dict[str, str]]],
        renderer: renderers.Renderer,
        max_tokens: int = 1024,
    ):
        self.prompts = prompts
        self.golden_answers = golden_answers
        self.renderer = renderer
        self.max_tokens = max_tokens

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        stop_condition = self.renderer.get_stop_sequences()
        sampling_params = tinker.SamplingParams(
            stop=stop_condition,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

        tasks = []
        for prompt in self.prompts:
            msg: renderers.Message = {"role": "user", "content": prompt}  # type: ignore[typeddict-item]
            model_input = self.renderer.build_generation_prompt([msg])
            tasks.append(
                sampling_client.sample_async(
                    prompt=model_input, num_samples=1, sampling_params=sampling_params
                )
            )

        results = await asyncio.gather(*tasks)
        tokenizer = sampling_client.get_tokenizer()
        responses = [tokenizer.decode(r.sequences[0].tokens) for r in results]

        scores = evaluate_tooluse_correctness(responses, self.golden_answers)
        accuracy = sum(scores) / len(scores) if scores else 0.0

        logger.info(f"ToolUse eval: {sum(scores)}/{len(scores)} correct ({accuracy:.2%})")
        return {
            "tooluse/accuracy": accuracy,
            "tooluse/num_correct": float(sum(scores)),
            "tooluse/num_total": float(len(scores)),
        }


# ---------------------------------------------------------------------------
# Standalone eval runner (for evaluating saved checkpoints)
# ---------------------------------------------------------------------------


async def run_eval(
    model_name: str,
    eval_dataset: str,
    eval_data_path: str,
    renderer_name: str,
    base_url: str | None = None,
    model_path: str | None = None,
    max_tokens: int = 2048,
    output_path: str | None = None,
) -> dict[str, float]:
    """Run standalone evaluation on a model checkpoint.

    Args:
        model_name: Base model name (e.g., "Qwen/Qwen3-8B").
        eval_dataset: "science" or "tooluse".
        eval_data_path: Path to the paper's eval Arrow data directory.
        renderer_name: Renderer name for the model.
        base_url: Optional Tinker service URL.
        model_path: Optional tinker:// checkpoint path. If None, evaluates the base model.
        max_tokens: Max generation tokens.
        output_path: Optional path to save results JSON.
    """
    from datasets import load_from_disk

    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service_client = tinker.ServiceClient(base_url=base_url)
    if model_path:
        sampling_client = service_client.create_sampling_client(
            base_model=model_name, model_path=model_path
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    ds = load_from_disk(eval_data_path)

    if eval_dataset == "science":
        prompts = [row["prompt"] for row in ds]  # type: ignore[union-attr]
        answers = [row["answer"] for row in ds]  # type: ignore[union-attr]
        evaluator = SciKnowEvalEvaluator(prompts, answers, renderer, max_tokens=max_tokens)
    elif eval_dataset == "tooluse":
        prompts = [row["prompt"] for row in ds]  # type: ignore[union-attr]
        golden_answers = [row["golden_answer"] for row in ds]  # type: ignore[union-attr]
        evaluator = ToolUseEvaluator(prompts, golden_answers, renderer, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown eval dataset: {eval_dataset}")

    metrics = await evaluator(sampling_client)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "config": {
                        "model_name": model_name,
                        "model_path": model_path,
                        "eval_dataset": eval_dataset,
                        "max_tokens": max_tokens,
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"Saved eval results to {output_path}")

    return metrics
