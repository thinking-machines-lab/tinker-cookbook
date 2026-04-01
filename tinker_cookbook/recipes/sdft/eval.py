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

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Science (SciKnowEval) evaluation
# ---------------------------------------------------------------------------


def extract_xml_answer(text: str) -> str:
    """Extract answer from model response.

    Tries multiple extraction strategies in order:
    1. <answer>...</answer> XML tags (paper format, Qwen2.5)
    2. "The answer is X" pattern (thinking models like Qwen3.5)
    3. After </think> tag — look for <answer> tags or "The answer is X"
    4. Last single letter A-D in the text after </think> (or full text)
    """
    # Strategy 1: <answer> tags
    if "<answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    # For thinking models, focus on text after </think>
    search_text = text
    if "</think>" in text:
        search_text = text.split("</think>")[-1]

    # Strategy 2: "The answer is X" pattern
    match = re.search(r"[Tt]he answer is\s*([A-D])", search_text)
    if match:
        return match.group(1)

    # Strategy 3: single letter A-D on its own line after </think>
    for line in search_text.strip().split("\n"):
        line = line.strip()
        if line in ("A", "B", "C", "D"):
            return line

    # Strategy 4: last single letter A-D in search_text
    matches = re.findall(r"\b([A-D])\b", search_text)
    if matches:
        return matches[-1]

    return text.strip()


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


# System prompt for thinking models (Qwen3.5): relies on native <think> block,
# asks model to write "The answer is X." for unambiguous parsing.
SCIENCE_SYSTEM_PROMPT_THINKING = (
    "Given a question and four options, please select the right answer. "
    'After your reasoning, write your final answer as "The answer is X." '
    "where X is the letter (A, B, C, or D)."
)


class SciKnowEvalEvaluator(SamplingClientEvaluator):
    """Evaluates SciKnowEval accuracy during training.

    Generates greedy completions for each prompt and checks exact match
    on the extracted answer, supporting both <answer> tags (Qwen2.5)
    and thinking model outputs (Qwen3.5).
    """

    def __init__(
        self,
        prompts: Sequence[list[dict[str, str]]],
        answers: list[str],
        renderer: renderers.Renderer,
        max_tokens: int = 2048,
        system_prompt_override: str | None = None,
    ):
        self.prompts = prompts
        self.answers = answers
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.system_prompt_override = system_prompt_override

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        stop_condition = self.renderer.get_stop_sequences()
        sampling_params = tinker.SamplingParams(
            stop=stop_condition,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

        tasks = []
        for prompt_messages in self.prompts:
            if self.system_prompt_override is not None:
                # Replace system prompt for thinking models
                messages = list(prompt_messages)
                if messages and messages[0].get("role") == "system":
                    messages[0] = {**messages[0], "content": self.system_prompt_override}
                else:
                    messages.insert(0, {"role": "system", "content": self.system_prompt_override})
                model_input = self.renderer.build_generation_prompt(messages)  # type: ignore[arg-type]
            else:
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
