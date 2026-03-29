"""BFCL benchmark -- Berkeley Function Calling Leaderboard.

Dataset: ``gorilla-llm/Berkeley-Function-Calling-Leaderboard`` on HuggingFace.
Metric: Function-calling accuracy via AST matching.

BFCL tests whether a model can correctly generate function calls (tool use)
given function documentation and a user query. We evaluate on the "simple"
subset and check whether the generated function call matches the expected
one by comparing function name and argument values.
"""

from __future__ import annotations

import json
import logging
import re
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import (
    get_text,
    make_completer,
    run_concurrent_eval,
)

logger = logging.getLogger(__name__)


def _extract_function_call(text: str) -> dict | None:
    """Try to extract a function call dict from model output.

    Looks for JSON objects or Python function call syntax.
    """
    # Try JSON extraction -- find the first '{' and parse from there
    start = text.find("{")
    if start != -1:
        # Walk forward to find the matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    break

    # Try to find JSON in code blocks
    code_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if code_match:
        try:
            parsed = json.loads(code_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except json.JSONDecodeError:
            pass

    return None


def _normalize_value(v: object) -> object:
    """Normalize a value for comparison (lowercase strings, sort lists, etc.)."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return sorted(_normalize_value(x) for x in v)
    if isinstance(v, dict):
        return {k: _normalize_value(val) for k, val in v.items()}
    return v


def _match_function_call(generated: dict, expected: dict) -> bool:
    """Check if a generated function call matches the expected one.

    Compares function name and arguments with normalized values.
    """
    # Check function name
    gen_name = generated.get("name", generated.get("function", ""))
    exp_name = expected.get("name", expected.get("function", ""))
    if isinstance(gen_name, str) and isinstance(exp_name, str):
        if gen_name.strip().lower() != exp_name.strip().lower():
            return False

    # Check arguments
    gen_args = generated.get("arguments", generated.get("parameters", {}))
    exp_args = expected.get("arguments", expected.get("parameters", {}))

    if not isinstance(gen_args, dict) or not isinstance(exp_args, dict):
        return False

    # All expected arguments must be present and match
    for key, exp_val in exp_args.items():
        if key not in gen_args:
            return False
        if _normalize_value(gen_args[key]) != _normalize_value(exp_val):
            return False

    return True


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on BFCL (function calling accuracy via AST matching).

    Loads the ``BFCL_v3_simple.json`` subset by default as it is the most
    representative single-function-call scenario.
    """
    try:
        ds = cast(
            Dataset,
            load_dataset(
                "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                data_files="BFCL_v3_simple.json",
                split="train",
            ),
        )
    except Exception as exc:
        logger.warning(f"Could not load BFCL dataset: {exc}. Skipping.")
        return EvalResult(benchmark="bfcl", score=0.0, num_examples=0, num_correct=0)

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        # BFCL rows have: id, question (list of messages), function (list of tool defs),
        # ground_truth (expected function call)
        question_msgs = row.get("question", [])
        functions = row.get("function", [])
        ground_truth = row.get("ground_truth", row.get("expected_output", None))

        if not question_msgs or ground_truth is None:
            return None

        # Parse ground truth
        if isinstance(ground_truth, str):
            try:
                gt_parsed = json.loads(ground_truth)
            except json.JSONDecodeError:
                return None
        elif isinstance(ground_truth, (dict, list)):
            gt_parsed = ground_truth
        else:
            return None

        if isinstance(gt_parsed, list):
            gt_parsed = gt_parsed[0] if gt_parsed else None
        if not isinstance(gt_parsed, dict):
            return None

        # Build the prompt with function definitions
        if isinstance(functions, str):
            func_text = functions
        elif isinstance(functions, list):
            func_text = json.dumps(functions, indent=2)
        else:
            func_text = str(functions)

        # Extract the user query
        if isinstance(question_msgs, list) and question_msgs:
            if isinstance(question_msgs[0], dict):
                user_query = question_msgs[-1].get("content", "")
            else:
                user_query = str(question_msgs[-1])
        else:
            user_query = str(question_msgs)

        prompt = (
            f"You have access to the following functions:\n\n{func_text}\n\n"
            f"User query: {user_query}\n\n"
            "Call the appropriate function with the correct arguments. "
            "Respond with a JSON object containing 'name' and 'arguments' keys."
        )

        messages: list[Message] = [{"role": "user", "content": prompt}]
        try:
            response = await completer(messages)
            content = get_text(response)
            generated = _extract_function_call(content)
            if generated is None:
                correct = False
            else:
                correct = _match_function_call(generated, gt_parsed)
            return {"correct": correct, "input": user_query[:200], "output": content[:500]}
        except Exception as e:
            logger.warning(f"BFCL eval failed: {e}")
            return None

    logger.info(f"BFCL: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0

    logger.info(f"BFCL final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="bfcl",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={"bfcl/accuracy": accuracy},
        examples=valid,
    )
