"""BFCL benchmark -- Berkeley Function Calling Leaderboard.

Dataset: ``gorilla-llm/Berkeley-Function-Calling-Leaderboard`` on HuggingFace.
Metric: Function-calling accuracy via JSON argument matching.
Pattern: Single-turn generate + programmatic grading.

BFCL tests whether a model can correctly generate function calls (tool use)
given function documentation and a user query. We evaluate on the "simple"
subset and check whether the generated function call matches the expected
one by comparing function name and argument values against the allowed answers.
This JSON-only evaluator remains experimental; it does not implement the full
BFCL AST evaluation protocol or model-specific function-name conversions.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.eval.benchmarks._common import (
    _resolve_trust_remote_code,
    limit_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Function call extraction and matching
# ---------------------------------------------------------------------------


def _extract_function_call(text: str) -> dict | None:
    """Try to extract a function call dict from model output."""
    start = text.find("{")
    if start != -1:
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


_PARAMETER_TYPES = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}


def _match_value(value: object, expected: object, schema: dict) -> bool:
    expected_type = _PARAMETER_TYPES.get(schema.get("type", ""))
    # An empty string also marks omission, not a value of a non-string parameter.
    if expected == "" and expected_type not in (None, str):
        return False
    if expected_type is float and type(value) is int:
        value = float(value)
    if type(value) is not expected_type and type(value) is not type(expected):
        return False
    if isinstance(value, dict) and isinstance(expected, dict):
        return _match_arguments(value, expected, schema)
    if isinstance(value, list) and isinstance(expected, list):
        return len(value) == len(expected) and all(
            _match_value(actual, answer, schema.get("items", {}))
            for actual, answer in zip(value, expected, strict=True)
        )
    if isinstance(value, str) and isinstance(expected, str) and expected_type in (None, str):

        def normalize(text: str) -> str:
            return re.sub(r"[ ,./\-_*^]", "", text).lower().replace("'", '"')

        return normalize(value) == normalize(expected)
    return value == expected


def _match_arguments(generated: dict, expected: dict, schema: dict) -> bool:
    """Match BFCL's parameter-to-allowed-values mappings, including nested dicts."""
    properties = schema.get("properties", {})
    if generated.keys() - expected.keys() or set(schema.get("required", [])) - generated.keys():
        return False
    for key, alternatives in expected.items():
        if not isinstance(alternatives, list) or not alternatives:
            return False
        if key not in generated:
            if "" not in alternatives:
                return False
            continue
        if properties and key not in properties:
            return False
        if not any(
            _match_value(generated[key], answer, properties.get(key, {})) for answer in alternatives
        ):
            return False
    return True


def _match_function_call(generated: dict, expected: dict, function: dict) -> bool:
    """Check a JSON call against a BFCL simple reference and function signature."""
    gen_name = generated.get("name", generated.get("function"))
    exp_name = function.get("name")
    if not isinstance(gen_name, str) or not gen_name or gen_name != exp_name:
        return False
    if len(expected) != 1:
        return False
    gen_args = generated.get("arguments", generated.get("parameters"))
    exp_args = next(iter(expected.values()))
    if not isinstance(gen_args, dict) or not isinstance(exp_args, dict):
        return False
    return _match_arguments(gen_args, exp_args, function["parameters"])


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class BFCLEnv(Env):
    """Single-turn env for one BFCL function-calling problem."""

    def __init__(
        self,
        prompt: str,
        user_query: str,
        expected: dict,
        renderer: Renderer,
        example_id: str = "",
        *,
        function: dict,
    ):
        self.prompt = prompt
        self.user_query = user_query
        self.expected = expected
        self.function = function
        self.renderer = renderer
        self.example_id = example_id

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        # Use raw decode — BFCL grades the function call itself, not text content
        response = str(self.renderer.tokenizer.decode(action))
        generated = _extract_function_call(response)
        if generated is None:
            correct = False
        else:
            correct = _match_function_call(generated, self.expected, self.function)
        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct)},
            logs={
                "example_id": self.example_id,
                "input": self.user_query[:200],
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class BFCLBenchmarkBuilder(BenchmarkBuilder):
    """BFCL: Berkeley Function Calling Leaderboard (simple subset, JSON calls)."""

    name = "bfcl"
    experimental = True

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
        try:
            kwargs: dict = {}
            trust = _resolve_trust_remote_code(None)
            if trust:
                kwargs["trust_remote_code"] = True
            ds = cast(
                Dataset,
                load_dataset(repo, data_files="BFCL_v3_simple.json", split="train", **kwargs),
            )
            # Ground truth is in a separate file — load and index by id
            gt_ds = cast(
                Dataset,
                load_dataset(
                    repo, data_files="possible_answer/BFCL_v3_simple.json", split="train", **kwargs
                ),
            )
            gt_by_id = {row["id"]: row["ground_truth"] for row in gt_ds}  # type: ignore[index]
        except Exception as exc:
            logger.warning(f"Could not load BFCL dataset: {exc}.")
            return []

        ds = limit_dataset(ds, config.max_examples)

        envs = []
        for row in ds:
            row = dict(row)
            question_msgs = row.get("question", [])
            functions = row.get("function", [])
            row_id = row.get("id", "")
            ground_truth = row.get("ground_truth", gt_by_id.get(row_id))

            if not question_msgs or ground_truth is None:
                continue

            # Parse ground truth
            if isinstance(ground_truth, str):
                try:
                    gt_parsed = json.loads(ground_truth)
                except json.JSONDecodeError:
                    continue
            elif isinstance(ground_truth, (dict, list)):
                gt_parsed = ground_truth
            else:
                continue

            if isinstance(gt_parsed, list):
                gt_parsed = gt_parsed[0] if len(gt_parsed) == 1 else None
            if not isinstance(gt_parsed, dict) or len(gt_parsed) != 1:
                continue

            if isinstance(functions, str):
                try:
                    functions = json.loads(functions)
                except json.JSONDecodeError:
                    continue
            if not isinstance(functions, list) or len(functions) != 1:
                continue
            function = functions[0]
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            if not isinstance(function.get("parameters"), dict):
                continue

            if isinstance(question_msgs, list) and isinstance(question_msgs[0], list):
                question_msgs = question_msgs[0] if len(question_msgs) == 1 else []
            if not question_msgs or not isinstance(question_msgs[-1], dict):
                continue
            user_query = question_msgs[-1].get("content")
            if not isinstance(user_query, str) or not user_query:
                continue

            func_text = json.dumps(functions, indent=2)
            prompt = (
                f"You have access to the following functions:\n\n{func_text}\n\n"
                f"User query: {user_query}\n\n"
                "Call the appropriate function with the correct arguments. "
                "Respond with a JSON object containing 'name' and 'arguments' keys."
            )

            example_id = make_example_id("bfcl", user_query)
            envs.append(
                BFCLEnv(
                    prompt,
                    user_query,
                    gt_parsed,
                    renderer,
                    example_id=example_id,
                    function=function,
                )
            )
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(BFCLBenchmarkBuilder())
