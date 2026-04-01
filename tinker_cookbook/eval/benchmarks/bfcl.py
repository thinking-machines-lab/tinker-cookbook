"""BFCL benchmark -- Berkeley Function Calling Leaderboard.

Dataset: ``gorilla-llm/Berkeley-Function-Calling-Leaderboard`` on HuggingFace.
Metric: Function-calling accuracy via AST matching.
Pattern: Single-turn generate + programmatic grading.

BFCL tests whether a model can correctly generate function calls (tool use)
given function documentation and a user query. We evaluate on the "simple"
subset and check whether the generated function call matches the expected
one by comparing function name and argument values.
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


def _normalize_value(v: object) -> object:
    """Normalize a value for comparison."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return sorted(_normalize_value(x) for x in v)
    if isinstance(v, dict):
        return {k: _normalize_value(val) for k, val in v.items()}
    return v


def _match_function_call(generated: dict, expected: dict) -> bool:
    """Check if a generated function call matches the expected one."""
    gen_name = generated.get("name", generated.get("function", ""))
    exp_name = expected.get("name", expected.get("function", ""))
    if (
        isinstance(gen_name, str)
        and isinstance(exp_name, str)
        and gen_name.strip().lower() != exp_name.strip().lower()
    ):
        return False

    gen_args = generated.get("arguments", generated.get("parameters", {}))
    exp_args = expected.get("arguments", expected.get("parameters", {}))

    if not isinstance(gen_args, dict) or not isinstance(exp_args, dict):
        return False

    for key, exp_val in exp_args.items():
        if key not in gen_args:
            return False
        if _normalize_value(gen_args[key]) != _normalize_value(exp_val):
            return False

    return True


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class BFCLEnv(Env):
    """Single-turn env for one BFCL function-calling problem."""

    def __init__(
        self, prompt: str, user_query: str, expected: dict, renderer: Renderer, example_id: str = ""
    ):
        self.prompt = prompt
        self.user_query = user_query
        self.expected = expected
        self.renderer = renderer
        self.example_id = example_id

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        # Use raw decode — BFCL grades the function call itself, not text content
        response = self.renderer.tokenizer.decode(action)
        generated = _extract_function_call(response)
        if generated is None:
            correct = False
        else:
            correct = _match_function_call(generated, self.expected)
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
    """BFCL: Berkeley Function Calling Leaderboard (simple subset, AST match)."""

    name = "bfcl"

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
            gt_by_id = {row["id"]: row["ground_truth"] for row in gt_ds}
        except Exception as exc:
            logger.warning(f"Could not load BFCL dataset: {exc}.")
            return []

        ds = limit_dataset(ds, config.max_examples)

        envs = []
        for row in ds:
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
                gt_parsed = gt_parsed[0] if gt_parsed else None
            if not isinstance(gt_parsed, dict):
                continue

            # Build prompt
            if isinstance(functions, str):
                func_text = functions
            elif isinstance(functions, list):
                func_text = json.dumps(functions, indent=2)
            else:
                func_text = str(functions)

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

            example_id = make_example_id("bfcl", user_query)
            envs.append(BFCLEnv(prompt, user_query, gt_parsed, renderer, example_id=example_id))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(BFCLBenchmarkBuilder())
