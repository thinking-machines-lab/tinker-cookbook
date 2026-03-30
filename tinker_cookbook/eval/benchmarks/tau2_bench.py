"""tau2-Bench benchmark -- customer service agent task completion (simplified).

Dataset: ``HuggingFaceH4/tau2-bench-data`` on HuggingFace.
Evaluation: Simplified single-turn approach -- given the full conversation
context, the model must predict the correct final action.
Metric: Exact match on the predicted action.
Pattern: Single-turn generate + programmatic grading.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a customer service agent. Given the conversation history and available \
tools below, determine the correct next action to take.

## System Instructions
{system_prompt}

## Available Tools
{tools}

## Conversation History
{conversation}

Based on the conversation, what is the correct action to take? Respond with \
ONLY the action in this exact JSON format:
```json
{{"action": "<action_name>", "arguments": {{...}}}}
```"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_action(text: str) -> dict | None:
    """Extract an action dict from a model response."""
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = text[start:end]
        if '"action"' in candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _normalize_action(action: dict) -> str:
    """Normalize an action dict to a canonical string for comparison."""
    name = str(action.get("action", "")).strip().lower()
    args = action.get("arguments", {})
    if isinstance(args, dict):
        normalized_args = {
            k: str(v).strip().lower() if isinstance(v, str) else v
            for k, v in sorted(args.items())
        }
        return json.dumps({"action": name, "arguments": normalized_args}, sort_keys=True)
    return json.dumps({"action": name}, sort_keys=True)


def _format_conversation(messages: list) -> str:
    """Format a conversation history for the prompt."""
    lines = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        else:
            lines.append(str(msg))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class Tau2BenchEnv(Env):
    """Single-turn env for one tau2-Bench action prediction task."""

    def __init__(self, prompt: str, expected_action: dict | str, renderer: Renderer):
        self.prompt = prompt
        self.expected_action = expected_action
        self.renderer = renderer

    async def initial_observation(self):
        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = self.renderer.tokenizer.decode(action)
        predicted = _extract_action(response)

        if predicted is None:
            correct = False
        elif isinstance(self.expected_action, dict):
            correct = _normalize_action(predicted) == _normalize_action(self.expected_action)
        else:
            pred_name = str(predicted.get("action", "")).strip().lower()
            expected_name = str(self.expected_action).strip().lower()
            correct = pred_name == expected_name

        return StepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(correct)},
            logs={
                "input": self.prompt[:200],
                "expected": str(self.expected_action)[:200],
                "predicted": json.dumps(predicted)[:200] if predicted else "",
                "output": response[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class Tau2BenchBenchmarkBuilder(BenchmarkBuilder):
    """tau2-Bench: customer service agent evaluation.

    TODO: Implement full multi-turn agent interaction with tool dispatch.
    Current implementation uses single-turn action prediction as a proxy.
    The full implementation requires a tool dispatch environment that
    simulates the customer service backend (database queries, actions).
    See https://github.com/sierra-research/tau2-bench for the reference.
    """

    name = "tau2_bench"
    multi_turn = True
    recommended_timeout = 600

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_dataset("HuggingFaceH4/tau2-bench-data", split="test"))
        if config.max_examples is not None:
            ds = ds.select(range(min(config.max_examples, len(ds))))

        envs = []
        for row in ds:
            system_prompt = row.get("system_prompt", row.get("instructions", ""))
            tools = row.get("tools", row.get("available_actions", "[]"))
            if isinstance(tools, list):
                tools = json.dumps(tools, indent=2)
            elif not isinstance(tools, str):
                tools = str(tools)

            conversation = row.get("conversation", row.get("messages", []))
            if isinstance(conversation, str):
                try:
                    conversation = json.loads(conversation)
                except json.JSONDecodeError:
                    pass

            expected_action = row.get("expected_action", row.get("gold_action", row.get("label", None)))
            if expected_action is None:
                continue

            if isinstance(expected_action, str):
                try:
                    expected_action = json.loads(expected_action)
                except json.JSONDecodeError:
                    pass

            if isinstance(conversation, list):
                conversation_str = _format_conversation(conversation)
            else:
                conversation_str = str(conversation)

            prompt = _PROMPT_TEMPLATE.format(
                system_prompt=str(system_prompt)[:2000],
                tools=str(tools)[:3000],
                conversation=conversation_str[:4000],
            )
            envs.append(Tau2BenchEnv(prompt, expected_action, renderer))
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(Tau2BenchBenchmarkBuilder())
