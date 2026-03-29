"""tau2-Bench benchmark -- customer service agent task completion.

Dataset: ``HuggingFaceH4/tau2-bench-data`` on HuggingFace.
Evaluation: Simplified single-turn approach -- given the full conversation
context, the model must predict the correct final action.  Score is exact
match on the predicted action.

Reference: https://github.com/sierra-research/tau2-bench
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


def _extract_action(text: str) -> dict | None:
    """Extract an action dict from a model response."""
    # Try JSON code block first
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try inline JSON -- find outermost braces containing "action"
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
        # Sort keys and lowercase string values for stable comparison
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


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    max_tokens: int = 32768,
    max_examples: int | None = None,
) -> EvalResult:
    """Evaluate on tau2-Bench using simplified single-turn action prediction."""
    ds = cast(Dataset, load_dataset("HuggingFaceH4/tau2-bench-data", split="test"))
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    completer = make_completer(sampling_client, renderer, max_tokens)

    async def eval_one(row: dict) -> dict | None:
        # Extract fields -- dataset schema may vary, handle flexibly
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
            return None

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
        messages: list[Message] = [{"role": "user", "content": prompt}]

        try:
            response = await completer(messages)
            content = get_text(response)
            predicted = _extract_action(content)
        except Exception as e:
            logger.warning(f"tau2-Bench eval failed: {e}")
            return None

        if predicted is None:
            return {"correct": False, "input": prompt[:200], "output": content[:500], "expected": str(expected_action)}

        # Compare normalized actions
        if isinstance(expected_action, dict):
            correct = _normalize_action(predicted) == _normalize_action(expected_action)
        else:
            # Fallback: string comparison on action name
            pred_name = str(predicted.get("action", "")).strip().lower()
            expected_name = str(expected_action).strip().lower()
            correct = pred_name == expected_name

        return {
            "correct": correct,
            "input": prompt[:200],
            "output": content[:500],
            "expected": str(expected_action)[:200],
            "predicted_action": predicted,
        }

    logger.info(f"tau2-Bench: evaluating {len(ds)} samples")
    results = await run_concurrent_eval(list(ds), eval_one)

    valid = [r for r in results if r is not None]
    num_correct = sum(1 for r in valid if r["correct"])
    accuracy = num_correct / len(valid) if valid else 0.0

    logger.info(f"tau2-Bench final: {num_correct}/{len(valid)} = {accuracy:.4f}")

    return EvalResult(
        benchmark="tau2_bench",
        score=accuracy,
        num_examples=len(valid),
        num_correct=num_correct,
        metrics={"tau2_bench/accuracy": accuracy},
        examples=valid,
    )
