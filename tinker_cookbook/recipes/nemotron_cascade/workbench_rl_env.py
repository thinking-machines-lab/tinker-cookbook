"""
Workbench Tool-Calling RL environment for Nemotron-Cascade-2 replication.

Uses the workbench portion of multi-domain-RL data (5.2K examples).
Tasks involve calling the right workplace tools (email, calendar, CRM, etc.).
Reward: fraction of correct tool calls matching ground_truth.

Categories: email, analytics, calendar, project_management, CRM.
"""

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import chz
import tinker
from datasets import Dataset

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Message, ToolSpec
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

logger = logging.getLogger(__name__)


def _normalize_args(args_str: str) -> dict:
    """Parse and normalize tool call arguments."""
    if isinstance(args_str, dict):
        return args_str
    try:
        return json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def check_tool_calls(
    model_tool_calls: list[dict] | None,
    ground_truth: list[dict],
) -> tuple[float, dict]:
    """Compare model's tool calls against ground truth.

    Returns (reward, details).
    Reward is fraction of ground truth calls that were correctly made.
    """
    if not ground_truth:
        return 1.0, {"reason": "no ground truth"}

    if not model_tool_calls:
        return 0.0, {"reason": "no tool calls made", "expected": len(ground_truth)}

    # Extract model call names and args
    model_calls = []
    for tc in model_tool_calls:
        if hasattr(tc, 'function'):
            name = tc.function.name
            args = _normalize_args(tc.function.arguments)
        elif isinstance(tc, dict):
            func = tc.get("function", tc)
            name = func.get("name", "")
            args = _normalize_args(func.get("arguments", "{}"))
        else:
            continue
        model_calls.append({"name": name, "arguments": args})

    # Check each ground truth call
    matched = 0
    for gt in ground_truth:
        gt_name = gt.get("name", "")
        gt_args = _normalize_args(gt.get("arguments", "{}"))

        for mc in model_calls:
            if mc["name"] == gt_name:
                # Check if arguments match (flexible: allow subset match)
                args_match = True
                for k, v in gt_args.items():
                    if str(mc["arguments"].get(k)) != str(v):
                        args_match = False
                        break
                if args_match:
                    matched += 1
                    break

    fraction = matched / len(ground_truth)
    return fraction, {
        "matched": matched,
        "total_expected": len(ground_truth),
        "total_model_calls": len(model_calls),
        "model_call_names": [mc["name"] for mc in model_calls],
        "expected_names": [gt["name"] for gt in ground_truth],
    }


class WorkbenchEnv(Env):
    """Tool-calling workbench environment."""

    def __init__(
        self,
        prompt_messages: list[dict],
        tools: list[dict],
        ground_truth: list[dict],
        renderer: renderers.Renderer,
        category: str = "workbench",
    ):
        self.prompt_messages = prompt_messages
        self.tools = tools
        self.ground_truth = ground_truth
        self.renderer = renderer
        self.category = category

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Build messages with tool definitions
        messages: list[Message] = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.prompt_messages
        ]

        # Convert tools to ToolSpec format for the renderer
        tool_specs: list[ToolSpec] = []
        for t in self.tools:
            func = t.get("function", t)
            tool_specs.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })

        # Use renderer's tool-aware prompt building if available
        if hasattr(self.renderer, 'create_conversation_prefix_with_tools'):
            prefix = self.renderer.create_conversation_prefix_with_tools(
                tool_specs,
                system_prompt=messages[0]["content"] if messages and messages[0]["role"] == "system" else "",
            )
            # Remove system message if it was included in prefix
            if messages and messages[0]["role"] == "system":
                messages = messages[1:]
            all_messages = prefix + messages
        else:
            all_messages = messages

        return self.renderer.build_generation_prompt(all_messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)

        # Extract tool calls from response
        tool_calls = message.get("tool_calls", [])

        stop_reason = (extra or {}).get("stop_reason")
        if stop_reason == "length":
            reward = 0.0
            details = {"reason": "overlong"}
        else:
            reward, details = check_tool_calls(tool_calls, self.ground_truth)

        # Logging
        prompt_msgs: list[Message] = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.prompt_messages
        ]
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=prompt_msgs))
        with logtree.scope_header("Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "reward": f"{reward:.3f}",
                    "category": self.category,
                    **{k: str(v)[:100] for k, v in details.items()},
                },
                caption="Tool calling reward",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "correct": reward,
                "has_tool_calls": float(bool(tool_calls)),
                "overlong": float(stop_reason == "length") if stop_reason else 0.0,
            },
        )


@dataclass(frozen=True)
class WorkbenchGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], WorkbenchEnv]
    num_envs: int
    dataset_name: str = "workbench"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


class WorkbenchRLDataset(RLDataset):
    def __init__(self, batch_size: int, group_size: int, renderer: renderers.Renderer, seed: int = 0):
        logger.info("Loading workbench RL data...")
        from datasets import load_dataset
        ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", name="multi-domain-RL", split="train")
        ds = cast(Dataset, ds)
        ds = ds.filter(lambda x: x.get("environment_name") == "workbench" and x.get("ground_truth"))
        self.ds = ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        logger.info(f"Workbench dataset: {len(self.ds)} examples")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end
        return [
            b for row in self.ds.select(range(batch_start, batch_end))
            if (b := self._make_builder(row)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_builder(self, row: dict) -> WorkbenchGroupBuilder | None:
        try:
            rcp = row["responses_create_params"]
            prompt_messages = rcp["input"]
            tools = rcp.get("tools", [])
            ground_truth = row["ground_truth"]
            category = row.get("category", "workbench")

            return WorkbenchGroupBuilder(
                env_thunk=lambda pm=prompt_messages, t=tools, gt=ground_truth, cat=category: WorkbenchEnv(
                    prompt_messages=pm, tools=t, ground_truth=gt,
                    renderer=self.renderer, category=cat,
                ),
                num_envs=self.group_size,
                dataset_name=category or "workbench",
            )
        except Exception as e:
            logger.warning(f"Failed to parse workbench row: {e}")
            return None


@chz.chz
class WorkbenchRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 16
    seed: int = 0

    async def __call__(self) -> tuple[WorkbenchRLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return WorkbenchRLDataset(
            batch_size=self.batch_size, group_size=self.group_size,
            renderer=renderer, seed=self.seed,
        ), None
