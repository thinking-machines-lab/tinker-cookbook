"""
Structured Output RL environment for Nemotron-Cascade-2 replication.

Uses the structured_outputs portion of the multi-domain-RL subset.
These are prompts where the model must produce JSON that conforms to
a given schema. Reward is binary: 1 if valid JSON matching schema, 0 otherwise.
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
from tinker_cookbook.renderers import Message
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


def validate_json_against_schema(response: str, schema_str: str) -> tuple[bool, str]:
    """Validate that response contains valid JSON matching the schema.

    Uses ``jsonschema`` for full JSON Schema validation (type checks, nested
    required fields, enum/pattern constraints, etc.).

    Returns (is_valid, reason).
    """
    import re

    from jsonschema import ValidationError, validate

    # Extract JSON from response
    response = response.strip()

    # Try to find JSON block
    json_obj = None
    # Try direct parse
    try:
        json_obj = json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    if json_obj is None:
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            try:
                json_obj = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    # Try to find any JSON object
    if json_obj is None:
        obj_match = re.search(r'\{[\s\S]*\}', response)
        if obj_match:
            try:
                json_obj = json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass

    # Try array
    if json_obj is None:
        arr_match = re.search(r'\[[\s\S]*\]', response)
        if arr_match:
            try:
                json_obj = json.loads(arr_match.group())
            except json.JSONDecodeError:
                pass

    if json_obj is None:
        return False, "No valid JSON found in response"

    # Parse schema
    try:
        schema = json.loads(schema_str)
    except json.JSONDecodeError:
        # If schema is not valid JSON, just check that response is valid JSON
        return True, "Valid JSON (schema unparseable)"

    # Full JSON Schema validation
    try:
        validate(instance=json_obj, schema=schema)
        return True, "Valid JSON matching schema"
    except ValidationError as e:
        # Use the most specific (deepest) validation error message
        return False, f"Schema validation failed: {e.message}"


class StructuredOutputEnv(Env):
    """Single-turn structured output environment."""

    def __init__(
        self,
        prompt_messages: list[dict],
        schema_str: str,
        renderer: renderers.Renderer,
    ):
        self.prompt_messages = prompt_messages
        self.schema_str = schema_str
        self.renderer = renderer

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        messages: list[Message] = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.prompt_messages
        ]
        return self.renderer.build_generation_prompt(messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        stop_reason = (extra or {}).get("stop_reason")
        if stop_reason == "length":
            valid = False
            reason = "Overlong response"
        else:
            valid, reason = validate_json_against_schema(content, self.schema_str)

        reward = 1.0 if valid else 0.0

        with logtree.scope_header("Prompt"):
            prompt_msgs: list[Message] = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.prompt_messages
            ]
            logtree.log_formatter(ConversationFormatter(messages=prompt_msgs))
        with logtree.scope_header("Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {"valid": valid, "reason": reason, "reward": f"{reward:.1f}"},
                caption="Structured output reward",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={"valid": float(valid), "overlong": float(stop_reason == "length") if stop_reason else 0.0},
        )


@dataclass(frozen=True)
class StructuredOutputGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], StructuredOutputEnv]
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return ["structured_output"]


class StructuredOutputRLDataset(RLDataset):
    def __init__(self, batch_size: int, group_size: int, renderer: renderers.Renderer, seed: int = 0):
        logger.info("Loading structured output RL data from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", name="multi-domain-RL", split="train")
        ds = cast(Dataset, ds)
        ds = ds.filter(lambda x: x.get("schema_str") is not None and x["schema_str"] != "")
        self.ds = ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        logger.info(f"Structured output dataset: {len(self.ds)} examples")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_builder(row)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_builder(self, row: dict) -> StructuredOutputGroupBuilder | None:
        try:
            prompt_messages = row["responses_create_params"]["input"]
            schema_str = row["schema_str"]
            return StructuredOutputGroupBuilder(
                env_thunk=lambda pm=prompt_messages, ss=schema_str: StructuredOutputEnv(
                    prompt_messages=pm, schema_str=ss, renderer=self.renderer,
                ),
                num_envs=self.group_size,
            )
        except Exception as e:
            logger.warning(f"Failed to parse structured output row: {e}")
            return None


@chz.chz
class StructuredOutputRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 16
    seed: int = 0

    async def __call__(self) -> tuple[StructuredOutputRLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return StructuredOutputRLDataset(
            batch_size=self.batch_size, group_size=self.group_size,
            renderer=renderer, seed=self.seed,
        ), None
