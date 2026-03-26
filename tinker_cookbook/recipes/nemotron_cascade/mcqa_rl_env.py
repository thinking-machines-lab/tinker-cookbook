"""
Multi-domain RL environment for Nemotron-Cascade-2 replication.

Uses the MCQA (multiple-choice question answering) portion of the
multi-domain-RL subset. These are STEM questions with expected_answer
fields that can be verified by exact match.

Paper hyperparameters (Multi-domain RL stage):
  - Data mix: ~55% MCQA (STEM), ~30% tool calling, ~15% structured output
  - Batch size: 128, Rollouts: 16, Temp: 1.0
  - LR: 3e-6, KL coeff: 0
  - Max response length: 49K tokens
  - Steps: ~70
"""

import json
import logging
import math
import re
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


def extract_answer(response: str) -> str:
    """Extract the answer from a model response.

    Tries multiple patterns:
    1. \\boxed{answer}
    2. The answer is (X)
    3. **Answer: X**
    4. Last single letter/option on its own line
    """
    # Try boxed format
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try "the answer is" pattern
    answer_match = re.search(
        r'(?:the answer is|answer:|final answer:?)\s*\(?([A-Za-z0-9])\)?',
        response, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip().upper()

    # Try bold answer pattern
    bold_match = re.search(r'\*\*([A-Z])\*\*', response)
    if bold_match:
        return bold_match.group(1).strip()

    # Try last standalone letter (common MCQA format)
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^[A-Z]\.?$', line):
            return line.replace('.', '').strip()

    # Fallback: look for any single capital letter option
    options = re.findall(r'\b([A-Z])\b', response[-200:])
    if options:
        return options[-1]

    return response.strip()[-10:]  # Last 10 chars as fallback


def check_answer(response: str, expected: str) -> bool:
    """Check if extracted answer matches expected answer."""
    extracted = extract_answer(response)
    # Normalize both
    extracted_norm = extracted.strip().upper().replace('.', '')
    expected_norm = expected.strip().upper().replace('.', '')

    # Direct match
    if extracted_norm == expected_norm:
        return True

    # Check if expected is contained in extracted (for longer answers)
    if expected_norm in extracted_norm or extracted_norm in expected_norm:
        return True

    return False


class MCQAEnv(Env):
    """Single-turn MCQA environment with exact-match verification."""

    def __init__(
        self,
        prompt_messages: list[dict],
        expected_answer: str,
        renderer: renderers.Renderer,
        category: str = "mcqa",
    ):
        self.prompt_messages = prompt_messages
        self.expected_answer = expected_answer
        self.renderer = renderer
        self.category = category

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

        # Check for overlong penalty
        stop_reason = (extra or {}).get("stop_reason")
        if stop_reason == "length":
            correct = False
            reward = 0.0
        else:
            correct = check_answer(content, self.expected_answer)
            reward = 1.0 if correct else 0.0

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
            extracted = extract_answer(content)
            logtree.table_from_dict(
                {
                    "expected": self.expected_answer,
                    "extracted": extracted,
                    "correct": correct,
                    "overlong": stop_reason == "length",
                    "reward": f"{reward:.1f}",
                },
                caption="MCQA reward",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "correct": float(correct),
                "overlong": float(stop_reason == "length") if stop_reason else 0.0,
            },
        )


@dataclass(frozen=True)
class MCQAGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], MCQAEnv]
    num_envs: int
    dataset_name: str = "mcqa"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


class MCQARLDataset(RLDataset):
    """MCQA dataset from Nemotron-Cascade-2-RL-data multi-domain subset."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        data_path: str | None = None,
        seed: int = 0,
    ):
        if data_path:
            logger.info(f"Loading MCQA data from {data_path}")
            rows = []
            with open(data_path) as f:
                for line in f:
                    row = json.loads(line)
                    # Only keep examples with expected_answer (MCQA)
                    if row.get("expected_answer"):
                        rows.append(row)
            self.ds = Dataset.from_list(rows).shuffle(seed=seed)
        else:
            logger.info("Loading multi-domain RL data from HuggingFace...")
            from datasets import load_dataset
            ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", name="multi-domain-RL", split="train")
            ds = cast(Dataset, ds)
            # Filter to MCQA examples only
            ds = ds.filter(lambda x: x.get("expected_answer") is not None and x["expected_answer"] != "")
            self.ds = ds.shuffle(seed=seed)

        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        logger.info(f"MCQA dataset: {len(self.ds)} examples, batch_size={batch_size}, group_size={group_size}")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_builder(row)) is not None  # pyright: ignore
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_builder(self, row: dict) -> MCQAGroupBuilder | None:
        try:
            prompt_messages = row["responses_create_params"]["input"]
            expected_answer = row["expected_answer"]
            category = row.get("category", "mcqa") or "mcqa"

            return MCQAGroupBuilder(
                env_thunk=lambda pm=prompt_messages, ea=expected_answer, cat=category: MCQAEnv(
                    prompt_messages=pm,
                    expected_answer=ea,
                    renderer=self.renderer,
                    category=cat,
                ),
                num_envs=self.group_size,
                dataset_name=category,
            )
        except Exception as e:
            logger.warning(f"Failed to parse MCQA row: {e}")
            return None


@chz.chz
class MCQARLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 16
    data_path: str | None = None
    seed: int = 0

    async def __call__(self) -> tuple[MCQARLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return MCQARLDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            data_path=self.data_path,
            seed=self.seed,
        ), None
