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
from tinker_cookbook.recipes.nemotron_cascade.utils import strip_think_blocks
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

logger = logging.getLogger(__name__)


def _get_think_content(text: str) -> str:
    """Extract the content inside <think> blocks (for truncated responses).

    When a model is truncated mid-think, the answer may only exist inside
    the thinking portion. This extracts that content for fallback extraction.
    """
    # First try closed think blocks
    parts = []
    for match in re.finditer(r'<think>([\s\S]*?)</think>', text):
        parts.append(match.group(1))
    # Also grab unclosed think block content (truncated response)
    unclosed = re.search(r'<think>([\s\S]*)$', re.sub(r'<think>[\s\S]*?</think>', '', text))
    if unclosed:
        parts.append(unclosed.group(1))
    return '\n'.join(parts)


def _extract_answer_from_text(text: str) -> str:
    """Core answer extraction logic applied to a text string.

    Tries multiple patterns in order of specificity. The Nemotron-Cascade-2
    MCQA data uses many different answer-format instructions, so we need
    broad coverage:
    1. \\boxed{answer}
    2. <final_answer>X</final_answer>  (XML-style)
    3. ((X))  (double parentheses)
    4. "Option Selected: X" / "Correct Option: X"
    5. "The answer is X" / "Answer: X" / "Final answer: X"
    6. **X** bold single letter / *X* italic single letter
    7. Last standalone capital letter on its own line
    """
    if not text.strip():
        return ""

    # Try boxed format (last match to get final answer if model writes multiple)
    boxed_matches = list(re.finditer(r'\\boxed\{([^}]+)\}', text))
    if boxed_matches:
        return boxed_matches[-1].group(1).strip()

    # Try XML final_answer tags
    xml_match = re.search(r'<final_answer>\s*([^<]+?)\s*</final_answer>', text, re.IGNORECASE)
    if xml_match:
        return xml_match.group(1).strip()

    # Try double parentheses ((X))
    paren_match = re.search(r'\(\(([A-Za-z0-9])\)\)', text)
    if paren_match:
        return paren_match.group(1).strip().upper()

    # Try "Option Selected: X" / "Correct Option: X" patterns (common in data)
    option_match = re.search(
        r'(?:option selected|correct option|selected option)[:\s]+\(?([A-Za-z0-9])\)?',
        text, re.IGNORECASE
    )
    if option_match:
        return option_match.group(1).strip().upper()

    # Try "the answer is" / "answer:" / "final answer:" patterns (last match)
    answer_matches = list(re.finditer(
        r'(?:the answer is|answer:|final answer:?|answer is)\s*\(?([A-Za-z0-9])\)?',
        text, re.IGNORECASE
    ))
    if answer_matches:
        return answer_matches[-1].group(1).strip().upper()

    # Try bold/italic answer pattern
    bold_match = re.search(r'\*\*([A-Z])\*\*', text)
    if bold_match:
        return bold_match.group(1).strip()
    italic_match = re.search(r'(?<!\*)\*([A-Z])\*(?!\*)', text)
    if italic_match:
        return italic_match.group(1).strip()

    # Try last standalone letter (common MCQA format)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^[A-Z]\.?$', line):
            return line.replace('.', '').strip()

    return ""


def extract_answer(response: str, *, include_think: bool = False) -> str:
    """Extract the answer from a model response.

    Strips <think> blocks first, then tries multiple patterns. If
    ``include_think`` is True and no answer is found in the non-thinking
    portion, falls back to searching inside the thinking content. This is
    useful for truncated responses where the model stated an answer during
    reasoning but never produced a formal answer outside <think>.
    """
    # First try the non-thinking portion
    non_think = strip_think_blocks(response)
    answer = _extract_answer_from_text(non_think)
    if answer:
        return answer

    # Fallback: search inside thinking content
    if include_think:
        think_content = _get_think_content(response)
        answer = _extract_answer_from_text(think_content)
        if answer:
            return answer

    # Final fallback
    if non_think:
        return non_think[-10:]
    if include_think:
        think_content = _get_think_content(response)
        if think_content:
            return think_content.strip()[-10:]
    return response.strip()[-10:]


def check_answer(response: str, expected: str, *, include_think: bool = False) -> bool:
    """Check if extracted answer matches expected answer (exact match only)."""
    extracted = extract_answer(response, include_think=include_think)
    # Normalize both
    extracted_norm = extracted.strip().upper().replace('.', '')
    expected_norm = expected.strip().upper().replace('.', '')

    return extracted_norm == expected_norm


_CONCISE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Think briefly, then give your final answer. "
    "Do not over-explain. Keep your reasoning short and focused."
)


class MCQAEnv(Env):
    """Single-turn MCQA environment with exact-match verification."""

    def __init__(
        self,
        prompt_messages: list[dict],
        expected_answer: str,
        renderer: renderers.Renderer,
        category: str = "mcqa",
        system_prompt: str | None = None,
    ):
        self.prompt_messages = prompt_messages
        self.expected_answer = expected_answer
        self.renderer = renderer
        self.category = category
        self.system_prompt = system_prompt

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        messages: list[Message] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.prompt_messages
        )
        return self.renderer.build_generation_prompt(messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        stop_reason = (extra or {}).get("stop_reason")
        overlong = stop_reason == "length"

        if overlong:
            # For truncated responses, the model may have stated the answer
            # during its <think> chain but never produced a formal answer.
            # Search the full response (including thinking) to recover signal.
            # Use the raw token-decoded text since parse_response may not have
            # parsed think blocks for truncated responses (no stop token).
            raw_text = self.renderer.tokenizer.decode(action) if hasattr(self.renderer, 'tokenizer') else content
            correct = check_answer(raw_text, self.expected_answer, include_think=True)
            # Partial credit: overlong-but-correct gets 0.5 to incentivize
            # finishing within the token budget while still providing signal.
            reward = 0.5 if correct else 0.0
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
            if overlong:
                raw_text = self.renderer.tokenizer.decode(action) if hasattr(self.renderer, 'tokenizer') else content
                extracted = extract_answer(raw_text, include_think=True)
            else:
                extracted = extract_answer(content)
            logtree.table_from_dict(
                {
                    "expected": self.expected_answer,
                    "extracted": extracted,
                    "correct": correct,
                    "overlong": overlong,
                    "reward": f"{reward:.2f}",
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
                "overlong": float(overlong),
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
        system_prompt: str | None = None,
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
        self.system_prompt = system_prompt
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
                env_thunk=lambda pm=prompt_messages, ea=expected_answer, cat=category, sp=self.system_prompt: MCQAEnv(
                    prompt_messages=pm,
                    expected_answer=ea,
                    renderer=self.renderer,
                    category=cat,
                    system_prompt=sp,
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
    system_prompt: str | None = _CONCISE_SYSTEM_PROMPT

    async def __call__(self) -> tuple[MCQARLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return MCQARLDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            data_path=self.data_path,
            seed=self.seed,
            system_prompt=self.system_prompt,
        ), None
