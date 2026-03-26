"""
Instruction Following RL environment for Nemotron-Cascade-2 replication.

Uses the IF-RL subset of nvidia/Nemotron-Cascade-2-RL-data which contains
IFEval-style prompts with verifiable instruction constraints.

Paper hyperparameters (IF-RL stage):
  - Batch size: 128 prompts
  - Rollouts per prompt (group_size): 16
  - Temperature: 1.0, top-p: 1.0
  - LR: 3e-6 (AdamW), entropy coeff: 0, KL coeff: 0
  - Max response length: 49K tokens
  - Steps: ~180 (with dynamic filtering)
  - Dynamic filtering: remove prompts where all rollouts agree
  - Overlong penalty: zero reward for incomplete responses
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import chz
import tinker
from datasets import Dataset, load_dataset

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


# ---------------------------------------------------------------------------
# IFEval-style instruction verification
# ---------------------------------------------------------------------------

def _check_keyword_inclusion(response: str, keywords: list[str]) -> bool:
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keywords)


def _check_keyword_exclusion(response: str, keywords: list[str]) -> bool:
    response_lower = response.lower()
    return all(kw.lower() not in response_lower for kw in keywords)


def _check_length_constraint(response: str, relation: str, num_words: int) -> bool:
    word_count = len(response.split())
    if relation == "at least":
        return word_count >= num_words
    elif relation == "at most":
        return word_count <= num_words
    return True


def _check_sentence_count(response: str, relation: str, num_sentences: int) -> bool:
    import re
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    count = len(sentences)
    if relation == "at least":
        return count >= num_sentences
    elif relation == "at most":
        return count <= num_sentences
    return True


def _check_paragraph_count(response: str, relation: str, num_paragraphs: int) -> bool:
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    count = len(paragraphs)
    if relation == "at least":
        return count >= num_paragraphs
    elif relation == "at most":
        return count <= num_paragraphs
    return True


def _check_postscript(response: str) -> bool:
    return "P.S." in response or "P.S" in response or "PS:" in response


def _check_placeholder(response: str) -> bool:
    return "[" in response and "]" in response


def _check_title(response: str) -> bool:
    # Check if response starts with a title (wrapped in <<>> or a line ending with newline)
    return response.strip().startswith("<<") or response.strip().startswith("#")


def _check_no_comma(response: str) -> bool:
    return "," not in response


def _check_all_uppercase(response: str) -> bool:
    # Check if the entire response is uppercase (ignoring non-alpha characters)
    alpha_chars = [c for c in response if c.isalpha()]
    return all(c.isupper() for c in alpha_chars) if alpha_chars else True


def _check_all_lowercase(response: str) -> bool:
    alpha_chars = [c for c in response if c.isalpha()]
    return all(c.islower() for c in alpha_chars) if alpha_chars else True


def _check_frequency(response: str, keyword: str, relation: str, frequency: int) -> bool:
    count = response.lower().count(keyword.lower())
    if relation == "at least":
        return count >= frequency
    elif relation == "at most":
        return count <= frequency
    elif relation == "exactly":
        return count == frequency
    return True


def _check_section_by_header(response: str, num_sections: int) -> bool:
    import re
    sections = re.findall(r'^#{1,6}\s', response, re.MULTILINE)
    return len(sections) >= num_sections


def verify_instruction(instruction_id: str, response: str, kwargs: dict) -> bool:
    """Verify a single IFEval instruction against a response.

    This is a simplified verifier covering the most common IFEval instruction types.
    """
    # Normalize instruction_id
    iid = instruction_id.lower().replace(":", "_")

    try:
        if "keywords" in iid and "inclusion" in iid:
            return _check_keyword_inclusion(response, kwargs.get("keywords", []))
        elif "keywords" in iid and "exclusion" in iid:
            return _check_keyword_exclusion(response, kwargs.get("keywords", []))
        elif "length_constraints" in iid and "number_words" in iid:
            return _check_length_constraint(
                response, kwargs.get("relation", "at least"), kwargs.get("num_words", 0)
            )
        elif "number_sentences" in iid:
            return _check_sentence_count(
                response, kwargs.get("relation", "at least"), kwargs.get("num_sentences", 0)
            )
        elif "number_paragraphs" in iid:
            return _check_paragraph_count(
                response, kwargs.get("relation", "at least"), kwargs.get("num_paragraphs", 0)
            )
        elif "postscript" in iid:
            return _check_postscript(response)
        elif "placeholder" in iid:
            return _check_placeholder(response)
        elif "title" in iid:
            return _check_title(response)
        elif "no_comma" in iid:
            return _check_no_comma(response)
        elif "letter_frequency" in iid or "keyword_frequency" in iid:
            return _check_frequency(
                response,
                kwargs.get("keyword", kwargs.get("letter", "")),
                kwargs.get("relation", "at least"),
                kwargs.get("frequency", kwargs.get("let_frequency", 0)),
            )
        elif "change_case" in iid and "english_uppercase" in iid:
            return _check_all_uppercase(response)
        elif "change_case" in iid and "english_lowercase" in iid:
            return _check_all_lowercase(response)
        elif "number_highlighted_sections" in iid or "section" in iid:
            return _check_section_by_header(
                response, kwargs.get("num_sections", kwargs.get("num_highlights", 0))
            )
        else:
            # Unknown instruction type - be lenient
            logger.debug(f"Unknown instruction type: {instruction_id}")
            return True
    except Exception as e:
        logger.warning(f"Error verifying instruction {instruction_id}: {e}")
        return False


def verify_all_instructions(
    response: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict],
) -> tuple[float, dict[str, bool]]:
    """Verify all instructions for a prompt. Returns (fraction_correct, per_instruction_results)."""
    if not instruction_id_list:
        return 1.0, {}

    results = {}
    for inst_id, kw in zip(instruction_id_list, kwargs_list):
        results[inst_id] = verify_instruction(inst_id, response, kw)

    n_correct = sum(results.values())
    fraction = n_correct / len(results)
    return fraction, results


# ---------------------------------------------------------------------------
# RL Environment
# ---------------------------------------------------------------------------


class IFRLEnv(Env):
    """Single-turn instruction-following environment."""

    def __init__(
        self,
        prompt_messages: list[dict],
        instruction_id_list: list[str],
        kwargs_list: list[dict],
        renderer: renderers.Renderer,
    ):
        self.prompt_messages = prompt_messages
        self.instruction_id_list = instruction_id_list
        self.kwargs_list = kwargs_list
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

        # Check for overlong penalty (response didn't complete)
        stop_reason = (extra or {}).get("stop_reason")
        if stop_reason == "length":
            reward = 0.0
            fraction_correct = 0.0
            per_instruction = {}
        else:
            fraction_correct, per_instruction = verify_all_instructions(
                content, self.instruction_id_list, self.kwargs_list
            )
            reward = fraction_correct

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
                    "fraction_correct": f"{fraction_correct:.3f}",
                    "n_instructions": len(self.instruction_id_list),
                    "overlong": stop_reason == "length",
                    "reward": f"{reward:.3f}",
                },
                caption="IF-RL reward",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "fraction_correct": fraction_correct,
                "overlong": float(stop_reason == "length") if stop_reason else 0.0,
            },
        )


@dataclass(frozen=True)
class IFRLGroupBuilder(EnvGroupBuilder):
    """Builds a group of IF-RL environments for the same prompt."""

    env_thunk: Callable[[], IFRLEnv]
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return ["if_rl"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class IFRLDataset(RLDataset):
    """IF-RL dataset from Nemotron-Cascade-2-RL-data."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        seed: int = 0,
    ):
        logger.info("Loading IF-RL dataset from HuggingFace...")
        ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", name="IF-RL", split="train")
        self.ds = cast(Dataset, ds).shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        logger.info(f"IF-RL dataset: {len(self.ds)} prompts, batch_size={batch_size}, group_size={group_size}")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(self, row: dict) -> IFRLGroupBuilder | None:
        try:
            # Extract prompt from responses_create_params.input
            prompt_messages = row["responses_create_params"]["input"]
            instruction_id_list = row.get("instruction_id_list", [])

            # Parse kwargs - handle both list of dicts and list of None
            raw_kwargs = row.get("kwargs", [])
            kwargs_list = []
            for kw in raw_kwargs:
                if kw is None:
                    kwargs_list.append({})
                elif isinstance(kw, str):
                    import json
                    try:
                        kwargs_list.append(json.loads(kw))
                    except json.JSONDecodeError:
                        kwargs_list.append({})
                elif isinstance(kw, dict):
                    kwargs_list.append(kw)
                else:
                    kwargs_list.append({})

            # Ensure kwargs_list matches instruction_id_list length
            while len(kwargs_list) < len(instruction_id_list):
                kwargs_list.append({})

            return IFRLGroupBuilder(
                env_thunk=lambda pm=prompt_messages, iil=instruction_id_list, kl=kwargs_list: IFRLEnv(
                    prompt_messages=pm,
                    instruction_id_list=iil,
                    kwargs_list=kl,
                    renderer=self.renderer,
                ),
                num_envs=self.group_size,
            )
        except Exception as e:
            logger.warning(f"Failed to parse IF-RL row: {e}")
            return None


@chz.chz
class IFRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 16
    seed: int = 0

    async def __call__(self) -> tuple[IFRLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return IFRLDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            seed=self.seed,
        ), None
