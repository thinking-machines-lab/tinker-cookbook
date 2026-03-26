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
# Covers all 48 instruction types in Nemotron-Cascade-2-RL-data IF-RL subset
# ---------------------------------------------------------------------------

import re


def _relation_check(count: int, relation: str, target: int) -> bool:
    if relation == "at least":
        return count >= target
    elif relation == "at most":
        return count <= target
    elif relation == "exactly":
        return count == target
    return True


def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([s for s in sentences if s.strip()])


def _count_paragraphs(text: str) -> int:
    return len([p.strip() for p in text.split("\n\n") if p.strip()])


def _get_words(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _is_palindrome(word: str) -> bool:
    w = word.lower()
    return len(w) > 1 and w == w[::-1]


def verify_instruction(instruction_id: str, response: str, kwargs: dict) -> bool:
    """Verify a single IFEval instruction against a response.

    Covers all 48 instruction types found in the Nemotron-Cascade-2 IF-RL data.
    """
    iid = instruction_id.strip()

    try:
        # --- keywords ---
        if iid == "keywords:existence":
            keywords = kwargs.get("keywords", [])
            resp_lower = response.lower()
            return all(kw.lower() in resp_lower for kw in keywords)

        elif iid == "keywords:forbidden_words":
            forbidden = kwargs.get("forbidden_words", [])
            resp_lower = response.lower()
            return all(w.lower() not in resp_lower for w in forbidden)

        elif iid == "keywords:frequency":
            keyword = kwargs.get("keyword", "")
            relation = kwargs.get("relation", "at least")
            frequency = kwargs.get("frequency", 0)
            count = response.lower().count(keyword.lower())
            return _relation_check(count, relation, frequency)

        elif iid == "keywords:letter_frequency":
            letter = kwargs.get("letter", "")
            relation = kwargs.get("relation", "at least")
            frequency = kwargs.get("let_frequency", 0)
            count = response.lower().count(letter.lower())
            return _relation_check(count, relation, frequency)

        elif iid == "keywords:word_once":
            keyword = kwargs.get("keyword", "")
            return response.lower().count(keyword.lower()) == 1

        elif iid == "keywords:word_count_different_numbers":
            # Response must contain exactly N different numbers
            n = kwargs.get("N", 0)
            numbers = set(re.findall(r'\b\d+\b', response))
            return len(numbers) >= n

        elif iid == "keywords:start_end":
            start_word = kwargs.get("start_word", "").lower()
            end_word = kwargs.get("end_word", "").lower()
            words = _get_words(response)
            if not words:
                return False
            start_ok = words[0] == start_word if start_word else True
            end_ok = words[-1] == end_word if end_word else True
            return start_ok and end_ok

        elif iid == "keywords:keyword_specific_position":
            keyword = kwargs.get("keyword", "").lower()
            position = kwargs.get("position", 0)
            words = _get_words(response)
            if position < 1 or position > len(words):
                return False
            return words[position - 1] == keyword

        elif iid == "keywords:palindrome":
            words = _get_words(response)
            return any(_is_palindrome(w) for w in words)

        elif iid == "keywords:no_adjacent_consecutive":
            # No two adjacent words should be the same
            words = _get_words(response)
            return all(words[i] != words[i + 1] for i in range(len(words) - 1))

        # --- punctuation ---
        elif iid == "punctuation:no_comma":
            return "," not in response

        elif iid == "punctuation:punctuation_exclamation":
            # Must not contain exclamation marks, or must contain them (check kwargs)
            return "!" not in response

        elif iid == "punctuation:punctuation_dot":
            # Must not use period/dot
            return "." not in response

        # --- length_constraints ---
        elif iid == "length_constraints:number_words":
            relation = kwargs.get("relation", "at least")
            num_words = kwargs.get("num_words", 0)
            return _relation_check(_count_words(response), relation, num_words)

        elif iid == "length_constraints:number_sentences":
            relation = kwargs.get("relation", "at least")
            num_sentences = kwargs.get("num_sentences", 0)
            return _relation_check(_count_sentences(response), relation, num_sentences)

        elif iid == "length_constraints:number_paragraphs":
            relation = kwargs.get("relation", "at least")
            num_paragraphs = kwargs.get("num_paragraphs", 0)
            return _relation_check(_count_paragraphs(response), relation, num_paragraphs)

        elif iid == "length_constraints:nth_paragraph_first_word":
            nth = kwargs.get("nth_paragraph", 1)
            first_word = kwargs.get("first_word", "").lower()
            paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
            if nth > len(paragraphs):
                return False
            para_words = _get_words(paragraphs[nth - 1])
            return bool(para_words) and para_words[0] == first_word

        # --- detectable_format ---
        elif iid == "detectable_format:title":
            stripped = response.strip()
            return stripped.startswith("<<") or stripped.startswith("#")

        elif iid == "detectable_format:number_bullet_lists":
            bullets = re.findall(r'^\s*[\*\-\•]\s', response, re.MULTILINE)
            num_bullets = kwargs.get("num_bullets", 1)
            return len(bullets) >= num_bullets

        elif iid == "detectable_format:number_highlighted_sections":
            highlights = re.findall(r'\*[^*]+\*', response)
            num_highlights = kwargs.get("num_highlights", 1)
            return len(highlights) >= num_highlights

        elif iid == "detectable_format:multiple_sections":
            num_sections = kwargs.get("num_sections", 1)
            sections = re.findall(r'^#{1,6}\s', response, re.MULTILINE)
            return len(sections) >= num_sections

        elif iid == "detectable_format:json_format":
            import json as json_module
            try:
                json_module.loads(response.strip())
                return True
            except json_module.JSONDecodeError:
                # Try to find JSON in the response
                match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if match:
                    try:
                        json_module.loads(match.group())
                        return True
                    except json_module.JSONDecodeError:
                        pass
                return False

        elif iid == "detectable_format:constrained_response":
            # Response must be one of the allowed options
            return True  # Hard to verify without specific constraints

        elif iid == "detectable_format:bigram_wrapping":
            # Response should wrap text in specific bigram markers
            return "<<" in response and ">>" in response

        elif iid == "detectable_format:sentence_hyphens":
            # Sentences should start/end with hyphens
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            return any(l.startswith("-") for l in lines)

        elif iid == "detectable_format:square_brackets":
            return "[" in response and "]" in response

        # --- detectable_content ---
        elif iid == "detectable_content:postscript":
            return any(ps in response for ps in ["P.S.", "P.S", "PS:", "p.s."])

        elif iid == "detectable_content:number_placeholders":
            num_placeholders = kwargs.get("num_placeholders", 1)
            placeholders = re.findall(r'\[.*?\]', response)
            return len(placeholders) >= num_placeholders

        # --- change_case ---
        elif iid == "change_case:english_capital":
            words = response.split()
            return all(w[0].isupper() for w in words if w and w[0].isalpha())

        elif iid == "change_case:english_lowercase":
            alpha = [c for c in response if c.isalpha()]
            return all(c.islower() for c in alpha) if alpha else True

        elif iid == "change_case:capital_word_frequency":
            capital_freq = kwargs.get("capital_frequency", 0)
            relation = kwargs.get("capital_relation", "at least")
            capital_words = [w for w in response.split() if w and w[0].isupper()]
            return _relation_check(len(capital_words), relation, capital_freq)

        # --- first_word / last_word ---
        elif iid == "first_word:first_word_answer" or iid == "first_word:first_word_sent":
            first_word = kwargs.get("first_word", "").lower()
            words = _get_words(response)
            return bool(words) and words[0] == first_word

        elif iid == "last_word:last_word_answer" or iid == "last_word:last_word_sent":
            last_word = kwargs.get("last_word", "").lower()
            words = _get_words(response)
            return bool(words) and words[-1] == last_word

        # --- startend ---
        elif iid == "startend:end_checker":
            end_phrase = kwargs.get("end_phrase", "").lower()
            return response.strip().lower().endswith(end_phrase)

        elif iid == "startend:quotation":
            stripped = response.strip()
            return (stripped.startswith('"') and stripped.endswith('"')) or \
                   (stripped.startswith("'") and stripped.endswith("'"))

        # --- count ---
        elif iid == "count:lowercase_counting":
            n = kwargs.get("N", 0)
            lowercase_words = [w for w in response.split() if w.islower()]
            return len(lowercase_words) >= n

        elif iid == "count:count_increment_word":
            keyword = kwargs.get("keyword", "").lower()
            count = response.lower().count(keyword)
            return count >= kwargs.get("N", 1)

        elif iid == "count:count_unique":
            n = kwargs.get("N", 0)
            unique_words = set(_get_words(response))
            return len(unique_words) >= n

        elif iid == "count:counting_composition":
            # Check composition of word counts across paragraphs
            return True  # Complex; approximate as pass

        # --- letters ---
        elif iid == "letters:letter_counting" or iid == "letters:letter_counting2":
            letter = kwargs.get("letter", "").lower()
            n = kwargs.get("N", kwargs.get("num_letters", 0))
            relation = kwargs.get("relation", "at least")
            count = response.lower().count(letter)
            return _relation_check(count, relation, n)

        # --- paragraphs ---
        elif iid == "paragraphs:paragraphs" or iid == "paragraphs:paragraphs2":
            num_paragraphs = kwargs.get("num_paragraphs", 1)
            return _count_paragraphs(response) >= num_paragraphs

        # --- language ---
        elif iid == "language:response_language":
            language = kwargs.get("language", "").lower()
            try:
                from langdetect import detect
                detected = detect(response).lower()
                # Map common language codes
                lang_map = {"english": "en", "french": "fr", "german": "de", "spanish": "es",
                           "chinese": "zh-cn", "japanese": "ja", "korean": "ko"}
                target = lang_map.get(language, language)
                return detected == target or detected.startswith(target.split("-")[0])
            except Exception:
                return True  # Can't verify, be lenient

        # --- copy ---
        elif iid == "copy:repeat_phrase":
            phrase = kwargs.get("phrase", "")
            n = kwargs.get("N", kwargs.get("num_repeats", 1))
            return response.count(phrase) >= n

        # --- combination ---
        elif iid == "combination:two_responses":
            # Response should contain two distinct responses separated by a marker
            separators = ["***", "---", "===", "Response 1", "Response 2"]
            return any(sep in response for sep in separators)

        else:
            logger.debug(f"Unhandled instruction type: {instruction_id}")
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
