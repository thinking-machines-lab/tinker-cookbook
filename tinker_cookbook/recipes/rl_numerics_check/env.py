"""Long-context, many-turn counting environment for checking RL training numerics.

The model reads a document one page per turn with a ``read_page`` tool and
reports how many times a target word appears across the document's pages. The
reward is the closeness of the reported count to the true count.

Difficulty is controlled by how much of the task is revealed in the prompt:

- ``num_pages_hint="exact"``: the prompt states the number of pages, which is
  sampled per problem.
- ``num_pages_hint="range"``: the prompt only states a range. The true number of
  pages is a single hidden value shared by every problem, and pages past the end
  still return text, so the model has to learn where the document ends.
- The word is sampled per problem from ``words``. ``word_hint`` sets how much of it
  the prompt reveals: all of it (``"exact"``), only its first and last letters
  (``"first_last"``), or nothing (``"hidden"``). First and last letters identify a
  unique word in ``words``, so the model can learn which word each hint stands for;
  ``"hidden"`` needs a single-word list, which the model has to learn.

Pages are contiguous windows of English Wikipedia text (WikiText-103), so every
count comes from natural text.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal, cast

import chz
import tinker

from tinker_cookbook import model_info
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer, ToolCall, get_text_content
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    InitialObservationOverflow,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import (
    AgentToolMessageEnv,
    ToolResult,
    error_tool_result,
    simple_tool_result,
    tool,
)

logger = logging.getLogger(__name__)

NumPagesHint = Literal["exact", "range"]
WordHint = Literal["exact", "first_last", "hidden"]

# Words that occur roughly 0.5-2.5 times per 1,000 words of WikiText-103 and whose
# rate varies little across topics, so a ~2K-token page usually has a few of them.
# No two share their first and last letters, so the "first_last" hint is unambiguous.
DEFAULT_WORDS: tuple[str, ...] = (
    "after",
    "against",
    "although",
    "before",
    "being",
    "between",
    "both",
    "could",
    "during",
    "however",
    "including",
    "into",
    "known",
    "later",
    "many",
    "most",
    "only",
    "over",
    "several",
    "such",
    "their",
    "then",
    "through",
    "time",
    "under",
    "until",
    "when",
    "while",
)

_WORD_RE = re.compile(r"[a-z]+")
_ANSWER_RE = re.compile(r"answer\s*:\s*\**\s*(-?\d[\d,]*)", re.IGNORECASE)
_WIKITEXT_REPLACEMENTS = ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."))

SYSTEM_PROMPT = "You are a careful assistant who reads documents with the read_page tool."

# Turns allowed beyond num_pages_max, for the final answer and any mistakes. Pages
# exist up to max_turns, so reading past the end of the document is possible.
EXTRA_TURNS = 4
MAX_CORPUS_CHARS = 100_000_000
DEFAULT_MAX_TRAJECTORY_TOKENS = 128 * 1024
FAILED_PARSE_REWARD = -0.1
CONTEXT_OVERFLOW_REWARD = -0.1
# Page ends move forward to the next whitespace, so a problem spans a little more
# than its pages' nominal length.
_PROBLEM_SPAN_SLACK = 1.1
_HIDDEN_NUM_PAGES_SEED = 0
# Used in the prompt to explain whole-word matching, so it must not be a target word.
_EXAMPLE_WORD = "light"
_REJECTED_CALL_MESSAGE = "Only one tool call per response is allowed; this call was not executed."
_REJECTED_CALL_CONTENT = error_tool_result(_REJECTED_CALL_MESSAGE).messages[0]["content"]
# Occasional turns fail to extend when the renderer re-renders a sampled turn slightly
# differently; only warn when most turns of a reasonably long episode fail.
_MIN_TURNS_FOR_EXTENSION_WARNING = 5


def count_word(text: str, word: str) -> int:
    """Count case-insensitive whole-word occurrences of ``word`` in ``text``.

    A word is a maximal run of ASCII letters, so "time's" and "time-based" each
    contain "time" once, while "times" and "lifetime" do not.
    """
    return sum(1 for w in _WORD_RE.findall(text.lower()) if w == word)


def parse_answer(text: str) -> int | None:
    """Return the integer after the last ``Answer:`` in ``text``, if any."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def closeness_reward(predicted: int, true_count: int, tolerance: float) -> float:
    """1 for an exact count, falling linearly to 0 at a relative error of ``tolerance``."""
    relative_error = abs(predicted - true_count) / max(true_count, 1)
    return max(0.0, 1.0 - relative_error / tolerance)


def clean_wikitext_line(line: str) -> str:
    for old, new in _WIKITEXT_REPLACEMENTS:
        line = line.replace(old, new)
    return line.strip()


@functools.cache
def load_wikitext_corpus(max_chars: int = MAX_CORPUS_CHARS) -> str:
    """Load up to ``max_chars`` characters of WikiText-103 train as one string."""
    import datasets

    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    lines: list[str] = []
    total = 0
    for batch in ds.iter(batch_size=10_000):
        for raw in cast(dict[str, list[str]], batch)["text"]:
            line = clean_wikitext_line(raw)
            if not line:
                continue
            lines.append(line)
            total += len(line) + 1
        if total >= max_chars:
            break
    corpus = "\n".join(lines)[:max_chars]
    logger.info(f"Loaded WikiText-103 corpus with {len(corpus):,} characters")
    return corpus


def _next_whitespace(text: str, pos: int) -> int:
    while pos < len(text) and not text[pos].isspace():
        pos += 1
    return pos


def split_pages(corpus: str, start: int, num_pages: int, page_chars: int) -> list[str]:
    """Cut ``num_pages`` consecutive pages of about ``page_chars`` characters.

    Page boundaries are moved forward to the next whitespace so no word is split
    across pages.
    """
    pages: list[str] = []
    pos = start
    for _ in range(num_pages):
        end = _next_whitespace(corpus, min(pos + page_chars, len(corpus)))
        pages.append(corpus[pos:end].strip())
        pos = end
    return pages


@chz.chz
class CountingTaskConfig:
    """What the task reveals and how long it is."""

    num_pages_hint: NumPagesHint = "exact"
    word_hint: WordHint = "exact"
    # With num_pages_hint="exact", N is sampled per problem from this range. With
    # "range", the prompt states this range and N is hidden_num_pages.
    num_pages_min: int = 28
    num_pages_max: int = 36
    # WikiText pages of 10,000 characters are about 2.2K tokens for GLM-5.3 and
    # Qwen3.6, so the defaults give documents of roughly 62K-80K tokens.
    page_chars: int = 10_000
    # The target word of each problem is sampled from this list, e.g. task.words=until
    # for a single word or task.words=until,after for two.
    words: tuple[str, ...] = DEFAULT_WORDS
    # The N shared by every problem when num_pages_hint="range". When unset, it is a
    # fixed draw from [num_pages_min, num_pages_max].
    secret_num_pages: int | None = None
    # Relative count error at which the reward reaches 0. At 1.0, answering a typical
    # count for the word without reading already scores about 0.8.
    reward_tolerance: float = 0.2

    @chz.validate
    def _check(self) -> None:
        if not 1 <= self.num_pages_min <= self.num_pages_max:
            raise ValueError("Need 1 <= num_pages_min <= num_pages_max")
        if self.reward_tolerance <= 0:
            raise ValueError("reward_tolerance must be positive")
        if not self.words:
            raise ValueError("words must be non-empty")
        for word in self.words:
            if not _WORD_RE.fullmatch(word):
                raise ValueError(f"Target words must be lowercase ASCII letters, got {word!r}")
        if len(set(self.words)) != len(self.words):
            raise ValueError("words must not repeat")
        if _EXAMPLE_WORD in self.words:
            raise ValueError(f"{_EXAMPLE_WORD!r} is the prompt's matching example")
        if self.word_hint == "first_last":
            by_hint: dict[tuple[str, str], str] = {}
            for word in self.words:
                other = by_hint.setdefault((word[0], word[-1]), word)
                if other != word:
                    raise ValueError(
                        f"{other!r} and {word!r} share first and last letters, so "
                        'word_hint="first_last" would not identify the word'
                    )
        if self.word_hint == "hidden" and len(self.words) > 1:
            raise ValueError('word_hint="hidden" needs a single word, since nothing identifies it')
        if self.secret_num_pages is not None and not (
            self.num_pages_min <= self.secret_num_pages <= self.num_pages_max
        ):
            raise ValueError("secret_num_pages must lie in [num_pages_min, num_pages_max]")

    @property
    def max_turns(self) -> int:
        return self.num_pages_max + EXTRA_TURNS

    @property
    def hidden_num_pages(self) -> int:
        """The N used when num_pages_hint="range"."""
        if self.secret_num_pages is not None:
            return self.secret_num_pages
        rng = random.Random(_HIDDEN_NUM_PAGES_SEED)
        return rng.randint(self.num_pages_min, self.num_pages_max)


@dataclass(frozen=True)
class CountingProblem:
    word: str
    num_pages: int
    # Every readable page, including pages past the end of the document.
    pages: tuple[str, ...]
    prompt: str
    true_count: int


def build_prompt(config: CountingTaskConfig, word: str, num_pages: int) -> str:
    if config.word_hint == "exact":
        word_phrase = f'the word "{word}"'
    elif config.word_hint == "first_last":
        word_phrase = f'a secret word that starts with "{word[0]}" and ends with "{word[-1]}"'
    else:
        word_phrase = "a secret word"

    if config.num_pages_hint == "exact":
        pages_phrase = (
            f"The document has {num_pages} pages, numbered 1 to {num_pages}. "
            f"Count occurrences across all {num_pages} pages."
        )
    else:
        pages_phrase = (
            f"The document has between {config.num_pages_min} and {config.num_pages_max} "
            "pages, numbered from 1; the exact number is not given. Requests for pages past "
            "the end of the document still return text, but that text is not part of the "
            "document and must not be counted."
        )

    return (
        f"Your task is to count how many times {word_phrase} appears in a document.\n\n"
        f"{pages_phrase}\n\n"
        "Rules:\n"
        "- Read pages with the read_page tool. Call read_page exactly once per response; "
        "additional calls in the same response are rejected.\n"
        "- Matching is case-insensitive and counts whole words only. A word is a maximal "
        'run of letters: for the word "light", "Light", "light\'s" and "light-based" each '
        'count once, but "lights" and "lighthouse" do not count.\n'
        "- When you have finished reading, reply without calling any tool and end your "
        'reply with "Answer: <count>", where <count> is a single integer.'
    )


def sample_problem(
    rng: random.Random,
    corpus: str,
    config: CountingTaskConfig,
) -> CountingProblem:
    word = rng.choice(config.words)
    if config.num_pages_hint == "exact":
        num_pages = rng.randint(config.num_pages_min, config.num_pages_max)
    else:
        num_pages = config.hidden_num_pages

    num_readable = config.max_turns
    span = int(num_readable * config.page_chars * _PROBLEM_SPAN_SLACK)
    if len(corpus) <= span:
        raise ValueError(
            f"Corpus has {len(corpus):,} characters but a problem needs about {span:,}"
        )
    start = rng.randrange(len(corpus) - span)
    line_start = corpus.find("\n", start)
    start = line_start + 1 if 0 <= line_start < start + config.page_chars else start
    pages = split_pages(corpus, start, num_readable, config.page_chars)
    return CountingProblem(
        word=word,
        num_pages=num_pages,
        pages=tuple(pages),
        prompt=build_prompt(config, word, num_pages),
        true_count=sum(count_word(page, word) for page in pages[:num_pages]),
    )


@dataclass
class EpisodeState:
    """Per-episode bookkeeping shared by the tool and the reward."""

    pages_read: set[int] = field(default_factory=set)
    invalid_page_requests: int = 0


class PageReader:
    def __init__(self, pages: Sequence[str], state: EpisodeState):
        self._pages = pages
        self._state = state

    @tool
    async def read_page(
        self, page: Annotated[int, "Page number to read, starting from 1"]
    ) -> ToolResult:
        """Read one page of the document. Call this at most once per response."""
        if not 1 <= page <= len(self._pages):
            self._state.invalid_page_requests += 1
            return error_tool_result(f"Page {page} is not available.", name="read_page")
        self._state.pages_read.add(page)
        return simple_tool_result(f"[Page {page}]\n{self._pages[page - 1]}", name="read_page")


@dataclass
class OneToolCallPerTurnEnv(AgentToolMessageEnv):
    """Runs only the first tool call of each turn and rejects the rest."""

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        messages = await super()._handle_tool_calls(tool_calls[:1])
        for tc in tool_calls[1:]:
            rejection = error_tool_result(
                _REJECTED_CALL_MESSAGE,
                call_id=tc.id or "",
                name=tc.function.name,
                error_type="rejected",
            )
            self.history.extend(rejection.messages)
            messages.extend(rejection.messages)
        return messages


@dataclass
class CountingReward:
    problem: CountingProblem
    state: EpisodeState
    tolerance: float

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        final_text = next(
            (get_text_content(m) for m in reversed(history) if m["role"] == "assistant"), ""
        )
        predicted = parse_answer(final_text)
        true_count = self.problem.true_count
        n = self.problem.num_pages
        pages_read = self.state.pages_read
        rejected = sum(
            1 for m in history if m["role"] == "tool" and m["content"] == _REJECTED_CALL_CONTENT
        )
        metrics = {
            "answer_parsed": float(predicted is not None),
            "true_count": float(true_count),
            "num_pages": float(n),
            "pages_read": float(len(pages_read)),
            "read_all_pages": float(all(p in pages_read for p in range(1, n + 1))),
            "read_past_end": float(any(p > n for p in pages_read)),
            "rejected_tool_calls": float(rejected),
            "invalid_page_requests": float(self.state.invalid_page_requests),
        }
        if predicted is None:
            return 0.0, metrics | {"exact_match": 0.0}
        return closeness_reward(predicted, true_count, self.tolerance), metrics | {
            "exact_match": float(predicted == true_count),
            "abs_error": float(abs(predicted - true_count)),
        }


@functools.cache
def _warn_turns_not_extending() -> None:
    logger.warning(
        "Most turns of an episode did not extend the previous observation and action, so "
        "those turns train as separate sequences with the full context so far. The renderer "
        "likely rewrites history, e.g. strips earlier reasoning; for Qwen3.5 and Qwen3.6, "
        "use renderer_name=qwen3_5_disable_thinking, or qwen3_5_preserve_thinking to keep "
        "reasoning."
    )


class ExtensionCheckedEnv(EnvFromMessageEnv):
    """Logs how often a turn's next observation extends the previous observation and action.

    Consecutive turns merge into one training sequence only when this holds, so a low
    rate means each turn trains separately.
    """

    _last_observation: list[int]
    _turns_checked: int
    _turns_extended: int

    async def initial_observation(
        self,
    ) -> tuple[tinker.ModelInput, StopCondition] | InitialObservationOverflow:
        result = await super().initial_observation()
        self._turns_checked = self._turns_extended = 0
        if not isinstance(result, InitialObservationOverflow):
            self._last_observation = result[0].to_ints()
        return result

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        result = await super().step(action, extra=extra)
        if not result.episode_done:
            next_observation = result.next_observation.to_ints()
            expected_prefix = self._last_observation + list(action)
            self._turns_checked += 1
            self._turns_extended += next_observation[: len(expected_prefix)] == expected_prefix
            self._last_observation = next_observation
        elif self._turns_checked:
            rate = self._turns_extended / self._turns_checked
            result = dataclasses.replace(result, metrics={**result.metrics, "extension_rate": rate})
            if self._turns_checked >= _MIN_TURNS_FOR_EXTENSION_WARNING and rate < 0.5:
                _warn_turns_not_extending()
        return result


def build_env(
    problem: CountingProblem,
    renderer: Renderer,
    task: CountingTaskConfig,
    max_trajectory_tokens: int | None,
    max_generation_tokens: int | None,
) -> EnvFromMessageEnv:
    state = EpisodeState()
    reader = PageReader(problem.pages, state)
    initial_messages = renderer.create_conversation_prefix_with_tools(
        tools=[reader.read_page.to_spec()], system_prompt=SYSTEM_PROMPT
    ) + [{"role": "user", "content": problem.prompt}]
    message_env = OneToolCallPerTurnEnv(
        tools=[reader.read_page],
        initial_messages=initial_messages,
        max_turns=task.max_turns,
        reward_fn=CountingReward(problem, state, task.reward_tolerance),
    )
    return ExtensionCheckedEnv(
        renderer=renderer,
        message_env=message_env,
        failed_parse_reward=FAILED_PARSE_REWARD,
        max_trajectory_tokens=max_trajectory_tokens,
        max_generation_tokens=max_generation_tokens,
        context_overflow_reward=CONTEXT_OVERFLOW_REWARD,
    )


@dataclass(frozen=True)
class CountingEnvGroupBuilder(EnvGroupBuilder):
    problem: CountingProblem
    task: CountingTaskConfig
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    max_trajectory_tokens: int | None
    max_generation_tokens: int | None

    async def make_envs(self) -> Sequence[Env]:
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name_for_tokenizer))
        return [
            build_env(
                self.problem,
                renderer,
                self.task,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["rl_numerics_check"]


class CountingDataset(RLDataset):
    def __init__(
        self,
        corpus: str,
        task: CountingTaskConfig,
        batch_size: int,
        n_batches: int,
        seed: int,
        make_group_builder: functools.partial[CountingEnvGroupBuilder],
    ):
        self.corpus = corpus
        self.task = task
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.seed = seed
        self.make_group_builder = make_group_builder

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        rng = random.Random(f"{self.seed}-{index}")
        return [
            self.make_group_builder(problem=sample_problem(rng, self.corpus, self.task))
            for _ in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class CountingDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str | None = None
    task: CountingTaskConfig = chz.field(default_factory=CountingTaskConfig)
    n_batches: int = 1000
    seed: int = 0
    max_trajectory_tokens: int | None = DEFAULT_MAX_TRAJECTORY_TOKENS
    max_generation_tokens: int | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name_for_tokenizer
        )
        make_group_builder = functools.partial(
            CountingEnvGroupBuilder,
            task=self.task,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            renderer_name=renderer_name,
            group_size=self.group_size,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
        )
        dataset = CountingDataset(
            corpus=load_wikitext_corpus(),
            task=self.task,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            seed=self.seed,
            make_group_builder=make_group_builder,
        )
        return dataset, None
