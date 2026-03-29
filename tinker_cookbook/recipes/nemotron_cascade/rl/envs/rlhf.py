"""
RLHF environment for Nemotron-Cascade-2 replication.

Uses a generative reward model (GenRM) to score pairwise comparisons of
rollouts.  Default GenRM is Kimi K2.5 (moonshotai/Kimi-K2.5) with thinking
disabled so that the model outputs VERDICT directly without consuming tokens
on chain-of-thought.  Prompts come from HelpSteer3.

Paper hyperparameters (RLHF stage):
  - Reward model: Qwen3-235B-A22B-Thinking as GenRM (we use Kimi K2.5)
  - Data: HelpSteer3 + arena-human-preference-140k + synthetic safety
  - Batch size: 128, group_size: 16
  - LR: 3e-6, KL coefficient: 0.03 (only stage with nonzero KL)
  - Max response length: 16K tokens
  - Steps: ~25
  - Length-normalized reward + quality-gated conciseness bonus
"""

import asyncio
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import chz
import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.renderers import Message, get_text_content
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
from tinker_cookbook.utils.misc_utils import safezip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENRM_MODEL_NAME = "moonshotai/Kimi-K2.5"
GENRM_RENDERER_NAME = "kimi_k25_disable_thinking"
GENRM_MAX_TOKENS = 4096
GENRM_TEMPERATURE = 0.0  # deterministic judging


# ---------------------------------------------------------------------------
# GenRM Prompt
# ---------------------------------------------------------------------------

GENRM_SYSTEM_PROMPT = """\
You are an expert judge evaluating the quality of AI assistant responses.
You will be given a prompt and two responses (Response A and Response B).
Compare them carefully and decide which response is better overall, \
considering helpfulness, accuracy, safety, and clarity.

Output your reasoning, then on the final line output exactly one of:
  VERDICT: A
  VERDICT: B
  VERDICT: TIE
"""

GENRM_USER_TEMPLATE = """\
## Prompt
{prompt}

## Response A
{response_a}

## Response B
{response_b}

Compare Response A and Response B. Which is better? \
After your reasoning, output VERDICT: A or VERDICT: B (or VERDICT: TIE)."""


def _build_genrm_messages(
    prompt_text: str,
    response_a: str,
    response_b: str,
) -> list[Message]:
    """Build the message list for a GenRM pairwise comparison."""
    return [
        {"role": "system", "content": GENRM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GENRM_USER_TEMPLATE.format(
                prompt=prompt_text,
                response_a=response_a,
                response_b=response_b,
            ),
        },
    ]


_VERDICT_RE = re.compile(r"VERDICT:\s*(A|B|TIE)", re.IGNORECASE)


def _parse_verdict(text: str) -> str | None:
    """Extract the verdict from the GenRM output. Returns 'A', 'B', 'TIE', or None."""
    match = _VERDICT_RE.search(text)
    if match:
        return match.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Reward utilities
# ---------------------------------------------------------------------------


def _length_normalized_reward(
    win_rate: float,
    response_length: int,
    mean_length: float,
    quality_threshold: float = 0.5,
    conciseness_weight: float = 0.1,
) -> float:
    """Compute length-normalized reward with quality-gated conciseness bonus.

    Per the paper:
      - Base reward is the win rate from pairwise comparisons.
      - A conciseness bonus is added only when the response is already high quality
        (win_rate > quality_threshold).
      - The bonus scales with how much shorter the response is relative to the group mean.
    """
    reward = win_rate
    if win_rate > quality_threshold and mean_length > 0:
        length_ratio = response_length / mean_length
        # Bonus for being shorter (capped at 0 — no penalty for being long)
        conciseness_bonus = max(0.0, 1.0 - length_ratio)
        reward += conciseness_weight * conciseness_bonus
    return reward


# ---------------------------------------------------------------------------
# GenRM Preference Model
# ---------------------------------------------------------------------------


class GenRMPreferenceModel:
    """Pairwise preference model using a generative reward model (GenRM).

    Sends two responses to the GenRM and parses its verdict.
    Returns +1 if B wins, -1 if A wins, 0 for tie or parse failure.
    """

    def __init__(self, completer: TinkerMessageCompleter):
        self.completer = completer

    async def judge(
        self,
        prompt_text: str,
        response_a: str,
        response_b: str,
    ) -> float:
        """Return +1 (B wins), -1 (A wins), or 0 (tie/error)."""
        messages = _build_genrm_messages(prompt_text, response_a, response_b)
        reply = await self.completer(messages)
        reply_text = get_text_content(reply)
        verdict = _parse_verdict(reply_text)
        if verdict == "A":
            return -1.0
        elif verdict == "B":
            return 1.0
        else:
            if verdict is None:
                logger.warning(f"GenRM returned unparseable verdict: {reply_text[:200]}")
            return 0.0


# ---------------------------------------------------------------------------
# RL Environment
# ---------------------------------------------------------------------------


class RLHFEnv(Env):
    """Single-turn RLHF environment for preference prompts."""

    def __init__(
        self,
        prompt_messages: list[Message],
        renderer: renderers.Renderer,
    ):
        self.prompt_messages = prompt_messages
        self.renderer = renderer

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self.renderer.build_generation_prompt(self.prompt_messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Return zero per-step reward; group-level reward is computed in compute_group_rewards."""
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


# ---------------------------------------------------------------------------
# Group Builder (pairwise GenRM rewards)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLHFGroupBuilder(EnvGroupBuilder):
    """Builds a group of RLHF environments and computes pairwise GenRM rewards.

    Follows the same tournament pattern as PairwisePreferenceGroupBuilder in
    preference_envs.py, but uses a generative reward model instead of a
    classifier-based one.
    """

    prompt_messages: list[Message]
    prompt_text: str  # plain-text version of the prompt for the GenRM
    policy_renderer_name: str
    policy_model_name: str
    genrm_model_name: str
    genrm_renderer_name: str
    genrm_max_tokens: int
    num_envs: int
    matchup_group_size: int = 4
    base_url: str | None = None
    quality_threshold: float = 0.5
    conciseness_weight: float = 0.1

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = get_tokenizer(self.policy_model_name)
        renderer = renderers.get_renderer(self.policy_renderer_name, tokenizer=tokenizer)
        return [RLHFEnv(self.prompt_messages, renderer) for _ in range(self.num_envs)]

    def _create_genrm(self) -> GenRMPreferenceModel:
        """Lazily create the GenRM (not stored as a field to stay pickleable)."""
        genrm_tokenizer = get_tokenizer(self.genrm_model_name)
        genrm_renderer = renderers.get_renderer(self.genrm_renderer_name, tokenizer=genrm_tokenizer)
        service_client = tinker.ServiceClient(base_url=self.base_url)
        genrm_sampling_client = service_client.create_sampling_client(
            base_model=self.genrm_model_name,
        )
        completer = TinkerMessageCompleter(
            sampling_client=genrm_sampling_client,
            renderer=genrm_renderer,
            max_tokens=self.genrm_max_tokens,
            temperature=GENRM_TEMPERATURE,
        )
        return GenRMPreferenceModel(completer)

    @logtree.scope_header_decorator
    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        assert all(len(t.transitions) == 1 for t in trajectory_group)
        n = len(trajectory_group)

        # Parse responses from trajectories
        tokenizer = get_tokenizer(self.policy_model_name)
        renderer = renderers.get_renderer(self.policy_renderer_name, tokenizer=tokenizer)
        response_messages: list[Message] = []
        response_texts: list[str] = []
        is_valid_list: list[bool] = []
        for traj in trajectory_group:
            msg, is_valid = renderer.parse_response(traj.transitions[0].ac.tokens)
            response_messages.append(msg)
            is_valid_list.append(is_valid)
            response_texts.append(get_text_content(msg))

        # Compute response lengths (in characters) for conciseness bonus
        response_lengths = [len(t) for t in response_texts]
        mean_length = sum(response_lengths) / max(len(response_lengths), 1)

        # Log prompt
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=self.prompt_messages))

        # Log each completion
        for idx, (msg, is_valid) in enumerate(zip(response_messages, is_valid_list, strict=True)):
            with logtree.scope_header(f"Completion {idx}"):
                logtree.log_formatter(ConversationFormatter(messages=[msg]))
                logtree.log_text(f"Valid format: {is_valid}, Length: {response_lengths[idx]}")

        # Build pairwise matchup indices (chunked to avoid O(n^2))
        comparison_indices = _get_pairs_chunked(n, self.matchup_group_size)
        logtree.log_text(
            f"Got {n} trajectories, doing {len(comparison_indices)} pairwise matchups "
            f"(matchup_group_size={self.matchup_group_size})."
        )

        # Run GenRM on all pairs concurrently
        genrm = self._create_genrm()
        with logtree.scope_header("Pairwise Comparisons"):
            j_rewards = await asyncio.gather(*[
                genrm.judge(
                    prompt_text=self.prompt_text,
                    response_a=response_texts[i],
                    response_b=response_texts[j],
                )
                for i, j in comparison_indices
            ])

            for idx, ((i, j), reward) in enumerate(
                zip(comparison_indices, j_rewards, strict=True)
            ):
                logtree.log_text(f"Matchup {idx}: ({i} vs {j}) — j_reward: {reward:.2f}")

        # Aggregate win/loss into per-trajectory scores
        win_minus_loss = [0.0] * n
        matchup_count = [0] * n
        for (i, j), j_reward in safezip(comparison_indices, j_rewards):
            win_minus_loss[j] += j_reward
            win_minus_loss[i] -= j_reward
            matchup_count[j] += 1
            matchup_count[i] += 1

        # Compute normalized win rates (map from [-1, 1] to [0, 1])
        win_rates = [
            (wml / mc + 1.0) / 2.0 if mc > 0 else 0.5
            for wml, mc in zip(win_minus_loss, matchup_count)
        ]

        # Apply length-normalized reward with quality-gated conciseness bonus
        results: list[tuple[float, Metrics]] = []
        with logtree.scope_header("Final Rewards"):
            for idx, (wr, length, is_valid) in enumerate(
                zip(win_rates, response_lengths, is_valid_list, strict=True)
            ):
                reward = _length_normalized_reward(
                    win_rate=wr,
                    response_length=length,
                    mean_length=mean_length,
                    quality_threshold=self.quality_threshold,
                    conciseness_weight=self.conciseness_weight,
                )
                # Penalize invalid format
                if not is_valid:
                    reward -= 0.5

                logtree.log_text(
                    f"Trajectory {idx}: win_rate={wr:.3f}, length={length}, "
                    f"reward={reward:.3f}, valid={is_valid}"
                )
                results.append((
                    reward,
                    {
                        "win_rate": wr,
                        "response_length": float(length),
                        "format_valid": float(is_valid),
                    },
                ))

        return results

    def logging_tags(self) -> list[str]:
        return ["rlhf"]


def _get_pairs_chunked(n: int, chunk_size: int) -> list[tuple[int, int]]:
    """Get one-way pairwise indices, chunked to limit total comparisons."""
    out = []
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        for i in range(chunk_start, chunk_end):
            for j in range(i + 1, chunk_end):
                out.append((i, j))
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _extract_prompt_text(messages: list[dict]) -> str:
    """Extract the user prompt text from a list of message dicts."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle structured content (list of parts)
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and "text" in p
                )
    return ""


class RLHFDataset(RLDataset):
    """RLHF dataset combining HelpSteer3 preference prompts.

    Loads prompts from nvidia/HelpSteer3 and constructs single-turn RLHF
    environments for each prompt.
    """

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        policy_renderer_name: str,
        policy_model_name: str,
        genrm_model_name: str = GENRM_MODEL_NAME,
        genrm_renderer_name: str = GENRM_RENDERER_NAME,
        genrm_max_tokens: int = GENRM_MAX_TOKENS,
        base_url: str | None = None,
        seed: int = 0,
        quality_threshold: float = 0.5,
        conciseness_weight: float = 0.1,
        matchup_group_size: int = 4,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.policy_renderer_name = policy_renderer_name
        self.policy_model_name = policy_model_name
        self.genrm_model_name = genrm_model_name
        self.genrm_renderer_name = genrm_renderer_name
        self.genrm_max_tokens = genrm_max_tokens
        self.base_url = base_url
        self.quality_threshold = quality_threshold
        self.conciseness_weight = conciseness_weight
        self.matchup_group_size = matchup_group_size

        logger.info("Loading HelpSteer3 dataset from HuggingFace...")
        ds = load_dataset("nvidia/HelpSteer3", split="train")
        self.ds = cast(Dataset, ds).shuffle(seed=seed)
        logger.info(
            f"RLHF dataset: {len(self.ds)} examples, batch_size={batch_size}, "
            f"group_size={group_size}"
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch index"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(self, row: dict) -> RLHFGroupBuilder | None:
        try:
            # HelpSteer3 stores conversations in a "context" field (list of message dicts).
            # We use all messages up to (but not including) the final assistant turn as
            # the prompt, so the policy generates a fresh response to the last user query.
            context = row.get("context")
            if not context or not isinstance(context, list):
                return None

            # Strip trailing assistant messages so the prompt ends with the last user turn
            prompt_messages: list[Message] = []
            for msg in context:
                prompt_messages.append({"role": msg["role"], "content": msg["content"]})

            # Remove trailing assistant turns so the model generates the response
            while prompt_messages and prompt_messages[-1]["role"] == "assistant":
                prompt_messages.pop()

            if not prompt_messages:
                return None

            # Extract plain-text version of the last user message for the GenRM
            prompt_text = _extract_prompt_text(prompt_messages)
            if not prompt_text:
                return None

            return RLHFGroupBuilder(
                prompt_messages=prompt_messages,
                prompt_text=prompt_text,
                policy_renderer_name=self.policy_renderer_name,
                policy_model_name=self.policy_model_name,
                genrm_model_name=self.genrm_model_name,
                genrm_renderer_name=self.genrm_renderer_name,
                genrm_max_tokens=self.genrm_max_tokens,
                num_envs=self.group_size,
                matchup_group_size=self.matchup_group_size,
                base_url=self.base_url,
                quality_threshold=self.quality_threshold,
                conciseness_weight=self.conciseness_weight,
            )
        except Exception as e:
            logger.warning(f"Failed to parse RLHF row: {e}")
            return None


# ---------------------------------------------------------------------------
# Dataset Builder (chz-serializable config)
# ---------------------------------------------------------------------------


@chz.chz
class RLHFDatasetBuilder(RLDatasetBuilder):
    """Builder for the RLHF dataset with GenRM pairwise rewards."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 16
    seed: int = 0
    genrm_model_name: str = GENRM_MODEL_NAME
    genrm_renderer_name: str = GENRM_RENDERER_NAME
    genrm_max_tokens: int = GENRM_MAX_TOKENS
    base_url: str | None = None
    quality_threshold: float = 0.5
    conciseness_weight: float = 0.1
    matchup_group_size: int = 4

    async def __call__(self) -> tuple[RLHFDataset, None]:
        return RLHFDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            policy_renderer_name=self.renderer_name,
            policy_model_name=self.model_name_for_tokenizer,
            genrm_model_name=self.genrm_model_name,
            genrm_renderer_name=self.genrm_renderer_name,
            genrm_max_tokens=self.genrm_max_tokens,
            base_url=self.base_url,
            seed=self.seed,
            quality_threshold=self.quality_threshold,
            conciseness_weight=self.conciseness_weight,
            matchup_group_size=self.matchup_group_size,
        ), None
