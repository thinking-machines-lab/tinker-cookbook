"""
Long-context RL environment for Nemotron-Cascade-2 replication.

Trains on long-context QA tasks using the SCROLLS dataset (Qasper subset).
The model receives a long document + question and generates an answer.
An LLM judge (Qwen3.5-397B-A17B) evaluates the answer quality.

Paper hyperparameters (Long-context RL stage):
  - Data: Long-context QA, input limited to 32K tokens
  - Reward: LLM judge (originally Qwen3-235B-Instruct) evaluates answers
  - Batch size: 128, Rollouts: 16, Temp: 1.0
  - LR: 3e-6, KL coeff: 0
  - Max response length: 49K tokens
  - Steps: ~30
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
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating the quality of answers to questions about documents. \
You will be given a question, the relevant context from the document, and the model's answer. \
Evaluate whether the answer is correct, complete, and well-supported by the context.

Respond with ONLY a single integer score from 0 to 10:
- 0: Completely wrong, irrelevant, or no answer
- 1-3: Partially addresses the question but mostly incorrect or missing key information
- 4-6: Partially correct, captures some key points but has notable gaps or errors
- 7-9: Mostly correct and well-supported by the context, minor issues only
- 10: Fully correct, complete, and well-supported by the context

Output ONLY the integer score, nothing else."""

JUDGE_USER_TEMPLATE = """\
Question: {question}

Reference context from the document:
{context}

Model's answer:
{answer}

Score (0-10):"""

# Maximum number of context characters to send to the judge to keep the
# judge prompt manageable. The full document is given to the *student*
# model; the judge only needs enough context to verify the answer.
_MAX_JUDGE_CONTEXT_CHARS = 12_000


def _parse_judge_score(response_text: str) -> float:
    """Extract a 0-10 integer score from the judge's response and normalise to [0, 1]."""
    # Find the LAST integer in the response.  The judge (a thinking model) may
    # emit <think> reasoning that contains stray numbers before the final score.
    matches = re.findall(r'\b(\d{1,2})\b', response_text)
    if matches:
        score = int(matches[-1])
        return min(max(score, 0), 10) / 10.0
    # Fallback: couldn't parse -> neutral reward
    logger.warning(f"Could not parse judge score from: {response_text!r}")
    return 0.0


async def get_llm_judge_reward(
    question: str,
    context: str,
    answer: str,
    judge_completer: TinkerMessageCompleter,
) -> tuple[float, str]:
    """Query the LLM judge and return (normalised_reward, raw_judge_response)."""
    # Truncate context for the judge prompt
    truncated_context = context[:_MAX_JUDGE_CONTEXT_CHARS]
    if len(context) > _MAX_JUDGE_CONTEXT_CHARS:
        truncated_context += "\n[...context truncated...]"

    messages: list[Message] = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                question=question,
                context=truncated_context,
                answer=answer,
            ),
        },
    ]

    try:
        judge_response = await judge_completer(messages)
        response_text = judge_response.get("content", "")
        if not isinstance(response_text, str):
            response_text = str(response_text)
        reward = _parse_judge_score(response_text)
        return reward, response_text
    except Exception as e:
        logger.warning(f"LLM judge call failed: {e}")
        return 0.0, f"ERROR: {e}"


# ---------------------------------------------------------------------------
# RL Environment
# ---------------------------------------------------------------------------


class LongContextRLEnv(Env):
    """Single-turn long-context QA environment with LLM judge reward."""

    def __init__(
        self,
        question: str,
        context: str,
        renderer: renderers.Renderer,
        reference_answer: str = "",
    ):
        self.question = question
        self.context = context
        self.renderer = renderer
        self.reference_answer = reference_answer

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Build the user prompt with the long document and question
        user_content = (
            f"Read the following document carefully and answer the question at the end.\n\n"
            f"--- DOCUMENT ---\n{self.context}\n--- END DOCUMENT ---\n\n"
            f"Question: {self.question}\n\n"
            f"Provide a clear, concise answer based only on the document above."
        )
        messages: list[Message] = [
            {"role": "user", "content": user_content},
        ]
        return self.renderer.build_generation_prompt(messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        # Check for overlong penalty (response didn't complete)
        stop_reason = (extra or {}).get("stop_reason")
        if stop_reason == "length":
            reward = 0.0
            judge_response_text = "N/A (overlong)"
        else:
            # Reward is deferred to compute_group_rewards (which has access
            # to the judge completer). Store the answer on the env for later.
            reward = 0.0
            judge_response_text = "deferred"

        # Store the model answer for the group builder's reward computation
        self._model_answer = content
        self._stop_reason = stop_reason

        # Logging
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(
                ConversationFormatter(
                    messages=[
                        {"role": "user", "content": f"[Document: {len(self.context)} chars] Q: {self.question}"}
                    ]
                )
            )
        with logtree.scope_header("Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "overlong": float(stop_reason == "length") if stop_reason else 0.0,
                "answer_length": float(len(content.split())),
            },
        )


@dataclass(frozen=True)
class LongContextRLGroupBuilder(EnvGroupBuilder):
    """Builds a group of long-context QA envs and scores them with an LLM judge."""

    env_thunk: Callable[[], LongContextRLEnv]
    num_envs: int
    # Judge configuration (stored as serialisable primitives for pickle safety)
    judge_model_name: str = "Qwen/Qwen3.5-397B-A17B"
    judge_renderer_name: str = "qwen3_5"
    judge_max_tokens: int = 512
    judge_temperature: float = 0.0
    judge_base_url: str | None = None

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Score each trajectory using the LLM judge."""
        # Lazily create judge completer (not stored as field for pickle safety)
        judge_tokenizer = get_tokenizer(self.judge_model_name)
        judge_renderer = renderers.get_renderer(self.judge_renderer_name, tokenizer=judge_tokenizer)
        service_client = tinker.ServiceClient(base_url=self.judge_base_url)
        judge_sampling_client = await service_client.create_sampling_client_async(
            base_model=self.judge_model_name,
        )
        judge_completer = TinkerMessageCompleter(
            sampling_client=judge_sampling_client,
            renderer=judge_renderer,
            max_tokens=self.judge_max_tokens,
            temperature=self.judge_temperature,
        )

        # Identify which trajectories need judge calls vs can be skipped
        judge_tasks: list[tuple[int, LongContextRLEnv]] = []
        results: list[tuple[float, Metrics] | None] = [None] * len(trajectory_group)

        for idx, (traj, env) in enumerate(zip(trajectory_group, env_group)):
            assert isinstance(env, LongContextRLEnv)
            stop_reason = getattr(env, "_stop_reason", None)
            if stop_reason == "length":
                results[idx] = (0.0, {"judge_reward": 0.0, "overlong": 1.0})
            elif not getattr(env, "_model_answer", "").strip():
                results[idx] = (0.0, {"judge_reward": 0.0, "empty_answer": 1.0})
            else:
                judge_tasks.append((idx, env))

        # Run all judge calls concurrently
        if judge_tasks:
            judge_results = await asyncio.gather(*[
                get_llm_judge_reward(
                    question=env.question,
                    context=env.context,
                    answer=getattr(env, "_model_answer", ""),
                    judge_completer=judge_completer,
                )
                for _, env in judge_tasks
            ])

            for (idx, env), (reward, judge_text) in zip(judge_tasks, judge_results):
                with logtree.scope_header("LLM Judge"):
                    logtree.table_from_dict(
                        {
                            "judge_raw": judge_text[:200],
                            "judge_reward": f"{reward:.2f}",
                            "question": env.question[:100],
                        },
                        caption="Long-context judge reward",
                    )
                results[idx] = (reward, {"judge_reward": reward})

        return [r for r in results if r is not None]

    def logging_tags(self) -> list[str]:
        return ["longctx_rl"]


# ---------------------------------------------------------------------------
# Dataset — SCROLLS / Qasper long-context QA
# ---------------------------------------------------------------------------


def _load_longcontext_data(seed: int, max_input_tokens: int, tokenizer_name: str) -> Dataset:
    """Load long-context QA data from NarrativeQA.

    Falls back to multiple datasets if primary is unavailable.
    """
    # Try NarrativeQA first (works reliably)
    try:
        logger.info("Loading NarrativeQA dataset from HuggingFace...")
        ds = load_dataset("deepmind/narrativeqa", split="test")
        ds = cast(Dataset, ds)
        logger.info(f"NarrativeQA: {len(ds)} examples")
        return ds.shuffle(seed=seed)
    except Exception as e:
        logger.warning(f"NarrativeQA failed: {e}")

    # Fallback: try LongBench
    try:
        logger.info("Trying LongBench/qasper...")
        ds = load_dataset("THUDM/LongBench", "qasper", split="test")
        ds = cast(Dataset, ds)
        logger.info(f"LongBench/qasper: {len(ds)} examples")
        return ds.shuffle(seed=seed)
    except Exception as e:
        logger.warning(f"LongBench failed: {e}")

    raise RuntimeError("No long-context dataset available")


def _parse_longcontext_example(row: dict) -> tuple[str, str, str] | None:
    """Parse a long-context QA row into (question, context, reference_answer).

    Supports NarrativeQA, LongBench/qasper, and SCROLLS formats.
    """
    # NarrativeQA format
    if "document" in row and isinstance(row["document"], dict):
        doc = row["document"]
        # summary and text can be dicts with a "text" key, or strings
        summary = doc.get("summary", "")
        if isinstance(summary, dict):
            summary = summary.get("text", "")
        text = doc.get("text", "")
        if isinstance(text, dict):
            text = text.get("text", "")
        context = summary or text or ""
        question_obj = row.get("question", {})
        question = question_obj.get("text", "") if isinstance(question_obj, dict) else str(question_obj)
        answers = row.get("answers", [])
        ref = answers[0].get("text", "") if answers and isinstance(answers[0], dict) else ""
        if question and context:
            return question, str(context), ref
        return None

    # LongBench format
    if "context" in row and "input" in row:
        return row.get("input", ""), row.get("context", ""), row.get("answers", [""])[0] if row.get("answers") else ""

    # SCROLLS format (input = question\ncontext)
    raw_input = row.get("input", "")
    if raw_input:
        parts = raw_input.split("\n", 1)
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip(), row.get("output", "")

    return None


class LongContextRLDataset(RLDataset):
    """Long-context QA dataset for RL training with LLM judge reward."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        judge_model_name: str,
        judge_renderer_name: str,
        judge_base_url: str | None,
        tokenizer_name: str,
        max_input_tokens: int = 32_768,
        seed: int = 0,
    ):
        ds = _load_longcontext_data(
            seed=seed,
            max_input_tokens=max_input_tokens,
            tokenizer_name=tokenizer_name,
        )
        # Parse into (question, context, answer) tuples
        parsed: list[dict] = []
        for row in ds:
            result = _parse_longcontext_example(row)  # pyright: ignore[reportArgumentType]
            if result is not None:
                q, c, a = result
                parsed.append({"question": q, "context": c, "reference_answer": a})

        self.examples = parsed
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.judge_model_name = judge_model_name
        self.judge_renderer_name = judge_renderer_name
        self.judge_base_url = judge_base_url
        logger.info(
            f"Long-context QA dataset: {len(self.examples)} examples, "
            f"batch_size={batch_size}, group_size={group_size}"
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.examples))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            self._make_env_group_builder(self.examples[i])
            for i in range(batch_start, batch_end)
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.examples) / self.batch_size)

    def _make_env_group_builder(self, example: dict) -> LongContextRLGroupBuilder:
        question = example["question"]
        context = example["context"]
        reference_answer = example["reference_answer"]

        return LongContextRLGroupBuilder(
            env_thunk=lambda q=question, c=context, ra=reference_answer: LongContextRLEnv(
                question=q,
                context=c,
                renderer=self.renderer,
                reference_answer=ra,
            ),
            num_envs=self.group_size,
            judge_model_name=self.judge_model_name,
            judge_renderer_name=self.judge_renderer_name,
            judge_base_url=self.judge_base_url,
        )


@chz.chz
class LongContextRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 16
    max_input_tokens: int = 32_768
    judge_model_name: str = "Qwen/Qwen3.5-397B-A17B"
    judge_renderer_name: str = "qwen3_5"
    judge_base_url: str | None = None
    seed: int = 0

    async def __call__(self) -> tuple[LongContextRLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return LongContextRLDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            judge_model_name=self.judge_model_name,
            judge_renderer_name=self.judge_renderer_name,
            judge_base_url=self.judge_base_url,
            tokenizer_name=self.model_name_for_tokenizer,
            max_input_tokens=self.max_input_tokens,
            seed=self.seed,
        ), None
