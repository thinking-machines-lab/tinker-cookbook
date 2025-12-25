import asyncio
import logging
import os
import random
import re
import string
from functools import partial, reduce
from pathlib import Path
from typing import Literal, Sequence, TypedDict, cast

import chz
import pandas as pd
import tinker
from huggingface_hub import hf_hub_download
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.tool_use.search.tools import ChromaToolClient, ChromaToolClientConfig
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

_CONNECTION_SEMAPHORE = asyncio.Semaphore(128)

SEARCH_TOOL_SYSTEM_PROMPT = """
You are an expert assistant who solves tasks using a Wikipedia search tool.
Tool calling. Execute the tool by wrapping calls in <tool_call>...</tool_call>

The search tool you are given has the following schema:
```
{
    "name": "search",
    "title": "Wikipedia search",
    "description": "Searches Wikipedia for relevant information based on the given query.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of fully-formed semantic queries. The tool will return search results for each query.",
            }
        },
        "required": ["query_list"],
    },
    "outputSchema": {
        "type": "string",
        "description": "The search results in JSON format",
    },
}
```

Here are instructions for how to solve a problem:
1. Think step by step before calling the tool and after you receive the result of the tool call. Decide what queries to call the tool with.
2. Call the tool with the queries you have decided on.
3. Think step by step again after you receive the result of the tool call. If you have the information you need, you can stop here.
4. Otherwise, come up with new queries that combine information from the previous results.
5. Include your final answer after the "Answer:" prefix. The answer should be between one to five words.

Here is an example of solving a real question:
“Between 2020 and 2025, which year did New York City see the most population growth and how did San Francisco population change in that year?”

1. Think step by step: In order to answer this question, I need to know the population of New York City and San Francisco between 2020 and 2025. I will search for the population of New York City in each year
2. Calling search tool: <tool_call>{"name": "search", "args": {"query_list": ["Population New York city between 2020 and 2025"]}}</tool_call> (Output omitted for brevity)
3. Think step by step again: I have the population of New York City in each year, and I see that the population of New York City grew the most in 2024. I need to know the population of San Francisco in 2024. I will search for the population of San Francisco in each year.
<tool_call>{"name": "search", "args": {"query_list": ["Population San Francisco between 2023 and 2024"]}}</tool_call> (Output omitted for brevity)
4. Answer: The population of New York City grew the most in 2024, and the population of San Francisco changed by XXXX in 2024.
"""


def normalize_answer(s: str) -> str:
    """Normalize answer by lowercasing, removing punctuation, articles, and fixing whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    # Apply transformations in order using reduce
    transformations = [lower, remove_punc, remove_articles, white_space_fix]
    return reduce(lambda text, func: func(text), transformations, s)


class SearchEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: list[str],
        chroma_tool_client: ChromaToolClient,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        timeout: float = 1.0,
        max_num_calls: int = 4,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem: str = problem
        self.answer: list[str] = answer
        self.timeout: float = timeout
        self.chroma_tool_client: ChromaToolClient = chroma_tool_client
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.past_messages: list[renderers.Message] = convo_prefix.copy() if convo_prefix else []
        self.current_num_calls: int = 0
        self.max_num_calls: int = max_num_calls

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        self.past_messages = convo.copy()
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    def get_question(self) -> str:
        return self.problem

    def _extract_answer(self, sample_str: str) -> str | None:
        if "Answer:" not in sample_str:
            return None
        message_pars = sample_str.split("Answer:")
        if len(message_pars) != 2:
            return None
        return message_pars[1].strip()

    def check_answer(self, sample_str: str) -> bool:
        model_answer = self._extract_answer(sample_str)
        if model_answer is None or len(self.answer) == 0:
            return False

        for gold_answer in self.answer:
            if normalize_answer(model_answer) == normalize_answer(gold_answer):
                return True
        return False

    def check_format(self, sample_str: str) -> bool:
        return self._extract_answer(sample_str) is not None

    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        return " OR ".join(self.answer) if self.answer else "N/A"

    async def call_search_tool(self, tool_call: renderers.ToolCall) -> list[renderers.Message]:
        async with _CONNECTION_SEMAPHORE:
            return await self.chroma_tool_client.invoke(tool_call)

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)

        self.past_messages.append(message)

        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
            failure_result = StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
            )
            # Check if tool_calls list is not empty
            if not tool_calls:
                return failure_result
            if tool_calls[0].function.name == "search":
                self.current_num_calls += 1
                if self.current_num_calls > self.max_num_calls:
                    return failure_result
                # NOTE(tianyi): seems wasteful: we should share the client somehow
                try:
                    tool_return_message = await self.call_search_tool(tool_calls[0])
                    self.past_messages.extend(tool_return_message)
                except Exception as e:
                    logger.error(f"Error calling search tool: {repr(e)}")
                    return failure_result

                next_observation = self.renderer.build_generation_prompt(self.past_messages)
                if next_observation.length > self.max_trajectory_tokens:
                    return failure_result
                return StepResult(
                    reward=0.0,
                    episode_done=False,
                    next_observation=self.renderer.build_generation_prompt(self.past_messages),
                    next_stop_condition=self.stop_condition,
                )
            else:
                return failure_result
        else:
            content = renderers.ensure_text(message["content"])
            correct_format = float(parse_success) and float(self.check_format(content))
            correct_answer = float(self.check_answer(content))
            total_reward = self.format_coef * (correct_format - 1) + correct_answer
            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format": correct_format,
                    "correct": correct_answer,
                },
            )

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "system",
                "content": SEARCH_TOOL_SYSTEM_PROMPT,
            },
        ]


class SearchR1Datum(TypedDict):
    question: str
    answer: list[str]
    data_source: str


def process_single_row(row_series: pd.Series) -> SearchR1Datum:
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    import numpy as np

    row = row_series.to_dict()
    question: str = row.get("question", "")

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # NOTE(tianyi)
    # I hate datasets with mixed types but it is what it is.
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()

    assert isinstance(ground_truth, list)
    for item in ground_truth:
        assert isinstance(item, str)
    ground_truth = cast(list[str], ground_truth)
    return {
        "question": question,
        "answer": ground_truth,
        "data_source": row["data_source"],
    }


def download_search_r1_dataset(split: Literal["train", "test"]) -> list[SearchR1Datum]:
    hf_repo_id: str = "PeterJinGo/nq_hotpotqa_train"
    parquet_filename: str = f"{split}.parquet"
    # TODO(tianyi): make download dir configurable for release
    user = os.getenv("USER", "unknown")
    assert user is not None
    tmp_download_dir = Path("/tmp") / user / "data" / hf_repo_id / split
    tmp_download_dir.mkdir(parents=True, exist_ok=True)

    local_parquet_filepath = hf_hub_download(
        repo_id=hf_repo_id,
        filename=parquet_filename,
        repo_type="dataset",
        local_dir=tmp_download_dir,
        local_dir_use_symlinks=False,
    )

    df_raw = pd.read_parquet(local_parquet_filepath)

    return df_raw.apply(process_single_row, axis=1).tolist()


class SearchR1Dataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        # tool args
        chroma_tool_client: ChromaToolClient,
        # optional args
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        split: Literal["train", "test"] = "train",
        subset_size: int | None = None,
        max_trajectory_tokens: int = 32 * 1024,
    ):
        self.batch_size: int = batch_size
        self.group_size: int = group_size
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.renderer: renderers.Renderer = renderer
        self.convo_prefix: list[renderers.Message] | None = convo_prefix
        self.chroma_tool_client: ChromaToolClient = chroma_tool_client
        self.seed: int = seed
        self.split: Literal["train", "test"] = split
        self.ds: list[SearchR1Datum] = download_search_r1_dataset(split)
        # shuffle with seed
        rng = random.Random(self.seed)
        rng.shuffle(self.ds)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            self._make_env_group_builder(row, self.group_size)
            for row in self.ds[index * self.batch_size : (index + 1) * self.batch_size]
        ]

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size

    def _make_env_group_builder(self, row: SearchR1Datum, group_size: int) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                SearchEnv,
                row["question"],
                row["answer"],
                self.chroma_tool_client,
                self.renderer,
                convo_prefix=self.convo_prefix,
                max_trajectory_tokens=self.max_trajectory_tokens,
            ),
            num_envs=group_size,
        )


@chz.chz
class SearchR1DatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    chroma_tool_config: ChromaToolClientConfig
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_eval_size: int = 1024
    max_trajectory_tokens: int = 32 * 1024

    async def __call__(self) -> tuple[SearchR1Dataset, None]:
        if self.convo_prefix == "standard":
            convo_prefix = SearchEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        chroma_tool_client = await ChromaToolClient.create(self.chroma_tool_config)

        train_dataset = SearchR1Dataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            chroma_tool_client=chroma_tool_client,
            convo_prefix=convo_prefix,
            split="train",
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
        )
        return (train_dataset, None)
