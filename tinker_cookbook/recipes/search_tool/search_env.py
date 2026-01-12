"""Search tool utilities.

This module provides utility functions and data types for search-based RL tasks.
Includes dataset loading, answer normalization, and task instructions.

Note: SearchEnv, SearchR1Dataset, and SearchR1DatasetBuilder have been replaced
by the new tool-use implementation in train.py using AgentToolMessageEnv.
"""

import os
import re
import string
from functools import reduce
from pathlib import Path
from typing import Literal, TypedDict, cast

import pandas as pd
from huggingface_hub import hf_hub_download

SEARCH_TASK_INSTRUCTIONS = """You are an expert assistant who solves tasks using a Wikipedia search tool.

Here are instructions for how to solve a problem:
1. Think step by step before calling the tool and after you receive the result of the tool call. Decide what queries to call the tool with.
2. Call the tool with the queries you have decided on.
3. Think step by step again after you receive the result of the tool call. If you have the information you need, you can stop here.
4. Otherwise, come up with new queries that combine information from the previous results.
5. Include your final answer after the "Answer:" prefix. The answer should be between one to five words.

Here is an example of solving a real question:
"Between 2020 and 2025, which year did New York City see the most population growth and how did San Francisco population change in that year?"

1. Think step by step: In order to answer this question, I need to know the population of New York City and San Francisco between 2020 and 2025. I will search for the population of New York City in each year
2. Calling search tool: <tool_call>{"name": "search", "arguments": {"query_list": ["Population New York city between 2020 and 2025"]}}</tool_call> (Output omitted for brevity)
3. Think step by step again: I have the population of New York City in each year, and I see that the population of New York City grew the most in 2024. I need to know the population of San Francisco in 2024. I will search for the population of San Francisco in each year.
<tool_call>{"name": "search", "arguments": {"query_list": ["Population San Francisco between 2023 and 2024"]}}</tool_call> (Output omitted for brevity)
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


class SearchR1Datum(TypedDict):
    question: str
    answer: list[str]
    data_source: str


def process_single_row(row_series: pd.Series) -> SearchR1Datum:
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row_series: DataFrame row containing the original data

    Returns:
        SearchR1Datum: Processed row data in the required format
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
    """Download and process the SearchR1 dataset.

    Args:
        split: Which split to download ("train" or "test")

    Returns:
        List of SearchR1Datum instances
    """
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
