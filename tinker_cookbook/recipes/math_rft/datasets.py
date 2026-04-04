"""
Dataset loaders for the math RFT recipe.

Supports loading from:
- HuggingFace datasets (gsm8k, math via DigitalLearningGmbH/MATH-lighteval)
- Local Arrow format (pre-downloaded to ~/data/)

Each loader returns a list of problem dicts with keys:
  problem: str, answer: str, level: str (optional), category: str (optional)
"""

import logging
import random

from datasets import Dataset, load_dataset, load_from_disk

from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

logger = logging.getLogger(__name__)


def _parse_math_row(row: dict) -> dict[str, str] | None:
    """Parse a single MATH row into our standard format."""
    try:
        # Try direct answer field first (MATH-500 format), then extract from solution
        answer = row.get("answer") or extract_boxed(row["solution"])
    except (ValueError, KeyError):
        return None

    level_raw = row.get("level", "")
    # Normalize: "Level 5" -> "5", "5" -> "5"
    level = level_raw.replace("Level ", "") if isinstance(level_raw, str) else str(level_raw)

    return {
        "problem": row["problem"],
        "answer": answer,
        "level": level,
        "category": row.get("subject", row.get("type", "")),
    }


def _parse_gsm8k_row(row: dict) -> dict[str, str] | None:
    """Parse a single GSM8K row into our standard format."""
    try:
        answer = extract_gsm8k_final_answer(row["answer"])
    except ValueError:
        return None
    return {"problem": row["question"], "answer": answer, "level": "", "category": ""}


def load_math_problems(
    data_path: str | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load MATH train (7.5K) and MATH-500 test problems.

    Training data comes from DigitalLearningGmbH/MATH-lighteval train split
    (7,500 problems across 5 difficulty levels and 7 categories).

    Test data uses HuggingFaceH4/MATH-500, the standard 500-problem evaluation
    benchmark used by DeepSeek-Math, Open-R1, and other papers.

    Args:
        data_path: If provided, load train from local Arrow at
            {data_path}/math_train. Otherwise downloads from HuggingFace.
        seed: Random seed for shuffling training data.

    Returns:
        (train_data, test_data) where each is a list of problem dicts.
    """
    # Load training data
    if data_path:
        logger.info(f"Loading MATH train from local path: {data_path}/math_train")
        train_ds = load_from_disk(f"{data_path}/math_train")
    else:
        logger.info("Loading MATH train from HuggingFace (DigitalLearningGmbH/MATH-lighteval)...")
        train_ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")

    assert isinstance(train_ds, Dataset)

    # Shuffle training data
    rng = random.Random(seed)
    train_indices = list(range(len(train_ds)))
    rng.shuffle(train_indices)

    train_data = []
    for idx in train_indices:
        parsed = _parse_math_row(train_ds[idx])
        if parsed:
            train_data.append(parsed)

    # Load MATH-500 test set (the standard eval benchmark)
    logger.info("Loading MATH-500 test set from HuggingFace (HuggingFaceH4/MATH-500)...")
    test_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    assert isinstance(test_ds, Dataset)

    test_data = []
    for row in test_ds:
        parsed = _parse_math_row(row)
        if parsed:
            test_data.append(parsed)

    logger.info(f"MATH: {len(train_data)} train, {len(test_data)} test (MATH-500) problems")
    return train_data, test_data


def load_gsm8k_problems(
    seed: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load GSM8K train and test problems.

    Args:
        seed: Random seed for shuffling training data.

    Returns:
        (train_data, test_data) where each is a list of problem dicts.
    """
    ds = load_dataset("openai/gsm8k", name="main")

    rng = random.Random(seed)
    train_rows = list(ds["train"])  # type: ignore[index]
    rng.shuffle(train_rows)

    train_data = []
    for row in train_rows:
        parsed = _parse_gsm8k_row(row)
        if parsed:
            train_data.append(parsed)

    test_data = []
    for row in ds["test"]:  # type: ignore[index]
        parsed = _parse_gsm8k_row(row)
        if parsed:
            test_data.append(parsed)

    logger.info(f"GSM8K: {len(train_data)} train, {len(test_data)} test problems")
    return train_data, test_data
