"""Code RL task definitions and loading utilities.

Reuses data loading logic from tinker_cookbook's code_rl module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from tinker_cookbook.recipes.code_rl.code_env import (
    _build_question,
    _ensure_dict,
    _load_deepcoder_split,
    _normalize_tests,
)


@dataclass(frozen=True)
class CodeRLTask:
    """A single code RL task with problem statement and test cases."""

    task_id: str
    problem: str
    tests: list[dict[str, Any]]
    starter_code: str | None = None


def load_deepcoder_tasks(
    split: Literal["train", "test"] = "train",
    max_tasks: int | None = None,
    seed: int = 0,
) -> list[CodeRLTask]:
    """
    Load tasks from the DeepCoder dataset (agentica-org/DeepCoder-Preview-Dataset).

    Args:
        split: Which split to load ("train" or "test")
        max_tasks: Maximum number of tasks to load (None for all)
        seed: Random seed for shuffling (train split only)

    Returns:
        List of CodeRLTask instances with normalized test cases
    """
    ds = _load_deepcoder_split(split)
    if split == "train":
        ds = ds.shuffle(seed=seed)

    tasks: list[CodeRLTask] = []
    for i, item in enumerate(ds):
        if max_tasks is not None and len(tasks) >= max_tasks:
            break

        row = cast(dict[str, Any], item)

        # Extract and normalize metadata
        metadata = _ensure_dict(row.get("metadata", {}))

        # Normalize test cases
        raw_tests = row.get("tests") or row.get("ground_truth")
        tests = _normalize_tests(raw_tests, metadata)
        if not tests:
            continue

        # Build problem prompt
        problem = _build_question(row)
        if problem is None:
            continue

        # Extract starter code if present
        starter_code = row.get("starter_code")
        if isinstance(starter_code, str) and not starter_code.strip():
            starter_code = None

        tasks.append(
            CodeRLTask(
                task_id=f"deepcoder_{split}_{i}",
                problem=problem,
                tests=tests,
                starter_code=starter_code if isinstance(starter_code, str) else None,
            )
        )

    return tasks
