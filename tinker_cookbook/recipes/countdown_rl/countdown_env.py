"""Countdown task environment for RL training.

The Countdown task asks the model to reach a target number by combining
a set of 3-4 input numbers using basic arithmetic operations (+, -, *, /).
Each number can be used at most once.

Dataset: https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
"""

import math
import re
from collections.abc import Sequence
from functools import partial
from typing import Literal

import chz
import tinker
from datasets import load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter


def evaluate_countdown_expression(
    expression: str, available_nums: list[int], target: int
) -> tuple[bool, float]:
    """Check if an arithmetic expression correctly reaches the target using only available numbers.

    Args:
        expression: An arithmetic expression string (e.g. "44 + 19 + 35").
        available_nums: The numbers that are allowed to be used.
        target: The target number to reach.

    Returns:
        A tuple of (is_correct, partial_score) where:
        - is_correct: True if the expression evaluates to the target using valid numbers.
        - partial_score: A float in [0, 1] giving partial credit:
          - 0.0 if the expression is invalid or uses wrong numbers
          - 0.3 if valid numbers but wrong result
          - 0.3 + up to 0.3 proximity bonus (closer to target = higher)
          - 1.0 if exactly correct
    """
    try:
        # Only allow basic arithmetic operators, digits, spaces, and parens
        if not re.match(r"^[\d\s\+\-\*/\(\)\.]+$", expression):
            return False, 0.0

        # Extract all numbers used in the expression
        used_nums = [int(n) for n in re.findall(r"\d+", expression)]

        # Check that each used number is available (respecting multiplicity)
        remaining = list(available_nums)
        for n in used_nums:
            if n in remaining:
                remaining.remove(n)
            else:
                return False, 0.0

        # Evaluate the expression safely
        result = eval(expression)  # noqa: S307
        if abs(result - target) < 1e-6:
            return True, 1.0

        # Partial credit: valid expression with correct numbers but wrong result
        # Proximity bonus: closer to target gives more credit (max 0.3 bonus)
        if target != 0:
            relative_error = abs(result - target) / abs(target)
            proximity = max(0.0, 1.0 - relative_error)  # 1.0 when exact, 0.0 when 100%+ off
        else:
            proximity = 1.0 if abs(result) < 1e-6 else 0.0
        return False, 0.3 + 0.3 * proximity
    except Exception:
        return False, 0.0


def extract_answer(response: str) -> str | None:
    """Extract the answer from a model response.

    Looks for content inside \\boxed{} first, then falls back to the last
    line containing arithmetic operators.
    """
    # Try \boxed{} format first
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Fallback: find the last line that looks like an arithmetic expression
    for line in reversed(response.strip().splitlines()):
        line = line.strip()
        if re.search(r"\d+\s*[\+\-\*/]", line):
            # Clean up: remove leading "=" or other prefixes
            line = re.sub(r"^[=:\s]+", "", line)
            return line.strip()

    return None


class CountdownEnv(ProblemEnv):
    """Environment for Countdown number game tasks.

    Supports two reward modes:
    - ``"binary"``: 1.0 for correct, 0.0 for incorrect (default ProblemEnv behavior).
    - ``"partial"``: Gives intermediate rewards for valid expressions that use
      correct numbers but evaluate to the wrong result, with a proximity bonus
      for getting close to the target. This converts some "all-bad" groups into
      "mixed" groups, increasing the fraction of useful GRPO gradients.
    """

    def __init__(
        self,
        target: int,
        nums: list[int],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        reward_mode: Literal["binary", "partial"] = "partial",
    ):
        super().__init__(renderer, convo_prefix)
        self.target = target
        self.nums = nums
        self.reward_mode = reward_mode

    def get_question(self) -> str:
        nums_str = ", ".join(str(n) for n in self.nums)
        return (
            f"Using the numbers [{nums_str}], create an arithmetic expression that equals {self.target}. "
            f"You can use +, -, *, / and each number at most once. "
            f"Put your final expression in \\boxed{{}}."
        )

    def check_answer(self, sample_str: str) -> bool:
        expr = extract_answer(sample_str)
        if expr is None:
            return False
        is_correct, _ = evaluate_countdown_expression(expr, self.nums, self.target)
        return is_correct

    def get_partial_reward(self, sample_str: str) -> float:
        """Return a partial reward in [0, 1] for the model's response.

        Used when ``reward_mode="partial"`` to provide graded feedback.
        """
        expr = extract_answer(sample_str)
        if expr is None:
            return 0.0
        _, partial_score = evaluate_countdown_expression(expr, self.nums, self.target)
        return partial_score

    def check_format(self, sample_str: str) -> bool:
        return extract_answer(sample_str) is not None

    def get_reference_answer(self) -> str:
        return f"target={self.target}, nums={self.nums}"

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Override step to support partial reward mode."""
        if self.reward_mode == "binary":
            return await super().step(action, extra=extra)

        # Partial reward mode: replace binary correct with graded reward
        convo = self.convo_prefix + [{"role": "user", "content": self.get_question()}]
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        correct_format = float(parse_success) and float(self.check_format(content))
        correct_answer = float(self.check_answer(content))
        partial_reward = self.get_partial_reward(content)

        # Use partial reward instead of binary correct
        reward_value = partial_reward if not correct_answer else 1.0
        total_reward = self.format_coef * (correct_format - 1) + reward_value

        # Log the attempt
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=convo))
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "reference_answer": self.get_reference_answer(),
                    "format_valid": bool(correct_format),
                    "correct": bool(correct_answer),
                    "partial_reward": f"{partial_reward:.3f}",
                    "total_reward": f"{total_reward:.3f}",
                },
                caption="Reward components",
            )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
                "partial_reward": partial_reward,
            },
        )

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": (
                    "Using the numbers [3, 7, 2], create an arithmetic expression that equals 13. "
                    "You can use +, -, *, / and each number at most once. "
                    "Put your final expression in \\boxed{}."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "3 * 2 = 6, 6 + 7 = 13. Yes!\n"
                    "\\boxed{3 * 2 + 7}"
                ),
            },
        ]


class CountdownDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        data: list[dict],
        convo_prefix: list[renderers.Message] | None = None,
        reward_mode: Literal["binary", "partial"] = "partial",
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.data = data
        self.convo_prefix = convo_prefix
        self.reward_mode = reward_mode

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.data))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    CountdownEnv,
                    row["target"],
                    row["nums"],
                    self.renderer,
                    convo_prefix=self.convo_prefix,
                    reward_mode=self.reward_mode,
                ),
                num_envs=self.group_size,
                dataset_name="countdown",
            )
            for row in self.data[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)


@chz.chz
class CountdownDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    n_train: int = 10000
    n_test: int = 500
    seed: int = 0
    include_fewshot: bool = True
    reward_mode: Literal["binary", "partial"] = "partial"

    async def __call__(self) -> tuple[CountdownDataset, CountdownDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        convo_prefix = CountdownEnv.standard_fewshot_prefix() if self.include_fewshot else None

        ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        ds = ds.shuffle(seed=self.seed)

        # Split into train and test
        train_data = [{"target": row["target"], "nums": row["nums"]} for row in ds.select(range(self.n_train))]
        test_data = [{"target": row["target"], "nums": row["nums"]} for row in ds.select(range(self.n_train, self.n_train + self.n_test))]

        train_dataset = CountdownDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            data=train_data,
            convo_prefix=convo_prefix,
            reward_mode=self.reward_mode,
        )
        # Test always uses binary reward for clean accuracy measurement
        test_dataset = CountdownDataset(
            batch_size=self.batch_size,
            group_size=1,
            renderer=renderer,
            data=test_data,
            convo_prefix=convo_prefix,
            reward_mode="binary",
        )
        return train_dataset, test_dataset
