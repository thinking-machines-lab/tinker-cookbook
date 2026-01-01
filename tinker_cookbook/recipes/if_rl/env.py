import random
from functools import partial
from typing import Any, Sequence

import chz
import tinker
from datasets import load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
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
from tinker_cookbook.utils import logtree
from tinker_cookbook.recipes.if_rl.ifbench.evaluate import (
    RewardType,
    evaluate_instructions,
    strip_thinking,
)


class IfBenchEnv(ProblemEnv):
    """Environment for IFBench instruction-following tasks.

    Each episode:
    1. Present the prompt (which contains embedded instructions)
    2. Model generates a response
    3. Evaluate response against all instruction constraints
    4. Reward based on fraction of instructions satisfied

    Reward types:
    - FULL_STRICT: 1.0 if ALL instructions pass strict eval, else 0.0
    - FULL_LOOSE: 1.0 if ALL instructions pass loose eval, else 0.0
    - PARTIAL_STRICT: fraction of instructions passing strict eval
    - PARTIAL_LOOSE: fraction of instructions passing loose eval
    """

    def __init__(
        self,
        renderer: renderers.Renderer,
        prompt: str,
        instruction_id_list: list[str],
        kwargs_list: list[dict[str, Any]],
        convo_prefix: list[renderers.Message] | None = None,
        reward_type: RewardType = RewardType.PARTIAL_LOOSE,
    ):
        super().__init__(renderer, convo_prefix)
        self.prompt = prompt
        self.instruction_id_list = instruction_id_list
        self.kwargs_list = kwargs_list
        self.reward_type = reward_type

    def get_question(self) -> str:
        return self.prompt

    def check_answer(self, sample_str: str) -> bool:
        """Check if all instructions are satisfied (used by parent class logging)."""
        _, scores = evaluate_instructions(
            sample_str, self.instruction_id_list, self.kwargs_list, self.prompt
        )
        if self.reward_type in (RewardType.FULL_LOOSE, RewardType.PARTIAL_LOOSE):
            return scores["all_loose"] > 0
        return scores["all_strict"] > 0

    def check_format(self, sample_str: str) -> bool:
        """For IFBench, format checking is part of instruction evaluation."""
        return True  # Format is checked via instructions

    def get_reference_answer(self) -> str:
        """Return instruction types for logging."""
        return f"Instructions: {', '.join(self.instruction_id_list)}"

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        content = strip_thinking(content)
        content = content.replace("<|im_end|>", "").strip()

        results, scores = evaluate_instructions(
            content, self.instruction_id_list, self.kwargs_list, self.prompt
        )

        # Compute reward based on reward_type
        reward = {
            RewardType.FULL_STRICT: scores["all_strict"],
            RewardType.FULL_LOOSE: scores["all_loose"],
            RewardType.PARTIAL_STRICT: scores["strict"],
            RewardType.PARTIAL_LOOSE: scores["loose"],
        }[self.reward_type]

        use_loose = self.reward_type in (RewardType.FULL_LOOSE, RewardType.PARTIAL_LOOSE)
        logtree.log_text(f"Prompt: {self.prompt[:200]}...")
        logtree.log_text(f"Response: {content[:500]}...")
        logtree.log_text(f"Instructions: {self.instruction_id_list}")
        for r in results:
            status = "✓" if (r.loose_pass if use_loose else r.strict_pass) else "✗"
            logtree.log_text(f"  {status} {r.instruction_id}")
        logtree.log_text(f"Reward: {reward:.2f}")

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "strict": scores["strict"],
                "loose": scores["loose"],
                "all_strict": scores["all_strict"],
                "all_loose": scores["all_loose"],
                "num_instructions": len(results),
                **{f"instr/{r.instruction_id}": float(r.loose_pass) for r in results},
            },
        )


class IfBenchDataset(RLDataset):
    """Dataset for IFBench instruction-following training."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        reward_type: RewardType = RewardType.PARTIAL_LOOSE,
        seed: int = 42,
        num_epochs: int = 1,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []
        self.reward_type = reward_type
        dataset = load_dataset("allenai/IFBench_test", split="train")
        self.data = list(dataset)

        # Repeat dataset for multiple epochs before shuffling
        self.data = self.data * num_epochs

        rng = random.Random(seed)
        rng.shuffle(self.data)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_data = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        return [self._make_env_group_builder(row) for row in batch_data]

    def __len__(self) -> int:
        return len(self.data) // self.batch_size

    def _make_env_group_builder(self, row) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                IfBenchEnv,
                renderer=self.renderer,
                prompt=row["prompt"],
                instruction_id_list=row["instruction_id_list"],
                kwargs_list=row["kwargs"],
                convo_prefix=self.convo_prefix,
                reward_type=self.reward_type,
            ),
            num_envs=self.group_size,
        )


@chz.chz
class IfBenchDatasetBuilder(RLDatasetBuilder):
    """Builder for IFBench dataset."""

    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    reward_type: RewardType = RewardType.PARTIAL_LOOSE
    seed: int = 42
    max_tokens: int = 2048
    num_epochs: int = 1

    async def __call__(self) -> tuple[IfBenchDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_dataset = IfBenchDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=None,
            reward_type=self.reward_type,
            seed=self.seed,
            num_epochs=self.num_epochs,
        )
        return train_dataset, None
