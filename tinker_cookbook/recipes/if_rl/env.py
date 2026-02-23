import enum
import random
from functools import partial
from typing import Sequence, cast

import chz
import tinker
from if_verifiable import IFBenchSample, evaluate_output_for_sample, get_eval_data

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


class RewardType(enum.Enum):
    FULL_STRICT = "full_strict"
    FULL_LOOSE = "full_loose"
    PARTIAL_STRICT = "partial_strict"
    PARTIAL_LOOSE = "partial_loose"


class IfBenchEnv(ProblemEnv):
    """Environment for IFBench (ai2) instruction-following tasks."""

    def __init__(
        self,
        renderer: renderers.Renderer,
        sample: IFBenchSample,
        convo_prefix: list[renderers.Message] | None = None,
        reward_type: RewardType = RewardType.PARTIAL_LOOSE,
    ):
        super().__init__(renderer, convo_prefix)
        self.sample = sample
        self.reward_type = reward_type

    def get_question(self) -> str:
        return self.sample.prompt

    def check_answer(self, sample_str: str) -> bool:
        _, scores = evaluate_output_for_sample("ifbench", self.sample, sample_str)
        if self.reward_type in (RewardType.FULL_LOOSE, RewardType.PARTIAL_LOOSE):
            return bool(scores.binary_loose)
        return bool(scores.binary_strict)

    def check_format(self, sample_str: str) -> bool:
        return True

    def get_reference_answer(self) -> str:
        return f"Instructions: {', '.join(self.sample.instruction_id_list)}"

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.convo_prefix + [{"role": "user", "content": self.get_question()}]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, _ = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        results, scores = evaluate_output_for_sample("ifbench", self.sample, content)

        reward = {
            RewardType.FULL_STRICT: float(scores.binary_strict),
            RewardType.FULL_LOOSE: float(scores.binary_loose),
            RewardType.PARTIAL_STRICT: scores.partial_strict,
            RewardType.PARTIAL_LOOSE: scores.partial_loose,
        }[self.reward_type]

        use_loose = self.reward_type in (RewardType.FULL_LOOSE, RewardType.PARTIAL_LOOSE)
        logtree.log_text(f"Prompt: {self.sample.prompt[:200]}...")
        logtree.log_text(f"Response: {content[:500]}...")
        logtree.log_text(f"Instructions: {self.sample.instruction_id_list}")
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
                "strict": scores.partial_strict,
                "loose": scores.partial_loose,
                "all_strict": float(scores.binary_strict),
                "all_loose": float(scores.binary_loose),
                "num_instructions": len(results),
                **{
                    f"instr/{r.instruction_id}": float(
                        r.loose_pass if use_loose else r.strict_pass
                    )
                    for r in results
                },
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
        self.data: list[IFBenchSample] = (
            list(cast(Sequence[IFBenchSample], get_eval_data("ifbench"))) * num_epochs
        )
        random.Random(seed).shuffle(self.data)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_data = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        return [self._make_env_group_builder(sample) for sample in batch_data]

    def __len__(self) -> int:
        return len(self.data) // self.batch_size

    def _make_env_group_builder(self, sample: IFBenchSample) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                IfBenchEnv,
                renderer=self.renderer,
                sample=sample,
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
        return IfBenchDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=None,
            reward_type=self.reward_type,
            seed=self.seed,
            num_epochs=self.num_epochs,
        ), None
