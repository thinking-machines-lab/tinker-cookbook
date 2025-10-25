from functools import partial
from typing import Sequence

import chz
import numpy as np
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class ArithmeticEnv(ProblemEnv):
    """
    A toy environment for solving addition problems.
    """

    def __init__(
        self,
        x: int,
        y: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix)
        self.x = x
        self.y = y

    def get_question(self) -> str:
        return f"What is {self.x} + {self.y}?"

    def check_answer(self, sample_str: str) -> bool:
        chunks = sample_str.split()
        try:
            answer = int(chunks[0])
        except (ValueError, IndexError):
            return False
        return answer == self.x + self.y

    def check_format(self, sample_str: str) -> bool:
        return True

    def get_reference_answer(self) -> str:
        return str(self.x + self.y)

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {"role": "user", "content": "What is 4 + 5?"},
            {"role": "assistant", "content": "9"},
        ]


class ArithmeticDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        renderer: renderers.Renderer,
        group_size: int,
        n_batches: int = 100,
        include_fewshot: bool = True,
    ):
        self._rng = np.random.RandomState(None)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.n_batches = n_batches
        self.include_fewshot = include_fewshot

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        self._rng.seed(index)
        return [self._make_env_group_builder(self._rng) for _ in range(self.batch_size)]

    def _make_env_group_builder(self, rng: np.random.RandomState) -> ProblemGroupBuilder:
        x = rng.randint(0, 101)
        y = rng.randint(0, 101)
        convo_prefix = ArithmeticEnv.standard_fewshot_prefix() if self.include_fewshot else None
        return ProblemGroupBuilder(
            env_thunk=partial(
                ArithmeticEnv, x, y, convo_prefix=convo_prefix, renderer=self.renderer
            ),
            num_envs=self.group_size,
        )

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class ArithmeticDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    n_batches: int
    group_size: int
    include_fewshot: bool = True

    async def __call__(self) -> tuple[ArithmeticDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return ArithmeticDataset(
            batch_size=self.batch_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            n_batches=self.n_batches,
            include_fewshot=self.include_fewshot,
            group_size=self.group_size,
        ), None
