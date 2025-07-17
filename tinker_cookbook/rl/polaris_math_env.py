from functools import partial

import chz
from datasets import load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.rl.math_env import MathDataset, MathEnv
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


def parse_difficulty(difficulty: str) -> int:
    """Parse difficulty string like '3/8' into an integer 3."""
    if not difficulty or "/" not in difficulty:
        return 0
    numerator = difficulty.split("/")[0]
    try:
        return int(numerator)
    except ValueError:
        return 0


class PolarisDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        easiness_filter: int | None = None,
    ):
        self.ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(seed=0)

        if easiness_filter is not None:
            self.ds = self.ds.filter(
                lambda x: parse_difficulty(x.get("difficulty", "")) >= easiness_filter
            )

        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer

    def _make_env_group_builder(self, x: dict, group_size: int) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(MathEnv, problem, answer, self.renderer),
            num_envs=group_size,
        )


@chz.chz
class PolarisDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    difficulty_filter: int | None = None

    def __call__(self) -> tuple[PolarisDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return PolarisDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            easiness_filter=self.difficulty_filter,
        ), None
