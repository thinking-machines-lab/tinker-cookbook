import math
from functools import partial

import chz
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.rl.math_grading import extract_boxed, grade_answer, run_with_timeout_signal
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class MathEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem = problem
        self.answer = answer

    def get_question(self) -> str:
        return self.problem

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return safe_grade(answer, self.answer)

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {"role": "user", "content": "How many r's are in strawberry?"},
            {
                "role": "assistant",
                "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
            },
        ]


def safe_grade(given_answer: str, ground_truth: str, timeout: float = 1.0):
    out = run_with_timeout_signal(
        grade_answer, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out


def _get_hendrycks_math_all():
    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset(dataset_name, name=cfg, split=split)
            pieces.append(ds)
    full_dataset = concatenate_datasets(pieces)
    return full_dataset


class MathDataset(RLDataset):
    def __init__(self, batch_size: int, group_size: int, renderer: renderers.Renderer):
        self.ds = _get_hendrycks_math_all().shuffle(seed=0)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        return [
            builder
            for row in self.ds.select(range(index * self.batch_size, (index + 1) * self.batch_size))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None
        ]

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size

    def _make_env_group_builder(self, x: dict, group_size: int) -> ProblemGroupBuilder | None:
        try:
            answer = extract_boxed(x["solution"])
        except ValueError:  # not sure if this happens
            logger.warning(f"No answer found for {x['solution']}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(MathEnv, x["problem"], answer, self.renderer),
            num_envs=group_size,
        )


@chz.chz
class MathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int

    def __call__(self) -> tuple[MathDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return MathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
        ), None
