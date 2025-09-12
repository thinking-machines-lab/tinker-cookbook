import functools
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chz
import tinker
from tinker.types import ModelInput
from tinker_cookbook import model_info
from tinker_cookbook.completers import (
    MessageCompleter,
    StopCondition,
    TinkerMessageCompleter,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

# A MessageCompleter is similar to the OpenAI chat completions API


class TwentyQuestionsEnv(Env):
    def __init__(self, answerer: MessageCompleter, answer: str, renderer: Renderer):
        self.answerer = answerer
        self.answer = answer
        self.sys_for_answerer: Message = {
            "role": "system",
            "content": f"You are are the answerer in a game of 20 questions. You should only ever respond with 'yes' or 'no'. Your secret word is {answer}. If the other player guesses it with Guess: <answer>, respond with 'yes' only if the answer is precisely your secret word.",
        }
        self.sys_for_player: Message = {
            "role": "system",
            "content": "You are the player in a game of 20 questions. You will ask a series of yes/no questions to the answerer. You will win if you can guess the answer in 20 questions or less. You will lose if you ask more than 20 questions. To guess the answer, write a line of the form 'Guess: <answer>' (without the angle brackets). The answer is always a single word -- don't use articles like 'a' or 'the'. Your questions should be one line, and less than 20 words.",
        }
        self.renderer = renderer
        self.turns = []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def convo_for_player(self) -> list[Message]:
        """Conversation from the player's perspective."""
        game_role_to_chat_role = {"answerer": "user", "player": "assistant"}
        return [self.sys_for_player] + [
            {"role": game_role_to_chat_role[turn["role"]], "content": turn["content"]}
            for turn in self.turns
        ]

    def convo_for_answerer(self) -> list[Message]:
        """Conversation from the answerer's perspective."""
        game_role_to_chat_role = {"answerer": "assistant", "player": "user"}
        return [self.sys_for_answerer] + [
            {"role": game_role_to_chat_role[turn["role"]], "content": turn["content"]}
            for turn in self.turns[-1:]
        ]

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self.renderer.build_generation_prompt(self.convo_for_player()), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        (action_message, _parse_success) = self.renderer.parse_response(action)
        self.turns.append({"role": "player", "content": action_message["content"]})
        answer_message = await self.answerer(self.convo_for_answerer())
        self.turns.append({"role": "answerer", "content": answer_message["content"]})
        match = re.match(r"Guess: (.*)", action_message["content"])
        maybe_answer = match.group(1) if match else None
        if (maybe_answer is not None) and (maybe_answer.lower() == self.answer.lower()):
            reward = 1
        else:
            reward = 0
        return StepResult(
            next_observation=self.renderer.build_generation_prompt(self.convo_for_player()),
            next_stop_condition=self.stop_condition,
            episode_done=(reward == 1) or (len(self.turns) // 2 >= 20),
            reward=reward,
        )


# The EnvGroupBuilder is trivial: just return a list of copies of the same environment.


@functools.cache
def get_words() -> list[str]:
    module_dir = Path(__file__).parent
    file_path = module_dir / "common_english_nouns.txt"

    rng = random.Random(0)
    with open(file_path, "r") as f:
        words = [line.strip() for line in f.readlines()]
    rng.shuffle(words)
    return words


@dataclass(frozen=True)
class TwentyQuestionsEnvGroupBuilder(EnvGroupBuilder):
    answerer: MessageCompleter
    answer: str
    renderer: Renderer
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            TwentyQuestionsEnv(self.answerer, self.answer, self.renderer)
            for _ in range(self.num_envs)
        ]


# The dataset just indexes into the list of possible answers.


@dataclass(frozen=True)
class TwentyQuestionsDataset(RLDataset):
    answerer: MessageCompleter
    answers: list[str]
    renderer: Renderer
    batch_size: int
    group_size: int

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        return [
            TwentyQuestionsEnvGroupBuilder(
                answerer=self.answerer,
                answer=self.answers[index * self.batch_size + i],
                renderer=self.renderer,
                num_envs=self.group_size,
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return len(self.answers) // self.batch_size


@chz.chz
class TwentyQuestionsDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    base_url: str | None = None
    num_epochs: int = 1
    answerer_base_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    def __call__(self) -> tuple[RLDataset, RLDataset]:
        if self.answerer_base_model.startswith("Qwen/Qwen3"):
            renderer_name = "qwen3_nothink"
        else:
            renderer_name = model_info.get_recommended_renderer_name(self.answerer_base_model)
        answerer_tokenizer = get_tokenizer(self.answerer_base_model)
        answerer_renderer = get_renderer(renderer_name, answerer_tokenizer)
        service_client = tinker.ServiceClient(base_url=self.base_url)
        answerer_sampling_client = service_client.create_sampling_client(
            base_model=self.answerer_base_model
        )
        answerer = TinkerMessageCompleter(
            sampling_client=answerer_sampling_client, renderer=answerer_renderer, max_tokens=5
        )
        words = get_words()
        num_test = min(len(words) // 5, 100)
        train_words = words[:-num_test]
        test_words = words[-num_test:]
        train_words = train_words * self.num_epochs
        player_renderer = get_renderer(
            self.renderer_name, get_tokenizer(self.model_name_for_tokenizer)
        )
        assert self.batch_size <= len(train_words)
        training_dataset = TwentyQuestionsDataset(
            answerer=answerer,
            answers=train_words,
            renderer=player_renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
        )
        test_dataset = TwentyQuestionsDataset(
            answerer=answerer,
            answers=test_words,
            renderer=player_renderer,
            batch_size=len(test_words),
            group_size=self.group_size,
        )
        return training_dataset, test_dataset
