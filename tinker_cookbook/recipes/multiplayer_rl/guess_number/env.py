import random
import re
from dataclasses import dataclass
from typing import Sequence

import chz
from tinker import ModelInput
from tinker_cookbook.completers import (
    StopCondition,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer, ensure_text
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

_UPPER_BOUND = 1024
SYSTEM_PROMPT = f"""
Your job is to guess a integer between 0 and {_UPPER_BOUND}. You can output your guess with the format Guess: <guess> (without the angle brackets), and say nothing else. Then we will tell you whether you guess is too high, too low, or correct.""".strip()
FAIL_TO_PARSE = "Failed to parse. Please output in the format Guess: <guess> (without the angle brackets), and say nothing else."
FORMAT_PENALTY = -1.0
MAX_TURNS = 10
RETURN_ON_FAIL = (Message(role="user", content=FAIL_TO_PARSE), FORMAT_PENALTY)


class GuessNumberEnv(Env):
    def __init__(self, gold_answer: int, renderer: Renderer):
        self.system_prompt: Message = {"role": "system", "content": SYSTEM_PROMPT}
        self.renderer: Renderer = renderer
        self.turns: list[Message] = []
        self.gold_answer: int = gold_answer

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @property
    def _obs(self) -> ModelInput:
        """Get the observation for the player in tokenized form"""
        convo_for_player = [self.system_prompt] + self.turns
        return self.renderer.build_generation_prompt(convo_for_player)

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self._obs, self.stop_condition

    def _get_user_turn(self, action_text: str) -> tuple[Message, float]:
        # parse the answer from the action text (i.e. LLM's guess)
        match = re.match(r"Guess: (.*)", action_text)
        maybe_answer = match.group(1) if match else None
        try:
            if maybe_answer is not None:
                int_answer = int(maybe_answer)
                if int_answer == self.gold_answer:
                    text, reward = "Correct", 1.0
                elif int_answer < self.gold_answer:
                    text, reward = "Too low", 0.0
                else:
                    text, reward = "Too high", 0.0
                return Message(role="user", content=text), reward
            else:
                return RETURN_ON_FAIL
        except ValueError:
            return RETURN_ON_FAIL

    async def step(self, action: Action) -> StepResult:
        # step 1: parse the action tokens into a message
        # this step is specific to our library, but usually templated, so you can just copy it.
        (action_message, _parse_success) = self.renderer.parse_response(action)

        # step 2: based on the string answer, we compute the reward and the user turn.
        # This part is NOT templated, so you need to implement it. But it is plain python without using special libraries.
        action_content = ensure_text(action_message["content"])
        user_turn, reward = self._get_user_turn(action_content)

        # step 3: update the conversation history
        self.turns.append({"role": "player", "content": action_content})
        self.turns.append(user_turn)
        episode_done = (reward == 1) or (len(self.turns) // 2 >= MAX_TURNS)

        # step 4: return the step result
        step_result = StepResult(
            next_observation=self._obs,
            next_stop_condition=self.stop_condition,
            episode_done=episode_done,
            reward=reward,
        )

        return step_result


@dataclass(frozen=True)
class GuessNumberEnvGroupBuilder(EnvGroupBuilder):
    answer: int
    renderer: Renderer
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [GuessNumberEnv(self.answer, self.renderer) for _ in range(self.num_envs)]


# The dataset just indexes into the list of possible answers.


@dataclass(frozen=True)
class GuessNumberDataset(RLDataset):
    answers: Sequence[int]
    renderer: Renderer
    batch_size: int
    group_size: int

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            GuessNumberEnvGroupBuilder(
                answer=self.answers[index * self.batch_size + i],
                renderer=self.renderer,
                num_envs=self.group_size,
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return len(self.answers) // self.batch_size


@chz.chz
class GuessNumberDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    renderer_name: str
    train_group_size: int
    base_url: str | None = None
    model_name: str
    train_fraction: float = 0.9
    test_group_size: int = 4

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        player_renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        train_numbers, test_numbers = self._get_train_and_test_numbers()
        assert self.batch_size <= len(train_numbers)
        training_dataset = GuessNumberDataset(
            answers=train_numbers,
            renderer=player_renderer,
            batch_size=self.batch_size,
            group_size=self.train_group_size,
        )
        test_dataset = GuessNumberDataset(
            answers=test_numbers,
            renderer=player_renderer,
            batch_size=len(test_numbers),  # test set only contains one batch
            group_size=self.test_group_size,
        )
        return training_dataset, test_dataset

    def _get_train_and_test_numbers(self) -> tuple[list[int], list[int]]:
        rng = random.Random(0)
        num_train_datapoints = int(_UPPER_BOUND * self.train_fraction)
        shuffled_numbers = rng.sample(range(0, _UPPER_BOUND), _UPPER_BOUND)
        train_numbers = shuffled_numbers[:num_train_datapoints]
        test_numbers = shuffled_numbers[num_train_datapoints:]
        return train_numbers, test_numbers
