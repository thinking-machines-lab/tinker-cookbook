import functools
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chz
import tinker
from tinker import ModelInput
from tinker_cookbook import model_info
from tinker_cookbook.completers import (
    MessageCompleter,
    StopCondition,
    TinkerMessageCompleter,
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
from tinker_cookbook.utils import logtree
from tinker_cookbook.model_info import get_recommended_renderer_name

ANSWERER_SYSTEM_PROMPT = """
You are the answerer in a game of 20 questions. You should only ever respond with 'yes' or 'no'. Your secret word is {answer}. If the other player guesses it with Guess: <answer>, respond with 'yes' only if the answer is precisely your secret word.
""".strip()

PLAYER_SYSTEM_PROMPT = """
You are the player in a game of 20 questions. You will ask a series of yes/no questions to the answerer. You will win if you can guess the answer in 20 questions or less. You will lose if you ask more than 20 questions. To guess the answer, write a line of the form 'Guess: <answer>' (without the angle brackets). The answer is always a single word -- don't use articles like 'a' or 'the'. Your questions should be one line, and less than 20 words.
""".strip()


class TwentyQuestionsEnv(Env):
    def __init__(self, answerer: MessageCompleter, answer: str, renderer: Renderer):
        self.answerer: MessageCompleter = answerer
        self.answer: str = answer
        self.sys_for_answerer: Message = {
            "role": "system",
            "content": ANSWERER_SYSTEM_PROMPT.format(answer=answer),
        }
        self.sys_for_player: Message = {
            "role": "system",
            "content": PLAYER_SYSTEM_PROMPT,
        }
        self.renderer: Renderer = renderer
        self.turns: list[Message] = []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _convo_for_player(self) -> list[Message]:
        """Conversation from the player's perspective."""
        game_role_to_chat_role = {"answerer": "user", "player": "assistant"}
        return [self.sys_for_player] + [
            {"role": game_role_to_chat_role[turn["role"]], "content": turn["content"]}
            for turn in self.turns
        ]

    def _get_obs(self) -> ModelInput:
        """Get the observation for the player in tokenized form"""
        return self.renderer.build_generation_prompt(self._convo_for_player())

    def _convo_for_answerer(self) -> list[Message]:
        """Conversation from the answerer's perspective."""
        game_role_to_chat_role = {"answerer": "assistant", "player": "user"}
        return (
            [self.sys_for_answerer]
            + [
                {"role": game_role_to_chat_role[turn["role"]], "content": turn["content"]}
                for turn in self.turns[
                    -1:
                ]  # show the answerer only the last turn because the response to each player turn should be independent of the previous turns
            ]
        )

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self._get_obs(), self.stop_condition

    def _compute_reward(self, content: str) -> float:
        """
        Returns 1.0 if the content contains the answer, 0.0 otherwise.
        """
        match = re.match(r"Guess: (.*)", content)
        maybe_answer = match.group(1) if match else None
        content_contains_answer = (maybe_answer is not None) and (
            maybe_answer.lower() == self.answer.lower()
        )
        return 1.0 if content_contains_answer else 0.0

    async def step(self, action: Action) -> StepResult:
        """
        In one step,
        1. The environment accepts an action from the player (a message containin a question or a guess).
        2. We obtain the response from the answerer and update the conversation history in self.turns.
        3. We calculate the reward and decide whether to end the episode.
        4. We return these information, along with the next observation built from the updated conversation history.
        """

        # step 1: accepts the action from the player (policy)
        (action_message, _parse_success) = self.renderer.parse_response(action)
        self.turns.append({"role": "player", "content": action_message["content"]})

        # step 2: the answerer responds
        answer_message = await self.answerer(self._convo_for_answerer())
        self.turns.append({"role": "answerer", "content": answer_message["content"]})

        # step 3: we calculate the reward and decide whether to end the episode.
        # the episode ends if the player guessed the answer or the player asked more than 20 questions
        action_content = ensure_text(action_message["content"])
        reward = self._compute_reward(action_content)
        episode_done = (reward == 1) or (len(self.turns) // 2 >= 20)

        # Log the turn
        turn_num = len(self.turns) // 2
        logtree.log_text(f"Turn {turn_num} - Player: {action_message['content']}")
        logtree.log_text(f"Turn {turn_num} - Answerer: {answer_message['content']}")
        if episode_done:
            logtree.log_text(
                f"Game Over - Secret: {self.answer}, Won: {'✓' if reward == 1 else '✗'}, Turns: {turn_num}"
            )

        # step 4: we return the next observation, reward, and whether the episode is done
        step_result = StepResult(
            next_observation=self._get_obs(),
            next_stop_condition=self.stop_condition,
            episode_done=episode_done,
            reward=reward,
        )

        return step_result


# The EnvGroupBuilder is trivial: just return a list of copies of the same environment.


@functools.cache
def _load_words_from_file() -> list[str]:
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
    answers: Sequence[str]
    renderer: Renderer
    batch_size: int
    group_size: int

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
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
    test_group_size: int = 32
    answerer_base_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        service_client = tinker.ServiceClient(base_url=self.base_url)
        answerer = self._construct_answer_completer(service_client)
        train_words, test_words = self._get_train_and_test_words()
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
            batch_size=len(test_words),  # test set only contains one batch
            group_size=self.test_group_size,
        )
        return training_dataset, test_dataset

    def _construct_answer_completer(self, service_client: tinker.ServiceClient) -> MessageCompleter:
        if self.answerer_base_model.startswith("Qwen/Qwen3"):
            answerer_renderer_name = "qwen3_disable_thinking"
        else:
            answerer_renderer_name = model_info.get_recommended_renderer_name(
                self.answerer_base_model
            )
        answerer_tokenizer = get_tokenizer(self.answerer_base_model)
        answerer_renderer = get_renderer(answerer_renderer_name, answerer_tokenizer)
        answerer_sampling_client = service_client.create_sampling_client(
            base_model=self.answerer_base_model
        )
        answerer = TinkerMessageCompleter(
            sampling_client=answerer_sampling_client, renderer=answerer_renderer, max_tokens=5
        )
        return answerer

    def _get_train_and_test_words(self) -> tuple[list[str], list[str]]:
        words = _load_words_from_file()
        num_test = min(len(words) // 5, 100)
        train_words = words[:-num_test]
        test_words = words[-num_test:]
        train_words = train_words * self.num_epochs
        return train_words, test_words


def construct_minimal_20q_env(answer: str) -> TwentyQuestionsEnv:
    answerer_model = "meta-llama/Llama-3.1-8B-Instruct"

    service_client = tinker.ServiceClient()
    answerer_sampling_client = service_client.create_sampling_client(base_model=answerer_model)
    answerer = TinkerMessageCompleter(
        sampling_client=answerer_sampling_client,
        renderer=get_renderer(
            get_recommended_renderer_name(answerer_model), get_tokenizer(answerer_model)
        ),
        max_tokens=5,
    )
    policy_renderer = get_renderer(
        get_recommended_renderer_name(answerer_model), get_tokenizer(answerer_model)
    )  # this argument is not actually used and is a placeholder
    env = TwentyQuestionsEnv(answerer, answer, policy_renderer)
    return env
