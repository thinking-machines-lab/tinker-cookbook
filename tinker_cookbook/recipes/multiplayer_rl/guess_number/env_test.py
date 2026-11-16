import json
import unittest
from collections.abc import Sequence
from typing import cast

from tinker import ModelInput

from tinker_cookbook.recipes.multiplayer_rl.guess_number.env import GuessNumberEnv
from tinker_cookbook.renderers import Message, ParseTermination, Renderer


class StrictRenderer:
    def __init__(self, message: Message):
        self.message = message

    def get_stop_sequences(self) -> list[int]:
        return [0]

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        return self.message, ParseTermination.STOP_SEQUENCE

    def build_generation_prompt(self, messages: Sequence[Message]) -> ModelInput:
        for message in messages:
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role: {message['role']}")
        return ModelInput.from_ints(list(json.dumps(list(messages)).encode()))


class GuessNumberEnvTest(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_turns_keep_the_parsed_assistant_message(self) -> None:
        first: Message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Guess: 3"}],
        }
        renderer = StrictRenderer(first)
        env = GuessNumberEnv(gold_answer=5, renderer=cast(Renderer, renderer))
        await env.initial_observation()

        result = await env.step([1])
        self.assertFalse(result.episode_done)
        self.assertEqual(result.reward, 0)
        self.assertEqual(env.turns, [first, {"role": "user", "content": "Too low"}])

        second: Message = {"role": "assistant", "content": "Guess: 5"}
        renderer.message = second
        result = await env.step([2])
        self.assertTrue(result.episode_done)
        self.assertEqual(result.reward, 1)
        self.assertEqual(
            env.turns,
            [
                first,
                {"role": "user", "content": "Too low"},
                second,
                {"role": "user", "content": "Correct"},
            ],
        )
        self.assertEqual(
            json.loads(bytes(result.next_observation.to_ints())),
            [env.system_prompt, *env.turns],
        )

    async def test_invalid_guess_still_renders_the_next_observation(self) -> None:
        message: Message = {"role": "assistant", "content": "Not a guess"}
        renderer = StrictRenderer(message)
        env = GuessNumberEnv(gold_answer=5, renderer=cast(Renderer, renderer))
        result = await env.step([1])
        self.assertFalse(result.episode_done)
        self.assertEqual(result.reward, -1)
        self.assertEqual(env.turns[0], message)
        self.assertEqual(env.turns[1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
