"""

"""

import asyncio
import tinker
from termcolor import colored
from tinker_cookbook.completers import (
    StopCondition,
    TokenCompleter,
    TokensWithLogprobs,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import Env, Trajectory
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name


async def get_async_input(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


class ManualPolicy(TokenCompleter):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.step_count = 0

    async def __call__(self, ob: tinker.ModelInput, stop: StopCondition) -> TokensWithLogprobs:
        observation_str = self.tokenizer.decode(ob.to_ints())
        print(colored(f"\n--- Step {self.step_count} ---", "green"))
        print(colored("Observation:", "blue"))
        print(observation_str)
        print(colored("-" * 60, "green"))

        action_str = await get_async_input(colored("Your action: ", "yellow"))
        action_tokens = self.tokenizer.encode(action_str, add_special_tokens=False)
        self.step_count += 1
        return TokensWithLogprobs(tokens=action_tokens, maybe_logprobs=None)

    

def print_trajectory_summary(trajectory: Trajectory):
    """Print a summary of the completed trajectory."""
    print(colored("\n=== Game Summary ===", "cyan", attrs=["bold"]))
    total_reward = sum(t.reward for t in trajectory.transitions)
    print(f"Total steps: {len(trajectory.transitions)}")
    print(f"Total reward: {total_reward}")

    if trajectory.transitions:
        print("\nReward per step:")
        for i, transition in enumerate(trajectory.transitions):
            if transition.reward != 0:
                print(f"  Step {i}: reward = {transition.reward}")

    print(colored("===================", "cyan", attrs=["bold"]))


async def play_env(env: Env, tokenizer: Tokenizer):
    """Play a single-player environment interactively."""
    print(colored("Starting interactive environment session...", "cyan", attrs=["bold"]))
    print("Type your actions when prompted. The episode will end when the episode is done.")

    policy = ManualPolicy(tokenizer)
    trajectory = await do_single_rollout(policy, env)

    print_trajectory_summary(trajectory)
    return trajectory





async def main():
    from tinker_cookbook.recipes.multiplayer_rl.twenty_questions.env import construct_minimal_20q_env
    answer = "apple"
    env = construct_minimal_20q_env(answer)
    await play_env(env, env.renderer.tokenizer)


if __name__ == "__main__":
    asyncio.run(main())