"""
To help you debug your environment, you can use the play_env function to play as the policy by typing in your responses in an environment interactively.

Options:
- multiline=True: Enable multi-line input mode (terminate with two blank lines)

We include an example of playing the Twenty Questions environment in the main function.
You can run it with:

```bash
python -m tinker_cookbook.rl.play_w_env
```
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


async def get_async_input(prompt: str, multiline: bool = False) -> str:
    loop = asyncio.get_event_loop()
    if not multiline:
        return await loop.run_in_executor(None, input, prompt)

    # Multiline mode: collect lines until two consecutive blank lines
    print(prompt + " (enter two blank lines when done)")
    lines = []
    prev_line = None
    while True:
        line = await loop.run_in_executor(None, input, "")
        if line == "" and prev_line == "":
            # Remove the first blank line from the list
            if lines and lines[-1] == "":
                lines.pop()
            break
        lines.append(line)
        prev_line = line
    return "\n".join(lines)


class ManualPolicy(TokenCompleter):
    def __init__(self, tokenizer: Tokenizer, multiline: bool = True, show_observation: bool = True):
        self.tokenizer = tokenizer
        self.step_count = 0
        self.multiline = multiline
        self.show_observation = show_observation

    async def __call__(self, ob: tinker.ModelInput, stop: StopCondition) -> TokensWithLogprobs:
        if self.show_observation:
            observation_str = self.tokenizer.decode(ob.to_ints())
            print(colored(f"\n--- Step {self.step_count} ---", "green"))
            print(colored("Observation:", "blue"))
            print(observation_str)
            print(colored("-" * 60, "green"))

        prompt_text = "Your action:" if self.multiline else "Your action: "
        action_str = await get_async_input(colored(prompt_text, "yellow"), multiline=self.multiline)
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


async def play_env(
    env: Env, tokenizer: Tokenizer, multiline: bool = True, show_observation: bool = True
):
    """Play a single-player environment interactively."""
    print(colored("Starting interactive environment session...", "cyan", attrs=["bold"]))
    print("Type your actions when prompted. The episode will end when the episode is done.")

    policy = ManualPolicy(tokenizer, multiline=multiline, show_observation=show_observation)
    trajectory = await do_single_rollout(policy, env)

    print_trajectory_summary(trajectory)
    return trajectory


async def main():
    from tinker_cookbook.recipes.multiplayer_rl.twenty_questions.env import (
        construct_minimal_20q_env,
    )

    answer = "apple"
    env = construct_minimal_20q_env(answer)
    await play_env(env, env.renderer.tokenizer)


if __name__ == "__main__":
    asyncio.run(main())
