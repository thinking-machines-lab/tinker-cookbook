"""Script for interactively playing single-player RL environments."""

import asyncio

import chz
import tinker
from termcolor import colored
from tinker import types
from tinker_cookbook import model_info
from tinker_cookbook.completers import (
    StopCondition,
    TinkerTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)
from tinker_cookbook.rl import train_cli
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import Env, Trajectory
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer


async def get_async_input(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


class ManualPolicy(TokenCompleter):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.step_count = 0

    async def __call__(self, ob: types.ModelInput, stop: StopCondition) -> TokensWithLogprobs:
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
    print("Type your actions when prompted. The game will end when the episode is done.")

    policy = ManualPolicy(tokenizer)
    trajectory = await do_single_rollout(policy, env)

    print_trajectory_summary(trajectory)
    return trajectory


@chz.chz
class Config:
    env: str
    model: str | None = None  # e.g., "meta-llama/Llama-3.1-8B-Instruct"


async def main(cfg: Config):
    if cfg.model is None:
        renderer_name = "role_colon"
    else:
        renderer_name = model_info.get_recommended_renderer_name(cfg.model)
    model_for_tokenizer = cfg.model or "meta-llama/Llama-3.1-8B-Instruct"
    dataset_builder = train_cli.get_dataset_builder(
        cfg.env,
        batch_size=1,
        group_size=1,
        model_name=model_for_tokenizer,
        renderer_name=renderer_name,
    )
    dataset, _ = dataset_builder()
    env_group_builder = dataset.get_batch(0)[0]
    envs = await env_group_builder.make_envs()
    env = envs[0]

    tokenizer = get_tokenizer(model_for_tokenizer)

    if cfg.model is None:
        # Interactive play
        await play_env(env, tokenizer)
    else:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=cfg.model)
        policy = TinkerTokenCompleter(sampling_client=sampling_client, max_tokens=20)

        print(colored(f"\nWatching {cfg.model} play...", "cyan", attrs=["bold"]))
        trajectory = await do_single_rollout(policy, env)

        # Print the game transcript
        for i, transition in enumerate(trajectory.transitions):
            print(colored(f"\n--- Step {i} ---", "green"))
            print(colored("Observation:", "blue"))
            print(tokenizer.decode(transition.ob.to_ints()))
            print(colored("Action:", "yellow"))
            print(tokenizer.decode(transition.ac.tokens))

        print_trajectory_summary(trajectory)


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
