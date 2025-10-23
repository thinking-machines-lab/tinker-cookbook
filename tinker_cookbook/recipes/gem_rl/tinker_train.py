"""GEM ❤️ Tinker.

A basic RL implementation to train agents on GEM environments using Tinker backends.
"""

import asyncio
import json
import logging
import os
import pprint
import time
from datetime import datetime
from typing import Any, List, Literal

import chz
import numpy as np
import tinker
import torch
import wandb
from termcolor import colored
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

import gem
from gem.wrappers.wrapper_factory import get_wrapper_fns

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3-8B-Base"
    batch_size: int = 128
    learning_rate: float = 4e-5
    lora_rank: int = 32
    max_tokens: int = 2048
    seed: int = 0
    max_steps: int = 200
    save_every: int = -1

    env_id: str = "rg:simple_equations"
    num_env: int = 4  # number of parallel environments
    env_wrappers: str = "concat"  # wrappers are typically used to concat chat history, etc.
    template: Literal["qwen3_general", "qwen3_game", "no"] = "qwen3_general"

    gamma: float = 0.9
    use_rebn: bool = True
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"

    eval_env_id: str = "eval:AIME24"
    eval_max_tokens: int = 8192
    eval_n: int = 32
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    eval_every: int = -1

    wandb_entity: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_dir: str | None = None


# Define a lightweight renderer following tinker's renderer logics
def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_game_no_think_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_general_template(question: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_no_template(observation: str) -> str:
    return observation


TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "qwen3_general": apply_qwen3_general_template,
    "no": apply_no_template,
}


def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    # Avoid gating of Llama 3 models:
    if model_name.startswith("meta-llama/Llama-3"):
        model_name = "baseten/Meta-Llama-3-tokenizer"
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)

    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}
    logger.info(f"Saved checkpoints: {paths}")
    full_dict = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths


def prepare_training_datums(
    transitions: List[dict], advantage_scaling: float = 1.0
) -> List[types.Datum]:
    training_datums = []
    for transition in transitions:
        ob_len_m1 = len(transition["obs_tokens"]) - 1  # -1 due to shifting
        tokens = transition["obs_tokens"] + transition["act_tokens"]

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        all_logprobs = [0.0] * ob_len_m1 + transition["act_logprobs"]
        all_advantages = [0.0] * ob_len_m1 + [transition["return"]] * (
            len(input_tokens) - ob_len_m1
        )
        assert (
            len(input_tokens) == len(target_tokens) == len(all_logprobs) == len(all_advantages)
        ), (
            f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
        )

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                "advantages": TensorData.from_torch(
                    torch.tensor(all_advantages) * advantage_scaling
                ),
            },
        )
        training_datums.append(datum)
    return training_datums


async def collect_episode(
    sampling_client, sampling_params, env: gem.Env, config, tokenizer, reset_idx=None
):
    transitions = []
    kwargs = {"idx": reset_idx} if reset_idx else {}
    obs, _ = env.reset(**kwargs)
    while True:
        # 1) prepare observation
        obs = TEMPLATE_FACTORY[config.template](obs)  # templated string
        obs_tokens = tokenizer.encode(obs, add_special_tokens=False)

        # 2) sample an action from the policy
        try:
            sample_result = await sampling_client.sample_async(
                prompt=types.ModelInput.from_ints(tokens=obs_tokens),
                num_samples=1,
                sampling_params=sampling_params,
            )
        except Exception:
            transitions = []
            break
        sampled_tokens = sample_result.sequences[0].tokens
        sampled_logprobs = sample_result.sequences[0].logprobs
        action = tokenizer.decode(sampled_tokens)
        unfinished = sample_result.sequences[0].stop_reason == "length"

        # 3) step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        obs = next_obs

        # 4) save into buffer
        transitions.append(
            {
                "obs_tokens": obs_tokens,
                "act_tokens": sampled_tokens,
                "act_logprobs": sampled_logprobs,
                "obs_text": tokenizer.decode(obs_tokens),
                "act_text": tokenizer.decode(sampled_tokens),
                "reward": reward,
                "done": done,
                "unfinished": unfinished,
                "info": info,
            }
        )

        if done:
            break
    return transitions


async def main(config: Config):
    # Setup logging
    wandb_name = config.wandb_name or config.model_name.split("/")[-1] + f"_{config.env_id}"
    wandb_name += "_" + datetime.now().strftime("%m%dT%H:%M:%S")
    save_path = os.path.join("./tinker_output", wandb_name)
    os.makedirs(save_path, exist_ok=True)

    wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        config=chz.asdict(config),
        dir=str(config.log_dir) if config.log_dir else None,
        name=wandb_name,
    )

    # Get tokenizer
    tokenizer = get_tokenizer(config.model_name)

    # Setup environment for training
    wrappers = get_wrapper_fns(config.env_wrappers, tokenizer=tokenizer)
    # init one env first, check if it has dataset; if so we avoid load from HF multiple times
    # by directly providing dataset when creating the env. (we can also use the gem.Env.spawn api).
    envs = [gem.make(config.env_id, seed=int(time.time_ns()), use_mp=False)]
    for i in range(config.num_env - 1):
        dataset = envs[0].dataset if hasattr(envs[0], "dataset") else None
        envs.append(
            gem.make(
                config.env_id,
                seed=int(time.time_ns()) * i,
                dataset=dataset,
                use_mp=False,
            )
        )
    for i in range(len(envs)):
        for wrapper in wrappers:
            envs[i] = wrapper(envs[i])

    # Setup environment for in-distribution eval
    eval_envs = [gem.make(config.eval_env_id, seed=int(time.time_ns()), use_mp=False, eval=True)]
    skip_eval = not hasattr(envs[0], "dataset")
    if not skip_eval:
        eval_data_size = len(eval_envs[0].dataset)
        for i in range((config.eval_n * eval_data_size) - 1):
            eval_envs.append(
                gem.make(
                    config.eval_env_id,
                    seed=int(time.time_ns()) * i,
                    dataset=eval_envs[0].dataset,
                    use_mp=False,
                    eval=True,
                )
            )
        for i in range(len(eval_envs)):
            for wrapper in wrappers:
                eval_envs[i] = wrapper(eval_envs[i])

    # Setup agent (tinker training client)
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model_name, rank=config.lora_rank
    )
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
    )
    eval_sampling_params = tinker.types.SamplingParams(
        max_tokens=config.eval_max_tokens,
        temperature=config.eval_temperature,
        top_p=config.eval_top_p,
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # Start agent-environment loop (Algo: https://arxiv.org/pdf/2510.01051#page=15.10):
    for policy_iteration_step in range(config.max_steps):
        print("=" * 10 + f" Step {policy_iteration_step} " + "=" * 10)
        metrics = {"step": policy_iteration_step}

        # create sampler
        sampling_path = (
            training_client.save_weights_for_sampler(name=f"{policy_iteration_step:06d}")
            .result()
            .path
        )
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        # save model
        if (
            config.save_every > 0
            and policy_iteration_step > 0
            and policy_iteration_step % config.save_every == 0
        ):
            await save_checkpoint_async(
                training_client,
                f"{policy_iteration_step:06d}",
                log_path=save_path,
                kind="state",
                loop_state={"policy_iteration_step": policy_iteration_step},
            )

        # eval model
        if config.eval_every > 0 and policy_iteration_step % config.eval_every == 0:
            if skip_eval:
                print("⚠️ Evaluation environment doesn't have .dataset attribute, skipping eval.")
            else:
                print(f"🔎 Start evaluation at step {policy_iteration_step}")
                st = time.time()
                eval_episodes = await asyncio.gather(
                    *[
                        collect_episode(
                            sampling_client,
                            eval_sampling_params,
                            env,
                            config,
                            tokenizer,
                            i,
                        )
                        for env, i in zip(eval_envs, list(range(eval_data_size)) * config.eval_n)
                    ]
                )
                eval_episodes = [x for x in eval_episodes if x != []]
                for drop_key in ["obs_tokens", "act_tokens", "act_logprobs"]:
                    _ = [t.pop(drop_key) for ep in eval_episodes for t in ep]
                json.dump(
                    eval_episodes,
                    open(
                        os.path.join(save_path, f"eval-{policy_iteration_step:06d}.json"),
                        "w",
                    ),
                    indent=4,
                )
                metrics["time/eval"] = time.time() - st
                metrics["eval/episode_return"] = np.mean(
                    [
                        sum(transition["reward"] for transition in episode)
                        for episode in eval_episodes
                    ]
                )
                metrics["eval/support"] = len(eval_episodes)

        # collect episodes with parallel environments
        print(f"🎲 Start collecting episodes at step {policy_iteration_step}")
        st = time.time()
        episodes_buffer = []
        while True:
            batch_episodes = await asyncio.gather(
                *[
                    collect_episode(sampling_client, sampling_params, env, config, tokenizer)
                    for env in envs
                ]
            )
            batch_episodes = [x for x in batch_episodes if x != []]
            episodes_buffer.extend(batch_episodes)
            if sum([len(ep) for ep in episodes_buffer]) >= config.batch_size:
                break
        metrics["time/sample"] = time.time() - st
        metrics["sampler/unfinished_rollout"] = np.mean(
            [
                np.mean([transition["unfinished"] for transition in episode])
                for episode in episodes_buffer
            ]
        )
        metrics["sampler/episode_return"] = np.mean(
            [sum(transition["reward"] for transition in episode) for episode in episodes_buffer]
        )
        metrics["sampler/num_turns_per_episode"] = np.mean(
            [len(episode) for episode in episodes_buffer]
        )
        gen_tokens_lens = [
            sum(len(transition["act_tokens"]) for transition in episode)
            for episode in episodes_buffer
        ]
        metrics["sampler/action_num_tokens"] = np.mean(gen_tokens_lens)
        metrics["sampler/num_episodes"] = len(episodes_buffer)

        # print at most two episodes for debugging purposes
        for n, episode in enumerate(episodes_buffer):
            print(f"----- episode {n} -----")
            for t, transition in enumerate(episode):
                obs = tokenizer.decode(transition["obs_tokens"])
                act = tokenizer.decode(transition["act_tokens"])
                obs = obs[:196] + "\n...\n" + obs[-200:] if len(obs) > 396 else obs
                act = act[:196] + "\n...\n" + act[-200:] if len(act) > 396 else act
                print(f"turn={t + 1}")
                print(colored(obs, "blue"))
                print(colored(act, "light_red", attrs=["bold"]))
                print(
                    colored(
                        "reward=" + str(transition["reward"]),
                        "light_magenta",
                        attrs=["bold"],
                    )
                )
            if n > 0:
                break

        # prepare transitions
        transitions = []
        for episode in episodes_buffer:
            # One transition typically consists of (s, a, r).
            # Here we augment it with a Monte Carlo return to
            # serve as the advantage estimation.
            rewards = [transition["reward"] for transition in episode]
            # Compute returns
            cur = 0.0
            for i in reversed(range(len(rewards))):
                cur = rewards[i] + config.gamma * cur
                episode[i]["return"] = cur
            transitions.extend(episode)

        # return batch normalization (https://arxiv.org/pdf/2510.01051#page=5.73 shows it's effective)
        if config.use_rebn:
            returns = torch.tensor([transition["return"] for transition in transitions]).float()
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            for i, transition in enumerate(transitions):
                transition["return"] = returns[i].item()

        # prepare training datums compatible with Tinker API
        training_datums = prepare_training_datums(transitions, 1 / len(transitions))

        # training step
        print(f"🎈 Start training at step {policy_iteration_step}")
        st = time.time()
        fwd_bwd_future = await training_client.forward_backward_async(
            training_datums,
            loss_fn=config.loss_fn,
        )
        optim_step_future = await training_client.optim_step_async(adam_params)
        total_gradient_steps += 1
        total_tokens_trained += sum([d.model_input.chunks[0].length for d in training_datums])
        fwd_bwd_result = await fwd_bwd_future.result_async()
        _ = await optim_step_future.result_async()
        metrics["time/train"] = time.time() - st
        metrics["train/n_samples"] = len(training_datums)

        # compute policy entropy and sampler-learner difference
        act_token_logprobs = []
        act_token_diffs = []
        for i in range(config.batch_size):
            transition = transitions[i]
            train_output = fwd_bwd_result.loss_fn_outputs[i]
            act_token_logprobs.extend(transition["act_logprobs"])
            act_token_diffs.append(
                torch.tensor(transition["act_logprobs"])
                - train_output["logprobs"].to_torch()[-len(transition["act_logprobs"]) :]
            )
        act_token_diffs = torch.cat(act_token_diffs)
        kl_sample_train_v1 = act_token_diffs.mean().item()
        kl_sample_train_v2 = 0.5 * (act_token_diffs**2).mean().item()
        metrics["sampler/token_entropy"] = -torch.tensor(act_token_logprobs).mean().item()
        metrics["train/kl_sample_train_v1"] = kl_sample_train_v1
        metrics["train/kl_sample_train_v2"] = kl_sample_train_v2
        metrics["train/total_gradient_steps"] = total_gradient_steps
        metrics.update(**{f"train/{k}": v for k, v in fwd_bwd_result.metrics.items()})

        pprint.pprint(metrics)
        wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
