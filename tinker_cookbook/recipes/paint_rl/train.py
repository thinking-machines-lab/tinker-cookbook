"""RL launcher: teach Inkling-Small to paint watercolors with p5.brush code.

    # LLM-as-a-Verifier reward (pairwise round-robin tournament)
    python -m tinker_cookbook.recipes.paint_rl.train reward_mode=verifier log_path=/tmp/paint_rl/verifier

    # LLM-as-a-Judge baseline (absolute A-J grade per painting and criterion)
    python -m tinker_cookbook.recipes.paint_rl.train reward_mode=judge log_path=/tmp/paint_rl/judge

The policy is ``thinkingmachines/Inkling-Small`` on Tinker. The reward model
is Inkling-Small itself at thinking effort 0.2 ("low"); the two reward modes
share every other setting, so their runs can be compared directly on the held-out
score from ``eval_checkpoints.py`` (DeepSeek-V4.1-Flash on DeepSeek's API, which
never feeds back into training).

Reward-model knobs are nested: ``verifier.n_evaluations=4``,
``judge.effort=0.2``, ``verifier.max_concurrency=1200``, ... (defaults in
``verifier.py``). Training needs only ``TINKER_API_KEY``; held-out evaluation
also needs ``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.paint_rl.env import PaintRLDatasetBuilder, RewardWeights
from tinker_cookbook.recipes.paint_rl.prompts import N_TEST_PROMPTS, N_TRAIN_PROMPTS
from tinker_cookbook.recipes.paint_rl.verifier import (
    EvaluatorConfig,
    JudgeConfig,
    RewardMode,
    VerifierConfig,
)
from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
from tinker_cookbook.rl.rollout_presets import default_rollout_config_for_model
from tinker_cookbook.rl.train import AsyncConfig, Config, main

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Run-level knobs; reward-model defaults live in ``verifier.py``'s configs."""

    # Policy
    model_name: str = "thinkingmachines/Inkling-Small"
    lora_rank: int = 32
    renderer_name: str | None = None  # default: tml_v0
    load_checkpoint_path: str | None = None
    # Thinking effort in [0, 1); 0.7 is the "medium" preset. Sketches land at
    # ~1.1k tokens at this effort, so 8192 leaves plenty of headroom.
    policy_effort: float = 0.7
    max_tokens: int = 8192
    temperature: float = 1.0

    # Optimization. Inkling has no calibrated get_lr(); sweep around this.
    learning_rate: float = 4e-5
    # Each step: 8 prompts x 5 rollouts.
    group_size: int = 5
    groups_per_batch: int = 8
    max_steps: int = 800
    num_substeps: int = 1
    kl_penalty_coef: float = 0.0
    loss_fn: LossFnType = "importance_sampling"
    max_steps_off_policy: int | None = None

    # Reward. Both modes default to Inkling-Small at effort 0.2 ("low") with
    # the same token budget and concurrency (see verifier.py); only the
    # scoring differs.
    reward_mode: RewardMode = "verifier"
    # Pairwise round-robin tournament, 3 criteria x 4 slot-swapped repeats.
    verifier: VerifierConfig = chz.field(default_factory=VerifierConfig)
    # Absolute A-J grade (1-10) per painting on each of the verifier's three criteria.
    judge: JudgeConfig = chz.field(default_factory=JudgeConfig)
    weights: RewardWeights = chz.field(default_factory=RewardWeights)

    # Held-out evaluator for inline evaluation (eval_every > 0) only;
    # eval_checkpoints.py has its own. None turns it off.
    evaluator: EvaluatorConfig | None = chz.field(default_factory=EvaluatorConfig)

    # Data / rendering
    # Training prompts are balanced over subjects, colors and styles. Held-out
    # prompts are a sample of the pool spread over every subject.
    n_train_prompts: int = N_TRAIN_PROMPTS
    n_test_prompts: int = N_TEST_PROMPTS
    test_group_size: int = 5
    canvas_size: int = 512
    # Sketches rendered at once: one step's 40 rollouts, ~4 pages per browser.
    render_concurrency: int = 40
    # Sketches grow longer over training (about 2x by step 150), and software
    # WebGL rendering of a long one can pass 2 min when the CPU is shared; a
    # timeout scores 0, so leave plenty of headroom.
    render_timeout_s: float = 300.0
    # Training randomness (prompt order, reward-model seeds). It leaves the prompt
    # split alone, so runs with different seeds share the same held-out prompts.
    seed: int = 0
    # Seeds the train/held-out prompt split.
    split_seed: int = 0

    # Logging
    log_path: str | None = None
    artifact_dir: str | None = None  # default: <log_path>/paintings
    wandb_project: str | None = None
    wandb_name: str | None = None
    # Held-out evaluation inside the training loop blocks training, so it is
    # off by default; run eval_checkpoints.py alongside instead (it evaluates
    # every saved checkpoint in a separate process). Set >0 to evaluate inline.
    eval_every: int = 0
    # Checkpoint (training state + sampler weights) every 10 steps; each one is
    # evaluated on the held-out prompts by eval_checkpoints.py.
    save_every: int = 10
    # Lifetime of the periodic checkpoints on Tinker, in seconds. None keeps
    # them all indefinitely (the cookbook default expires them after 7 days);
    # the final checkpoint is always kept.
    checkpoint_ttl_seconds: int | None = None
    compute_post_kl: bool = False
    remove_constant_reward_groups: bool = False
    # Cap on scoring one group. A pairwise step queues ~480 reward-model calls,
    # which can take far longer than the 900 s Inkling preset allows when the
    # endpoint is busy; a group that hits the cap is dropped from the step.
    grader_timeout_s: float = 3600.0
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    base_url: str | None = None


def _termination_policy(cli: CLIConfig) -> TerminationRewardPolicy:
    """The model's default termination policy with this recipe's grading cap."""
    default = default_rollout_config_for_model(cli.model_name).termination
    if default is None:
        return TerminationRewardPolicy(grader_timeout_seconds=cli.grader_timeout_s)
    return chz.replace(default, grader_timeout_seconds=cli.grader_timeout_s)


async def cli_main(cli: CLIConfig) -> None:
    """Assemble the RL training config from the CLI config and train."""
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli.model_name,
        explicit_renderer_name=cli.renderer_name,
        load_checkpoint_path=cli.load_checkpoint_path,
        base_url=cli.base_url,
    )
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"paint-{cli.reward_mode}-{cli.model_name.split('/')[-1]}-r{cli.lora_rank}"
        f"-lr{cli.learning_rate}-g{cli.group_size}x{cli.groups_per_batch}-{stamp}"
    )
    log_path = cli.log_path or f"/tmp/tinker-examples/paint_rl/{run_name}"
    artifact_dir = cli.artifact_dir or str(Path(log_path) / "paintings")

    dataset_builder = PaintRLDatasetBuilder(
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        batch_size=cli.groups_per_batch,
        group_size=cli.group_size,
        n_batches=cli.max_steps,
        policy_effort=cli.policy_effort,
        reward_mode=cli.reward_mode,
        verifier_config=cli.verifier,
        judge_config=cli.judge,
        evaluator_config=cli.evaluator,
        weights=cli.weights,
        canvas_size=cli.canvas_size,
        render_concurrency=cli.render_concurrency,
        render_timeout_s=cli.render_timeout_s,
        n_train_prompts=cli.n_train_prompts,
        n_test_prompts=cli.n_test_prompts,
        test_group_size=cli.test_group_size,
        seed=cli.seed,
        split_seed=cli.split_seed,
        artifact_dir=artifact_dir,
    )
    config = Config(
        recipe_name="recipe_paint_rl",
        learning_rate=cli.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli.model_name,
        renderer_name=renderer_name,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name or run_name,
        log_path=log_path,
        base_url=cli.base_url,
        load_checkpoint_path=cli.load_checkpoint_path,
        compute_post_kl=cli.compute_post_kl,
        kl_penalty_coef=cli.kl_penalty_coef,
        num_substeps=cli.num_substeps,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        ttl_seconds=cli.checkpoint_ttl_seconds,
        loss_fn=cli.loss_fn,
        max_steps=cli.max_steps,
        async_config=AsyncConfig(
            max_steps_off_policy=cli.max_steps_off_policy,
            groups_per_batch=cli.groups_per_batch,
        )
        if cli.max_steps_off_policy is not None
        else None,
        # A prompt whose whole group scores the same teaches nothing once the
        # advantages are centered; drop it instead of paying for the backward pass.
        remove_constant_reward_groups=cli.remove_constant_reward_groups,
        termination=_termination_policy(cli),
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    logger.info("Paintings and sketches will be saved under %s", artifact_dir)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
