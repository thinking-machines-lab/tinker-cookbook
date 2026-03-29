import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 14: Full RLHF Pipeline

    Train a model through the complete 3-stage RLHF pipeline:

    ```
    Stage 1: SFT              Stage 2: Preference Model     Stage 3: RL
    +------------------+      +------------------------+     +-------------------+
    | Base model       |      | Base model             |     | SFT policy        |
    | + no_robots data | ---> | + HHH pairwise data    | --> | + PM as reward    |
    | = initial policy |      | = preference model     |     | = RLHF policy     |
    +------------------+      +------------------------+     +-------------------+
    ```

    Each stage builds on the previous one. The SFT policy initializes the RL agent, and the preference model provides the reward signal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    We use Llama 3.2 3B as the base model. All three stages use LoRA for parameter-efficient training.
    """)
    return


@app.cell
def _():
    import asyncio
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import tinker
    from tinker_cookbook import checkpoint_utils, model_info
    from tinker_cookbook.renderers import TrainOnWhat

    BASE_MODEL = "meta-llama/Llama-3.2-3B"
    LORA_RANK = 64
    MAX_LENGTH = 16384
    BATCH_SIZE = 256
    LOG_ROOT = "/tmp/tinker-examples/rlhf-tutorial"

    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Renderer:    {renderer_name}")
    print(f"LoRA rank:   {LORA_RANK}")

    return (
        BASE_MODEL,
        BATCH_SIZE,
        LORA_RANK,
        LOG_ROOT,
        MAX_LENGTH,
        asyncio,
        checkpoint_utils,
        model_info,
        renderer_name,
        tinker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 1: Supervised Fine-Tuning (SFT)

    Train the base model on the [no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) dataset -- human-written instruction-following examples from the InstructGPT methodology. This produces the initial policy that the RL stage will refine.

    Key choices:
    - **Dataset**: NoRobots -- clean, human-written instruction data
    - **Loss**: standard next-token prediction on assistant messages only (`TrainOnWhat.ALL_ASSISTANT_MESSAGES`)
    - **Learning rate**: 2e-4 (standard SFT rate)

    Expected: `test/nll` decreases from ~1.99 to ~1.92 in 20 steps.
    """)
    return


@app.cell
def _(
    BASE_MODEL,
    BATCH_SIZE,
    LOG_ROOT,
    LORA_RANK,
    MAX_LENGTH,
    TrainOnWhat,
    asyncio,
    renderer_name,
):
    from tinker_cookbook.recipes.chat_sl.chat_datasets import NoRobotsBuilder
    from tinker_cookbook.supervised import train as supervised_train
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    # Configure the SFT dataset
    sft_common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    sft_dataset_builder = NoRobotsBuilder(common_config=sft_common_config)

    # Configure and run SFT training
    sft_log_path = f"{LOG_ROOT}/sft"
    sft_config = supervised_train.Config(
        log_path=sft_log_path,
        model_name=BASE_MODEL,
        renderer_name=renderer_name,
        dataset_builder=sft_dataset_builder,
        evaluator_builders=[],
        num_epochs=1,
        learning_rate=2e-4,
        lr_schedule="linear",
        save_every=100,
        eval_every=20,
        lora_rank=LORA_RANK,
        wandb_project=None,
        wandb_name="rlhf-tutorial-sft",
        max_steps=None,
    )

    asyncio.run(supervised_train.main(sft_config))
    print("Stage 1 (SFT) complete.")

    return (
        NoRobotsBuilder,
        sft_config,
        sft_dataset_builder,
        sft_log_path,
        supervised_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 2: Train the Preference Model

    Train a model to judge which of two completions is better, using the [Anthropic HHH](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset of pairwise comparisons.

    The `ComparisonRenderer` formats each pair as:
    ```
    [prompt] ==== Completion A ==== [text] ==== Completion B ==== [text] ==== Preference ====
    ```

    The model learns to predict "A" or "B" -- which completion the human preferred. This model becomes the reward signal for Stage 3.

    Key choices:
    - **Dataset**: HHH -- Anthropic's helpful/harmless/honest pairwise preference data
    - **Learning rate**: 3e-4 (slightly higher than SFT, since the task is simpler)

    Expected: `test/nll` drops from ~7 to ~0.7 in 40 steps, converging to ~0.55 by step 600.
    """)
    return


@app.cell
def _(
    BASE_MODEL,
    BATCH_SIZE,
    LOG_ROOT,
    LORA_RANK,
    MAX_LENGTH,
    asyncio,
    renderer_name,
    supervised_train,
):
    from tinker_cookbook.preference.preference_datasets import (
        ChatDatasetBuilderFromComparisons,
    )
    from tinker_cookbook.recipes.preference.datasets import HHHComparisonBuilder
    from tinker_cookbook.supervised.types import (
        ChatDatasetBuilderCommonConfig as CommonConfig,
    )

    # The HHH dataset provides labeled pairwise comparisons
    comparison_builder = HHHComparisonBuilder()

    # Wrap comparisons with the renderer for supervised training
    rm_common_config = CommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )
    rm_dataset_builder = ChatDatasetBuilderFromComparisons(
        common_config=rm_common_config,
        comparison_builder=comparison_builder,
    )

    # Train the preference model
    rm_log_path = f"{LOG_ROOT}/rm"
    rm_config = supervised_train.Config(
        log_path=rm_log_path,
        model_name=BASE_MODEL,
        renderer_name=renderer_name,
        dataset_builder=rm_dataset_builder,
        evaluator_builders=[],
        num_epochs=1,
        learning_rate=3e-4,
        lr_schedule="linear",
        save_every=100,
        eval_every=20,
        lora_rank=LORA_RANK,
        wandb_project=None,
        wandb_name="rlhf-tutorial-rm",
        max_steps=None,
    )

    asyncio.run(supervised_train.main(rm_config))
    print("Stage 2 (Preference Model) complete.")

    return (
        HHHComparisonBuilder,
        comparison_builder,
        rm_config,
        rm_log_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 3: RL Training

    Now we combine the SFT policy (Stage 1) with the preference model (Stage 2) to run reinforcement learning.

    The RL loop works as follows:
    1. For each prompt, sample **multiple completions** from the policy (`group_size=4`)
    2. Use the preference model to score **all pairs** of completions (tournament)
    3. Compute a reward for each completion based on its **win fraction**
    4. Update the policy to produce more of the winning completions

    This is a form of self-play: the policy competes against itself, graded by the preference model.

    Key choices:
    - **Learning rate**: 1e-5 (much lower than SFT -- RL updates are noisier)
    - **Group size**: 4 completions per prompt
    - **Tournament**: `ALL_PAIRS_BOTH_WAYS` -- every pair is evaluated in both orderings

    Expected: `test/win_rate` increases from ~40% to ~70% in 100 steps.
    """)
    return


@app.cell
def _(
    BASE_MODEL,
    BATCH_SIZE,
    HHHComparisonBuilder,
    LOG_ROOT,
    LORA_RANK,
    asyncio,
    checkpoint_utils,
    model_info,
    renderer_name,
    rm_log_path,
    sft_log_path,
):
    from tinker_cookbook.preference.comparison_policy_evaluator import (
        ComparisonEvaluator,
    )
    from tinker_cookbook.preference.types import PreferenceModelBuilderFromChatRenderer
    from tinker_cookbook.rl import preference_envs, train as rl_train

    # Load checkpoints from Stages 1 and 2
    sft_ckpt = checkpoint_utils.get_last_checkpoint(sft_log_path)
    rm_ckpt = checkpoint_utils.get_last_checkpoint(rm_log_path)
    assert sft_ckpt is not None, f"No SFT checkpoint in {sft_log_path}"
    assert rm_ckpt is not None, f"No RM checkpoint in {rm_log_path}"

    print(f"SFT checkpoint: {sft_ckpt.state_path}")
    print(f"RM checkpoint:  {rm_ckpt.sampler_path}")

    # Build the preference model from the RM checkpoint
    pm_builder = PreferenceModelBuilderFromChatRenderer(
        renderer_name=renderer_name,
        model_name=BASE_MODEL,
        rm_weights_path=rm_ckpt.sampler_path,
    )

    # Build the RL dataset: prompts from HHH, rewards from the preference model
    rl_comparison_builder = HHHComparisonBuilder()
    rl_dataset_builder = preference_envs.PairwisePreferenceRLDatasetBuilder(
        comparison_builder=rl_comparison_builder,
        policy_renderer_name=renderer_name,
        policy_model_name=BASE_MODEL,
        preference_model_builder=pm_builder,
        batch_size=BATCH_SIZE,
        group_size=4,
        tournament_pattern=preference_envs.TournamentPattern.ALL_PAIRS_BOTH_WAYS,
    )

    # Build an evaluator that measures win rate on held-out comparisons
    def make_evaluator() -> ComparisonEvaluator:
        eval_builder = HHHComparisonBuilder(test_size=256)
        _, test_set = eval_builder.get_train_and_test_datasets()
        assert test_set is not None
        comparisons = [
            eval_builder.example_to_labeled_comparison(ex).comparison
            for ex in test_set
            if eval_builder.example_to_labeled_comparison(ex) is not None
        ]
        return ComparisonEvaluator(
            preference_model_builder=pm_builder,
            comparisons=comparisons,
            renderer_name=renderer_name,
            model_name_for_tokenizer=BASE_MODEL,
        )

    # Configure and run RL
    rl_log_path = f"{LOG_ROOT}/rl"
    rl_config = rl_train.Config(
        model_name=BASE_MODEL,
        renderer_name=renderer_name,
        dataset_builder=rl_dataset_builder,
        load_checkpoint_path=sft_ckpt.state_path,
        learning_rate=1e-5,
        max_tokens=1024,
        log_path=rl_log_path,
        evaluator_builders=[make_evaluator],
        wandb_project=None,
        wandb_name="rlhf-tutorial-rl",
        lora_rank=LORA_RANK,
        save_every=100,
        eval_every=20,
        num_groups_to_log=4,
        max_steps=None,
    )

    asyncio.run(rl_train.main(rl_config))
    print("Stage 3 (RL) complete.")

    return (rl_config, rl_log_path)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation: Compare Base vs RLHF Policy

    After training, we can compare the base model against the RLHF-trained model by sampling from both and judging with the preference model.
    """)
    return


@app.cell
def _(
    BASE_MODEL,
    checkpoint_utils,
    rl_log_path,
    tinker,
):
    from tinker import types

    # Create sampling clients for both models
    service = tinker.ServiceClient()

    # Base model (no fine-tuning)
    base_sampler = service.create_sampling_client(base_model=BASE_MODEL)

    # RLHF model (load RL checkpoint)
    rl_ckpt = checkpoint_utils.get_last_checkpoint(rl_log_path)
    assert rl_ckpt is not None
    rlhf_sampler = service.create_sampling_client(
        base_model=BASE_MODEL,
        load_path=rl_ckpt.sampler_path,
    )

    tokenizer_eval = base_sampler.get_tokenizer()

    # Sample from both models on the same prompt
    test_prompt = "What is the most important thing to consider when learning a new programming language?"
    prompt_tokens = types.ModelInput.from_ints(tokenizer_eval.encode(test_prompt))
    params = types.SamplingParams(max_tokens=200, temperature=0.7, stop=["\n\n"])

    base_result = base_sampler.sample(prompt=prompt_tokens, sampling_params=params, num_samples=1).result()
    rlhf_result = rlhf_sampler.sample(prompt=prompt_tokens, sampling_params=params, num_samples=1).result()

    print("=== Base Model ===")
    print(test_prompt + tokenizer_eval.decode(base_result.sequences[0].tokens))
    print()
    print("=== RLHF Model ===")
    print(test_prompt + tokenizer_eval.decode(rlhf_result.sequences[0].tokens))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    The 3-stage RLHF pipeline:

    | Stage | Goal | Dataset | Key Metric |
    |-------|------|---------|------------|
    | SFT | Initialize policy on instructions | no_robots | NLL: 1.99 -> 1.92 |
    | Preference Model | Learn human preferences | HHH (Anthropic) | NLL: 7 -> 0.55 |
    | RL | Optimize policy against PM | HHH prompts | Win rate: 40% -> 70% |

    Key takeaways:
    - **SFT** gives the model basic instruction-following ability
    - **Preference Model** provides a learned reward signal, replacing expensive human feedback at RL time
    - **RL** uses self-play with tournament scoring -- sample multiple completions, grade all pairs, reward winners
    - Learning rates decrease across stages: 2e-4 (SFT) > 3e-4 (PM) > 1e-5 (RL) -- RL needs small steps due to noisy gradients

    For production use, see the [RLHF recipe](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/preference/rlhf/rlhf_pipeline.py) which adds CLI configuration, wandb logging, and checkpoint management.
    """)
    return


if __name__ == "__main__":
    app.run()
