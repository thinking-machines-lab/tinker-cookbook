import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from tinker_cookbook import tokenizer_utils

    return (tokenizer_utils,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 407: Full RLHF Pipeline

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

    This is the longest tutorial in the series. Runtime and win-rate improvements depend on the model, data, and service load. The tutorial caps RL at 40 steps; inspect evaluation metrics before increasing the budget. The original tutorial's timings and win rates were measured on a different model and are not Inkling 2 Mini benchmarks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    We use `thinkingmachines/Inkling-2-Mini` as the starting checkpoint for further post-training. All three stages use LoRA for parameter-efficient training.
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import tinker

    from tinker_cookbook import checkpoint_utils, cli_utils, model_info
    from tinker_cookbook.renderers import TrainOnWhat

    BASE_MODEL = "thinkingmachines/Inkling-2-Mini"
    LORA_RANK = 64
    MAX_LENGTH = 16384
    BATCH_SIZE = 256
    LOG_ROOT = cli_utils.runs_path("tutorials", "rlhf")

    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Renderer:    {renderer_name}")
    print(f"LoRA rank:   {LORA_RANK}")
    return (
        BASE_MODEL,
        BATCH_SIZE,
        LOG_ROOT,
        LORA_RANK,
        MAX_LENGTH,
        TrainOnWhat,
        checkpoint_utils,
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
    - **Learning rate**: 2e-4 (an illustrative starting point to sweep)

    Monitor held-out `test/nll` to check whether SFT improves prediction on your data.
    """)
    return


@app.cell
def _(mo):
    api_key = mo.ui.text(kind="password", label="Paste your Tinker API key")
    api_key  # noqa: B018
    return (api_key,)


@app.cell
async def _(
    BASE_MODEL,
    BATCH_SIZE,
    LOG_ROOT,
    LORA_RANK,
    MAX_LENGTH,
    TrainOnWhat,
    api_key,
    mo,
    renderer_name,
):
    import os

    mo.stop(
        "TINKER_API_KEY" not in os.environ and not api_key.value,
        "Paste your API key above",
    )

    if api_key.value:
        os.environ["TINKER_API_KEY"] = api_key.value

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
        recipe_name="tutorial_rlhf_sft",
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

    await supervised_train.main(sft_config)
    print("Stage 1 (SFT) complete.")
    return sft_log_path, supervised_train


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

    Monitor held-out `test/nll` to check whether the preference model learns to predict human choices.
    """)
    return


@app.cell
async def _(
    BASE_MODEL,
    BATCH_SIZE,
    LOG_ROOT,
    LORA_RANK,
    MAX_LENGTH,
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
        recipe_name="tutorial_rlhf_rm",
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

    await supervised_train.main(rm_config)
    print("Stage 2 (Preference Model) complete.")
    return HHHComparisonBuilder, rm_log_path


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

    Track `test/win_rate` to measure improvement over the initial policy.

    We cap training at `max_steps=40` to limit the tutorial run. Raise `max_steps` after checking held-out win rate, reward variance, and KL; these hyperparameters have not been calibrated for Inkling 2 Mini.
    """)
    return


@app.cell
async def _(
    BASE_MODEL,
    BATCH_SIZE,
    HHHComparisonBuilder,
    LOG_ROOT,
    LORA_RANK,
    checkpoint_utils,
    renderer_name,
    rm_log_path,
    sft_log_path,
):
    from tinker_cookbook.preference.comparison_policy_evaluator import (
        ComparisonEvaluator,
    )
    from tinker_cookbook.preference.types import PreferenceModelBuilderFromChatRenderer
    from tinker_cookbook.rl import preference_envs
    from tinker_cookbook.rl import train as rl_train

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
        recipe_name="tutorial_rlhf_rl",
        renderer_name=renderer_name,
        dataset_builder=rl_dataset_builder,
        load_checkpoint_path=sft_ckpt.state_path,
        learning_rate=1e-5,
        max_tokens=4096,
        log_path=rl_log_path,
        evaluator_builders=[make_evaluator],
        wandb_project=None,
        wandb_name="rlhf-tutorial-rl",
        lora_rank=LORA_RANK,
        save_every=100,
        eval_every=10,
        num_groups_to_log=4,
        # Limit the tutorial to 40 steps; use held-out metrics to set the budget. A final
        # checkpoint is saved at the end of training either way. Raise this
        # (or set to None) to train longer.
        max_steps=40,
    )

    await rl_train.main(rl_config)
    print("Stage 3 (RL) complete.")
    return (rl_log_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation: Compare Base vs RLHF Policy

    After training, we can compare the base model against the RLHF-trained model by sampling from both and judging with the preference model.
    """)
    return


@app.cell
async def _(
    BASE_MODEL,
    checkpoint_utils,
    renderer_name,
    rl_log_path,
    tinker,
    tokenizer_utils,
):
    from tinker import types

    from tinker_cookbook import renderers

    # Create sampling clients for both models
    service = tinker.ServiceClient()

    # Base model (no fine-tuning)
    base_sampler = await service.create_sampling_client(base_model=BASE_MODEL)

    # RLHF model (load RL checkpoint)
    rl_ckpt = checkpoint_utils.get_last_checkpoint(rl_log_path)
    assert rl_ckpt is not None
    rlhf_sampler = await service.create_sampling_client(
        model_path=rl_ckpt.sampler_path,
        base_model=BASE_MODEL,
    )

    tokenizer_eval = tokenizer_utils.get_tokenizer(BASE_MODEL)
    renderer_eval = renderers.get_renderer(renderer_name, tokenizer_eval)

    # Sample from both models on the same prompt
    test_prompt = (
        "What is the most important thing to consider when learning a new programming language?"
    )
    prompt_tokens = renderer_eval.build_generation_prompt(
        [{"role": "user", "content": test_prompt}], effort=0.9
    )
    params = types.SamplingParams(
        max_tokens=4096, temperature=1.0, stop=renderer_eval.get_stop_sequences()
    )

    base_result = await base_sampler.sample(
        prompt=prompt_tokens, sampling_params=params, num_samples=1
    )
    rlhf_result = await rlhf_sampler.sample(
        prompt=prompt_tokens, sampling_params=params, num_samples=1
    )

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
    | SFT | Initialize policy on instructions | no_robots | Track held-out NLL |
    | Preference Model | Learn human preferences | HHH (Anthropic) | Track held-out NLL |
    | RL | Optimize policy against PM | HHH prompts | Track held-out win rate |

    Key takeaways:
    - **SFT** adapts the instruction-tuned model to the target dataset
    - **Preference Model** provides a learned reward signal, replacing expensive human feedback at RL time
    - **RL** uses self-play with tournament scoring -- sample multiple completions, grade all pairs, reward winners
    - Learning rates: 2e-4 (SFT), 3e-4 (PM), 1e-5 (RL) -- RL needs much smaller steps due to noisy gradients

    For production use, see the [RLHF recipe](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/preference/rlhf/rlhf_pipeline.py) which adds CLI configuration, wandb logging, and checkpoint management.
    """)
    return


if __name__ == "__main__":
    app.run()
