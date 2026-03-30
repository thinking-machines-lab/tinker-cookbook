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
    # Tutorial 202: Loss Functions

    Tinker provides several built-in loss functions for `forward_backward`, covering supervised learning and three flavors of policy gradient. When none of those fit, `forward_backward_custom` lets you define an arbitrary differentiable loss on log-probabilities.

    In this tutorial you will:

    1. Prepare simple training data
    2. Run `forward_backward` with `cross_entropy`, `importance_sampling`, `ppo`, and `cispo`
    3. Write a custom loss with `forward_backward_custom`
    4. Compare loss values and understand when to use each
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import tinker
    import torch
    from tinker import TensorData

    return TensorData, tinker, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Create a LoRA training client and a tokenizer. We use a small model to keep costs low.
    """)
    return


@app.cell
async def _(tinker):
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=16
    )
    tokenizer = training_client.get_tokenizer()

    print(f"Training client ready for {MODEL_NAME}")
    return MODEL_NAME, service_client, tokenizer, training_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare training data

    We need a `Datum` for each loss function. The inputs differ by loss type:

    | Loss | Required `loss_fn_inputs` |
    |---|---|
    | `cross_entropy` | `target_tokens`, `weights` |
    | `importance_sampling` | `target_tokens`, `logprobs`, `advantages` |
    | `ppo` | `target_tokens`, `logprobs`, `advantages` |
    | `cispo` | `target_tokens`, `logprobs`, `advantages` |

    We will create one SFT datum and one RL datum (reused across the RL losses).
    """)
    return


@app.cell
def _(TensorData, tinker, tokenizer, torch):
    # -- SFT datum (for cross_entropy) --
    prompt_text = "The capital of France is"
    target_text = " Paris."
    prompt_ids = tokenizer.encode(prompt_text)
    target_ids = tokenizer.encode(target_text)

    all_ids = prompt_ids + target_ids
    model_input_sft = tinker.ModelInput.from_ints(all_ids[:-1])

    # Target tokens: the next token at each position
    sft_target_tokens = all_ids[1:]
    # Weights: 0 for prompt positions, 1 for completion positions
    sft_weights = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(target_ids)

    sft_datum = tinker.Datum(
        model_input=model_input_sft,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(sft_target_tokens)),
            "weights": TensorData.from_torch(torch.tensor(sft_weights)),
        },
    )

    # -- RL datum (for importance_sampling, ppo, cispo) --
    # Simulate a sampled completion with fake logprobs and advantages
    model_input_rl = tinker.ModelInput.from_ints(all_ids[:-1])
    rl_target_tokens = all_ids[1:]
    # Fake sampling logprobs (as if from a sampler)
    rl_logprobs = [0.0] * (len(prompt_ids) - 1) + [-1.5] * len(target_ids)
    # Positive advantage: this completion was good
    rl_advantages = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(target_ids)

    rl_datum = tinker.Datum(
        model_input=model_input_rl,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(rl_target_tokens)),
            "logprobs": TensorData.from_torch(torch.tensor(rl_logprobs)),
            "advantages": TensorData.from_torch(torch.tensor(rl_advantages)),
        },
    )

    print(f"SFT datum: {len(all_ids)-1} tokens, {sum(sft_weights):.0f} completion tokens")
    print(f"RL datum:  {len(all_ids)-1} tokens, advantage=+1.0 on completion")
    return model_input_rl, model_input_sft, rl_datum, sft_datum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run forward_backward with each loss function

    We call `forward_backward` once per loss function. Since we are only comparing outputs (not actually training), we do **not** call `optim_step`.
    """)
    return


@app.cell
async def _(rl_datum, sft_datum, training_client):
    # Cross-entropy (SFT)
    ce_future = await training_client.forward_backward_async([sft_datum], loss_fn="cross_entropy")
    ce_result = await ce_future.result_async()
    print(f"cross_entropy      loss:sum = {ce_result.metrics['loss:sum']:.4f}")

    # Importance sampling (REINFORCE with IS correction)
    is_future = await training_client.forward_backward_async(
        [rl_datum], loss_fn="importance_sampling"
    )
    is_result = await is_future.result_async()
    print(f"importance_sampling loss:sum = {is_result.metrics['loss:sum']:.4f}")

    # PPO (clipped objective)
    ppo_future = await training_client.forward_backward_async([rl_datum], loss_fn="ppo")
    ppo_result = await ppo_future.result_async()
    print(f"ppo                loss:sum = {ppo_result.metrics['loss:sum']:.4f}")

    # CISPO (clipped ratio weighting the log-prob)
    cispo_future = await training_client.forward_backward_async([rl_datum], loss_fn="cispo")
    cispo_result = await cispo_future.result_async()
    print(f"cispo              loss:sum = {cispo_result.metrics['loss:sum']:.4f}")
    return ce_result, cispo_result, is_result, ppo_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect the outputs

    Each result contains `loss_fn_outputs` with per-token logprobs, and `metrics` with the scalar loss. The RL losses also return the probability ratio between the training and sampling policies.
    """)
    return


@app.cell
def _(ce_result, is_result):
    # Cross-entropy returns logprobs of target tokens
    ce_logprobs = ce_result.loss_fn_outputs[0]["logprobs"]
    print("cross_entropy logprobs (last 3 tokens):", ce_logprobs[-3:])

    # Importance sampling also returns logprobs
    is_logprobs = is_result.loss_fn_outputs[0]["logprobs"]
    print("importance_sampling logprobs (last 3):", is_logprobs[-3:])
    return ce_logprobs, is_logprobs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PPO with custom clipping thresholds

    PPO and CISPO accept `loss_fn_config` to override the default clip thresholds (0.8 and 1.2). Tighter clipping makes updates more conservative.
    """)
    return


@app.cell
async def _(rl_datum, training_client):
    # Tighter clipping
    ppo_tight_future = await training_client.forward_backward_async(
        [rl_datum],
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
    )
    ppo_tight = await ppo_tight_future.result_async()
    print(f"PPO (tight clip) loss:sum = {ppo_tight.metrics['loss:sum']:.4f}")

    # Wider clipping (more like vanilla IS)
    ppo_wide_future = await training_client.forward_backward_async(
        [rl_datum],
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.5, "clip_high_threshold": 1.5},
    )
    ppo_wide = await ppo_wide_future.result_async()
    print(f"PPO (wide clip)  loss:sum = {ppo_wide.metrics['loss:sum']:.4f}")
    return ppo_tight, ppo_wide


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom loss with forward_backward_custom

    `forward_backward_custom` lets you define any differentiable loss on the log-probabilities. The function signature is:

    ```python
    def my_loss(data: list[Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        ...
        return loss, metrics
    ```

    The `logprobs` tensors have `requires_grad=True`, so you can backprop through them. Tinker handles the chain rule to get gradients on model weights.

    Here is a simple example: a loss that penalizes low-confidence predictions by squaring the target-token logprobs.
    """)
    return


@app.cell
async def _(sft_datum, torch, training_client):
    def logprob_squared_loss(data, logprobs_list):
        """Sum of squared target-token logprobs. Penalizes low-confidence predictions."""
        total_loss = torch.tensor(0.0)
        for logprobs in logprobs_list:
            # logprobs close to 0 = high confidence (small penalty)
            # logprobs << 0 = low confidence (large penalty)
            total_loss = total_loss + (logprobs**2).sum()
        return total_loss, {"logprob_sq": total_loss.item()}

    custom_future = await training_client.forward_backward_custom_async(
        [sft_datum], logprob_squared_loss
    )
    custom_result = await custom_future.result_async()
    print(f"Custom loss metrics: {custom_result.metrics}")
    return custom_result, logprob_squared_loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A more practical custom loss: **DPO-style pairwise preference loss**. `forward_backward_custom` can operate on multiple datums at once, making it natural for losses that compare pairs of sequences.

    ```python
    def dpo_loss(data, logprobs_list):
        # data[0] = preferred, data[1] = rejected
        preferred_lp = logprobs_list[0].sum()
        rejected_lp = logprobs_list[1].sum()
        beta = 0.1
        loss = -torch.nn.functional.logsigmoid(beta * (preferred_lp - rejected_lp))
        return loss, {"dpo_loss": loss.item()}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When to use each loss

    | Loss | Use case | Key property |
    |---|---|---|
    | `cross_entropy` | Supervised fine-tuning (SFT), distillation | Standard NLL; trains on known-good outputs |
    | `importance_sampling` | RL with on-policy or near-on-policy data | Corrects for sampler/learner mismatch; unbounded ratio |
    | `ppo` | RL with multiple gradient steps per rollout | Clips the IS ratio to prevent large updates |
    | `cispo` | RL; alternative to PPO | Clips the ratio but applies it as a weight on log-prob; sometimes more stable |
    | `forward_backward_custom` | DPO, custom regularizers, research losses | Full flexibility; 1.5x FLOPs due to extra forward pass |

    **Rule of thumb:** Start with `cross_entropy` for SFT and `importance_sampling` for RL. Switch to `ppo` or `cispo` if you see training instability from large policy updates. Use `forward_backward_custom` for anything that does not fit the built-in losses.
    """)
    return


if __name__ == "__main__":
    app.run()
