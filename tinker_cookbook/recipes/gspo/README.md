# Group Sequence Policy Optimization (GSPO)

This recipe implements GSPO from ["Group Sequence Policy Optimization"](https://arxiv.org/abs/2507.18071) as a GSM8K training loop using `forward_backward_custom`.

GSPO is similar to GRPO, but it computes one importance ratio for the whole sampled response instead of one clipped ratio per token. The sequence ratio is the geometric mean of token probability ratios:

```text
s_i = exp(mean_t(log pi_theta(y_t) - log pi_old(y_t)))
loss_i = -min(s_i * A_i, clip(s_i, clip_low, clip_high) * A_i)
```

Advantages are normalized within each rollout group by standard deviation. Groups with identical rewards are skipped because they have no policy-gradient signal.

## Running

Set `TINKER_API_KEY`, then run a short smoke test:

```bash
python -m tinker_cookbook.recipes.gspo.train \
    batch_size=8 \
    group_size=4 \
    max_steps=3 \
    max_tokens=128
```

Run a larger GSM8K experiment:

```bash
python -m tinker_cookbook.recipes.gspo.train \
    model_name=meta-llama/Llama-3.1-8B \
    batch_size=128 \
    group_size=16 \
    learning_rate=4e-5 \
    lora_rank=32 \
    max_tokens=256
```

By default, logs are written to a timestamped directory under `/tmp/tinker-examples/gspo/`. Pass `log_path=...` to resume or to use a specific output directory.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `meta-llama/Llama-3.1-8B` | Base model or checkpoint model family |
| `batch_size` | `128` | Number of GSM8K questions per training batch |
| `group_size` | `16` | Rollouts sampled per question |
| `learning_rate` | `4e-5` | Adam learning rate |
| `lora_rank` | `32` | LoRA adapter rank |
| `max_tokens` | `256` | Maximum sampled completion length |
| `clip_low` | `0.9997` | Lower sequence-ratio clip bound |
| `clip_high` | `1.0004` | Upper sequence-ratio clip bound |
| `max_steps` | `None` | Optional cap on training batches |
| `load_checkpoint_path` | `None` | Optional Tinker checkpoint path for initial weights |
| `wandb_project` | `None` | Optional W&B project name |

## Expected Metrics

The main reward metric is `reward/total`, the average GSM8K correctness across rollout groups before filtering uniform-reward groups. The curve is noisy because rewards are binary and on-policy samples change every batch, but a healthy run should show `reward/total` trending upward over many batches. If the metric stays near zero, first check that the model is producing answers in `\boxed{...}` because the reward parser requires that format.

Training metrics from the custom loss are logged under `train/`, including:

| Metric | Meaning |
|--------|---------|
| `train/gspo_loss` | Mean clipped GSPO objective for the batch |
| `train/clip_frac` | Fraction of sequence ratios outside the clip range |
| `train/mean_log_ratio` | Mean sequence-level log ratio |

## Files

| File | Description |
|------|-------------|
| `loss.py` | Pure torch GSPO loss closure for `forward_backward_custom` |
| `train.py` | GSM8K training loop |
| `gspo_test.py` | Unit tests for the loss |
