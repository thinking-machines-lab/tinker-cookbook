# Distillation

Distillation refers to a class of methods where a teacher model is supervising the training of a student model, which can often be more efficient than training the student model in isolation. We provide off-policy and on-policy distillation recipes on top of the [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M), [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K), and [Tulu3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)* datasets.

Specifically, we provide the scripts needed to reproduce our experiments from the [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation) blog post, which can be run with LoRA using Tinker.

\* For our post, we regenerated the assistant turns using [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

## Distillation for reasoning

Our results can be reproduced by running:
1. Supervised finetuning on [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
2. On-policy distillation on [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K)

### Supervised finetuning

We observe an AIME'24 score of ~55% using a rank-128 LoRA after 3000 steps. We use a learning rate of 1e-3 with LoRA and 1e-4 with full finetuning.

```bash
python -m tinker_cookbook.recipes.distillation.off_policy_reasoning \
    model_name=Qwen/Qwen3-8B-Base \
    learning_rate=1e-3 \
    batch_size=128 \
    lora_rank=128 \
    wandb_project=cookbook_distillation
```

### On-policy distillation

We observe an AIME'24 score of ~65% using a rank-128 LoRA after 100 steps. For on-policy distillation experiments, we use a learning rate of 1e-4 with LoRA and 5e-5 with full finetuning.

```bash
python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
    model_name=Qwen/Qwen3-8B-Base \
    load_checkpoint_path=tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final \
    dataset=deepmath \
    learning_rate=1e-4 \
    groups_per_batch=512 \
    lora_rank=128 \
    wandb_project=cookbook_distillation
```

This script can also be used to replicate the experiments in our Discussion section, after you have run RL to obtain an appropriate checkpoint for the teacher model.

### Checkpoints

The results of running the above scripts with various LoRA ranks can be found here:

| Stage | Rank 8 | Rank 32 | Rank 128 |
|-------|--------|---------|----------|
| Supervised finetuning (SFT) | `tinker://c15f09f1-f225-5f98-bab1-ec8dfac5da2a:train:0/weights/final` | `tinker://b9190d16-c849-51d5-a690-1b5de146a284:train:0/weights/final` | `tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final` |
| On-policy distillation | `tinker://4a97bc02-f4d0-5ecd-888a-3e8cc5b0f7f6:train:0/sampler_weights/000080` | `tinker://bfffa2b2-a78f-59be-a2ef-cc9188bfce7e:train:0/sampler_weights/000080` | `tinker://1dd8de47-be86-54d3-9355-ebf80827be26:train:0/sampler_weights/000080` |

See the on-policy distillation launch command above for an example of how to load the checkpoint path.

## Distillation for personalization

In this section, we ran:
1. Supervised finetuning on internal documents + resampled Tulu3 data
2. On-policy distillation on [Tulu3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) prompts

### On-policy distillation

In our experiment, we saw [IF-eval](https://huggingface.co/datasets/google/IFEval) recover within approximately 100 steps; we expect similar results in other settings. In order to use this script, you will have to provide your own SFT initialization.

```bash
python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
    model_name=Qwen/Qwen3-8B-Base \
    dataset=tulu3 \
    learning_rate=1e-4 \
    groups_per_batch=64 \
    lora_rank=128 \
    wandb_project=cookbook_distillation
```

## Additional details

### Reward calculation

In on-policy distillation, we use an `Environment` that has no rewards (neither correctness nor format). The only supervision comes from minimizing the KL against a teacher model. You can optionally increase `kl_discount_factor` to optimize discounted future KL, but we generally do not observe this to improve performance.

### Distillation with multiple teachers

For every dataset, we can define a teacher model and batch size (`groups_per_batch`) to use:

```python
{
    "dataset_builder": RLDatasetBuilder,
    "teacher_model": {
        "base_model": str,  # e.g. "Qwen/Qwen3-32B"
        "load_checkpoint_path": str | None  # e.g. "tinker://<unique_id>/sampler_weights/final
    },
    "groups_per_batch": int
}
```

The trainer will then sample from each configuration, and concatenate all the individual dataset batches to form the batch for training. This can be used to run multi-teacher distillation, although we do not showcase this in our blog post.

```bash
python -m tinker_cookbook.recipes.distillation.on_policy_multi_teacher \
    learning_rate=1e-4 \
    deepmath_groups_per_batch=256 \
    tulu3_groups_per_batch=256 \
    lora_rank=128 \
    wandb_project=cookbook_distillation
```
