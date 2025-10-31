# Supervised Learning

## SFT on NoRobots

```bash
python -m tinker_cookbook.recipes.chat_sl.train
    model_name=Qwen/Qwen3-8B-Base \
    dataset=no_robots \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    eval_every=20 \
    save_every=20 \
    wandb_project=cookbook_sl
```

After 140 steps of training, `test/nll` decreases to 1.788.

## SFT on Tulu3 dataset

```bash
python -m tinker_cookbook.recipes.chat_sl.train
    model_name=Qwen/Qwen3-8B-Base \
    dataset=tulu3 \
    learning_rate=5e-4 \
    batch_size=128 \
    lora_rank=64 \
    eval_every=500 \
    save_every=500 \
    wandb_project=cookbook_sl
```

After 1740 steps of training, `test/nll` decreases to 0.50.
Performance can be further improved by training longer with a higher `lora_rank` and lower `batch_size`.

## Adding your own dataset

The base classes in [tinker_cookbook/supervised/data.py](../../supervised/data.py) support loading new data in the following way:
- `SupervisedDatasetFromHFDataset` loads dataset on huggingface hub with a postprocessing function
- `StreamingSupervisedDatasetFromHFDataset` works similarly, but supports streaming
- `FromConversationFileBuilder` supports data loading from a JSONL file
