## SFT on tulu3 dataset.

```bash
model_name=Qwen/Qwen2.5-VL-7B-Instruct
python recipes.chat_sft.train_cli \ # TODO(tianyi): this needs to be updated
    log_path=/tmp/tinker-examples/tulu-v3-sft \
    model_name=$model_name \
    dataset=tulu3
```

## SFT to train a pairwise preference model

```bash
model_name=meta-llama/Llama-3.1-8B-Instruct
python recipes.chat_sft.train_cli dataset=hhh model_name=$model_name learning_rate=4e-4
```
