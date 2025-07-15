## SFT on tulu3 dataset.

```bash
model_name=Qwen/Qwen2.5-VL-7B-Instruct
uv run python -m tinker_cookbook.supervised.train_cli \
    log_relpath=tinker-examples/tulu-v3-sft \
    model_name=$model_name \
    dataset=tulu3
```

## SFT to train a pairwise preference model

```bash
model_name=meta-llama/Llama-3.1-8B-Instruct
uv run python -m tinker_cookbook.supervised.train_cli dataset=hhh model_name=$model_name learning_rate=4e-4
```

I found that learning rate with a sweep.

## RL on arithmetic.

Trivial, but runs fast enough that you can see it learn. Reward should go from 0.66 to 1 in the first few steps.

```bash
model_name=meta-llama/Llama-3.2-1B
uv run python -m tinker_cookbook.rl.train_cli \
    log_relpath=tinker-examples/arithmetic-rl \
    model_name=$model_name \
    env=arithmetic \
    group_size=4 \
    groups_per_batch=100 \
    learning_rate=1e-4 \
    max_tokens=5
```

## RL on math.

```bash
model_name=Qwen/Qwen2.5-VL-7B-Instruct
uv run python -m tinker_cookbook.rl.train_cli \
    log_relpath=tinker-examples/math-rl \
    model_name=$model_name \
    env=math \
    group_size=16 \
    groups_per_batch=64 \
    learning_rate=2e-5 \
    max_tokens=512
```

## RL on a reward model

```bash
uv run python -m tinker_cookbook.rl.train_cli \
    env=hhh \
    log_relpath=tinker-examples/hhh-rl \
    groups_per_batch=256 \
    group_size=4 \
    max_tokens=400 \
    learning_rate=2e-5
```
## RL on twenty questions.

```bash
model_name=meta-llama/Llama-3.1-8B-Instruct
uv run python -m tinker_cookbook.rl.train_cli \
    log_relpath=tinker-examples/twenty-questions-rl \
    model_name=$model_name \
    env=twenty_questions \
    learning_rate=1e-4 \
    max_tokens=20
```