## RL on arithmetic.

Trivial, but runs fast enough that you can see it learn. Reward should go from 0.66 to 1 in the first few steps.

```bash
python -m recipes.math_rl.train model_name="meta-llama/Llama-3.2-1B" group_size=4 groups_per_batch=100 learning_rate=1e-4
```

## RL on math.

```bash
python -m recipes.math_rl.train env=math model_name="Qwen/Qwen3-8B" group_size=16 groups_per_batch=64 learning_rate=2e-5 max_tokens=512
```
