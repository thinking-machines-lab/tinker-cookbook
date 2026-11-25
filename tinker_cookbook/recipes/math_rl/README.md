# Using Reinforcement Learning to Solve Math Problems

Math problems have been the most active testbed for RL with LLMs. This recipe collects environments and grading functions that allow you to test on several popular math datasets.

## Installation

```bash
uv pip install 'tinker-cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

## RL on arithmetic.

Trivial, but runs fast enough that you can see it learn. Reward should go from 0.66 to 1 in the first few steps.

```bash
python -m tinker_cookbook.recipes.math_rl.train model_name="Qwen/Qwen3.5-9B-Base" group_size=4 groups_per_batch=100 learning_rate=1e-4
```

## RL on MATH dataset.

```bash
python -m tinker_cookbook.recipes.math_rl.train env=math model_name="Qwen/Qwen3.5-9B" group_size=16 groups_per_batch=64 learning_rate=2e-5 max_tokens=512 loss_fn=importance_sampling
```

After 180 steps of training, we observe `"test/env/all/correct": 0.838`, which is logged to `~/tinker-runs/math_rl/math-Qwen-Qwen3.5-9B-32rank-2e-05lr-16group-64batch-importance_sampling-seed0-${DATE}/metrics.jsonl`.

```
<|im_start|>user
How many r's are in strawberry? Write your answer in \boxed{} format.<|im_end|>
<|im_start|>assistant
Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \boxed{3}<|im_end|>
<|im_start|>user
Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$ Write your answer in \boxed{} format.<|im_end|>
<|im_start|>assistant
<think>
The point (0,3): x=0, y=3. r = sqrt(0^2+3^2)=3. \theta = arctan(y/x) but x=0, y>0 so \theta=\pi/2. So (3, \pi/2). Format: \boxed{(3,\frac{\pi}{2})}.
</think>

\boxed{(3,\frac{\pi}{2})}<|im_end|>
```

## Choosing a loss function

The `loss_fn` flag selects which policy-gradient loss to use. The default is `importance_sampling` (REINFORCE with IS correction). The examples below show how to select `cispo` and `dro`.

### CISPO (Conservative Importance Sampling PPO)

CISPO clips the importance-sampling ratio but applies it as a weight on the log-probability rather than on the advantage. This can be more stable than PPO when the policy drifts far off-policy.

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    env=arithmetic loss_fn=cispo \
    model_name="meta-llama/Llama-3.2-1B" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

CISPO defaults to a one-sided clip (`0.0`, `4.0`). You can set that default explicitly via `loss_fn_config`:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    env=arithmetic loss_fn=cispo \
    'loss_fn_config={"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}' \
    model_name="meta-llama/Llama-3.2-1B" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

### DRO (Distributionally Robust Optimization)

DRO is an off-policy and offline RL objective with a quadratic penalty on the log-ratio between the training and sampling policies. It requires a client-side soft-advantage formulation. The `math_rl` recipe currently uses group-mean-centered advantages, so the command below illustrates the API only; it is not a validated DRO training configuration.

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    env=arithmetic loss_fn=dro \
    model_name="meta-llama/Llama-3.2-1B" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

# RL on GSM8K

```bash
python -m tinker_cookbook.recipes.math_rl.train env=gsm8k model_name="Qwen/Qwen3.5-9B" group_size=64 groups_per_batch=32 learning_rate=8e-5 max_tokens=1024
```

Generally, you should observe that training reward goes above 0.8 within a few steps. After 220 steps of training, we achieve `"test/env/all/correct": 0.9621`. A smaller `group_size` (8) and larger `groups_per_batch` (64) reaches `"test/env/all/correct": 0.9447`.

Here is an example chain of thought,
```
A plane takes off at 6:00 a.m. and flies for 4 hours from New York City to Chicago. The plane stays at the port in Chicago for 1 hour and then departs for Miami. If the aircraft took three times as many hours to fly to Miami than it took to fly from New York to Chicago, calculate the total time to travel from New York to Miami. Write your answer in \boxed{} format.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Let's break it down step by step:
1. The plane flies from New York City to Chicago for 4 hours. This duration is given.
2. The plane stays at the port in Chicago for 1 hour.
3. The time it takes to fly from Chicago to Miami is three times the time it took to fly from New York to Chicago, which is 4 hours. So, the time to fly from Chicago to Miami is 3 * 4 = 12 hours.
Now, let's calculate the total time:
* Flight from New York City to Chicago: 4 hours
* Stay at the port in Chicago: 1 hour
* Flight from Chicago to Miami: 12 hours
Total time = 4 + 1 + 12 = 17 hours
So, the total time to travel from New York to Miami is 17 hours.
\boxed{17}<|eot_id|>
```
