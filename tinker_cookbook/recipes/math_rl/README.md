# Using Reinforcement Learning to Solve Math Problems

Math problems have been the most active testbed for RL with LLMs. This recipe collects environments and grading functions that allow you to test on several popular math datasets.

## Installation

```bash
uv pip install 'tinker-cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

## RL on arithmetic.

Trivial, but runs fast enough that you can see it learn. Reward should go from 0.66 to 1 in the first few steps.

```bash
python -m tinker_cookbook.recipes.math_rl.train model_name="meta-llama/Llama-3.2-1B" group_size=4 groups_per_batch=100 learning_rate=1e-4
```

## RL on MATH dataset.

```bash
python -m tinker_cookbook.recipes.math_rl.train env=math model_name="Qwen/Qwen3-8B" group_size=16 groups_per_batch=64 learning_rate=2e-5 max_tokens=512
```

After 180 steps of training, we observe `"test/env/all/correct": 0.767578125`, which is logged to `/tmp/tinker-examples/math_rl/math-Qwen_Qwen3-8B-32rank-2e-05lr-${DATE}/metrics.jsonl`.

```
<|im_start|>user
The numbers 2, 3, 5, 7, 11, 13 are arranged in a multiplication table, with three along the top and the other three down the left.  The multiplication table is completed a
nd the sum of the nine entries is tabulated.  What is the largest possible sum of the nine entries?

\[
\begin{array}{c||c|c|c|}
\times & a & b & c \\ \hline \hline
d & & & \\ \hline
e & & & \\ \hline
f & & & \\ \hline
\end{array}
\] Write your answer in \boxed{} format.<|im_end|>
<|im_start|>assistant
<think>
Okay, so we have to arrange the primes 2,3,5,7,11,13 into a 3x3 multiplication table. The sum of all entries is (a+b+c)*(d+e+f) - because the sum of the products is the pr
oduct of the sums. To maximize the total sum, we need to maximize the product of the sums of the two groups. So, split the numbers into two groups of three, one group for
the top row (a,b,c) and one for the left column (d,e,f). The product (sum1)*(sum2) is maximized when the sums are as close as possible? Or maybe when one is as large as po
ssible and the other as small? Wait, for a fixed sum, the product is maximized when they're equal. But since the total sum of all numbers is 2+3+5+7+11+13= 41. So sum1 + s
um2 =41. To maximize sum1*sum2, we need sum1 and sum2 as close as possible to 41/2=20.5. So find two subsets of three numbers that add up to around 20.5.

Let me try combinations. Let's see:

Possible sums for three numbers:

- 2+3+13=18

- 2+5+13=20
- 2+7+11=20

- 3+5+13=21

- 3+7+11=21

- 5+7+11=23

So the best possible is sum1=20 and sum2=21, product 420. So the maximum sum is 420. \boxed{420}</think>

\boxed{420}<|im_end|>
```

## Choosing a loss function

The `loss_fn` flag selects which policy-gradient loss to use. The default is `importance_sampling` (REINFORCE with IS correction). You can switch to `ppo`, `cispo`, or `dro` to compare training stability and final performance on the same task.

### CISPO (Conservative Importance Sampling PPO)

CISPO clips the importance-sampling ratio but applies it as a weight on the log-probability rather than on the advantage. This can be more stable than PPO when the policy drifts far off-policy.

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    env=arithmetic loss_fn=cispo \
    model_name="meta-llama/Llama-3.2-1B" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

You can tune the clipping range via `loss_fn_config`:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    env=arithmetic loss_fn=cispo \
    'loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}' \
    model_name="meta-llama/Llama-3.2-1B" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

### DRO (Distributionally Robust Optimization)

DRO optimizes for worst-case performance across the batch, which can improve robustness on harder examples at the cost of slower average improvement.

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    env=arithmetic loss_fn=dro \
    model_name="meta-llama/Llama-3.2-1B" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

### Comparing loss functions on MATH

To compare loss functions on a harder task, run the same configuration with different `loss_fn` values:

```bash
for loss in importance_sampling ppo cispo dro; do
    python -m tinker_cookbook.recipes.math_rl.train \
        env=math loss_fn=$loss \
        model_name="Qwen/Qwen3-8B" \
        group_size=16 groups_per_batch=64 learning_rate=2e-5 max_tokens=512 \
        wandb_project=math-loss-comparison wandb_name="math-$loss"
done
```

Compare the `test/env/all/correct` metric across runs to see how each loss affects convergence speed and final accuracy. As a rule of thumb:
- Start with `importance_sampling` (unbounded ratio, fastest convergence on-policy)
- Try `ppo` or `cispo` if you see instability from large policy updates
- Try `dro` if you want to improve worst-case performance across problem types

# RL on GSM8K

```bash
python -m tinker_cookbook.recipes.math_rl.train env=gsm8k model_name="meta-llama/Llama-3.1-8B-Instruct" group_size=64 groups_per_batch=32 learning_rate=8e-5 max_tokens=1024
```

Generally, you should observe that training reward goes above 0.8 within a few steps. After 220 steps of training, we achieve `"test/env/all/correct": 0.9090`. A smaller group_size (8) and larger `groups_per_batch` (64) will achieve `0.8824` accuracy in around a quarter of the time.

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
