# TINKER DOCUMENTATION
This file contains the complete Tinker documentation and SDK reference.

## Table of Contents

1. Documentation (MDX files)
2. Type Definitions (from tinker.types)

---

# PART 1: DOCUMENTATION

## File: index.mdx

# Tinker: a training API for researchers and developers

Tinker lets you focus on what matters in LLM fine-tuning – your data and algorithms – while we handle the heavy lifting of distributed training.

You write a simple loop that runs on your CPU-only machine, including the data or environment and the loss function. We figure out how to make the training work on a bunch of GPUs, doing the exact computation you specified, efficiently. To change the model you're working with, you only need to change a single string in your code.

Tinker gives you full control over the training loop and all the algorithmic details. It's not a magic black box that makes fine-tuning "easy". It's a clean abstraction that shields you from the complexity of distributed training while preserving your control.

Here's how the division of responsibilities works in practice:

| **You focus on** | **You write** | **We handle** |
|---|---|---|
| 📊 **Datasets and RL environments**<br />Your custom training data | 💻 **Simple Python script**<br />Runs on your CPU | ⚡ **Efficient distributed training of large models**<br />Llama, Qwen, and more |
| 🎯 **Training logic**<br />Your loss functions, training loop, and evals | 🔧 **API calls**<br />`forward_backward()`<br />`optim_step()`<br />`sample()` | 🛡️ **Reliability**<br />Hardware failures handled transparently |

## Features

What the Tinker service currently supports:

- Tinker lets you fine-tune open-weight models like the Qwen and Llama series, including large mixture-of-experts models like Qwen3-235B-A22B.
- Tinker implements low-rank adaptation (LoRA) fine-tuning, not full fine-tuning. However, we believe that LoRA gives the same performance as full fine-tuning for many important use cases, especially in RL (see [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)).
- You can download the weights of your trained model to use outside of Tinker, for example with your inference provider of choice.

## A quick look at functionality

Tinker's main functionality is contained in a few key functions:

- `forward_backward`: feed in your data and loss function, and we'll compute and accumulate the gradients for you.
- `optim_step`: update your model using the accumulated gradients
- `sample`: Generate outputs from your trained model
- other functions for saving and loading weights and optimizer state

## What's next?

Some features we expect to support in the future:

- Image input for applicable models
- Full fine-tuning


---

## File: losses.mdx

# Loss functions in Tinker

For most use cases, you can use the Tinker API's built-in loss functions by passing in a string identifier to `forward_backward`, which supports cross entropy and policy gradient objectives. When you need more control, `forward_backward_custom` enables arbitrary differentiable loss functions at the cost of an additional forward pass; we explain both approaches in this doc.

When you call `forward_backward`, you specify a loss function using a string that selects from a predetermined set of options, comprising the most common losses used for language model training.
- **Input:** `forward_backward` expects a certain set of input tensors, passed in via `datum.loss_fn_inputs`, which is a dict mapping `str` to either a numpy or torch tensor
- **Output:** `forward_backward` returns a `ForwardBackwardOutput`, which has a set of output tensors in `fwd_bwd_result.loss_fn_outputs`

For an example of using `forward_backward`, see `rl/train.py` in the Cookbook:
```python
async def forward_backward(
    training_client: tinker.TrainingClient,
    batch_d: List[tinker.Datum],
) -> List[torch.Tensor]:
    """Accumulate gradients on a minibatch of data"""
    fwd_bwd_future = await training_client.forward_backward_async(
        list(map(remove_mask, batch_d)), loss_fn="importance_sampling"
    )
    fwd_bwd_result = await fwd_bwd_future.result_async()

    # Extract training logprobs from loss_fn_outputs
    training_logprobs_D: list[torch.Tensor] = []
    for output in fwd_bwd_result.loss_fn_outputs:
        training_logprobs = output["logprobs"].to_torch()
        training_logprobs_D.append(training_logprobs)

    return training_logprobs_D
```

## Basic loss functions

Currently, the Tinker API supports `cross_entropy` (for supervised learning), `importance_sampling` (for RL), and `ppo` (for RL).

All tensors below have shape `(N,)` where `N` is `model_input.length`. They can be provided as `numpy.ndarray` or `torch.Tensor`, and the return values will use the same tensor type.

### Supervised learning: `cross_entropy`

For SL, we implement the standard cross-entropy loss (i.e., negative-log-likelihood), which optimizes the policy $p_\theta$ to maximize the log-probability of the tokens $x$:

$$
\mathcal{L(\theta)} = -\mathbb{E}_x[\log p_\theta(x)]
$$

In practice, this looks like `-(weights * logp(target_tokens)).sum()`, where `weights` is either 0 or 1, typically generated from `renderers.build_supervised_example` (i.e., to specify the desired assistant turns to train on).

- **Input tensors:**
  - `target_tokens: array[(N,), int]` - Target token IDs
  - `weights: array[(N,), float]` - Token-level loss weights (typically from the renderer)
- **Output tensors:**
  - `logprobs: array[(N,), float]` - Log probabilities of predicted tokens
- **Output diagnostics:**
  - `loss:sum` (scalar) - Sum of weighted cross-entropy losses

### Policy gradient: `importance_sampling`

For RL, we implement a common variant of the policy gradient objective, used in practical settings where the *learner policy* $p$ may differ from the *sampling policy* $q$, which is common due to e.g. [non-determinism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). To remove the bias caused by this difference, we can use a modified "importance sampling" objective:

$$
\nabla \mathbb{E}_{x\sim p_\theta}\bigl[r(x) \bigr] = \mathbb{E}_{x\sim q}\Bigl[r(x) \cdot \frac{\nabla p_\theta(x)}{q(x)}\Bigr]
$$

which yields the correct expected reward in expectation. This is implemented as `(exp(new_logprobs - logprobs) * advantages).sum()`, where `advantages` may additionally subtract a baseline from the rewards. Note this only works in the bandit setting, which is common in both RLHF and RLVR setups.

- **Input tensors:**
  - `target_tokens: array[(N,), int]` - Target token IDs (from the sampler $q$)
  - `logprobs: array[(N,), float]` - Reference log probabilities $q$ for the target tokens
  - `advantages: array[(N,), float]` - Advantage values for RL
- **Output tensors:**
  - `logprobs: array[(N,), float]` - Log probabilities $p$ for the target tokens
- **Output diagnostics:**
  - `loss:sum` (scalar) - Sum of importance-weighted policy gradient losses

**Addendum:** Let's consider naively applying the policy gradient objective when $q \neq p$:

$$
\begin{align*}
\mathbb{E}_{x\sim q}\bigr[ r(x) \cdot \nabla \log p_\theta(x) \bigl] &= \sum_x q(x) r(x) \cdot \nabla \log p_\theta(x) \\
    &= \sum_x q(x) (r(x) - \bar{r}) \nabla \log p_\theta(x) + \sum_x q(x) \bar{r} \cdot \nabla \log p_\theta(x) \\
    &= \mathbb{E}_{x\sim q}\bigl[(r(x) - \bar{r}) \nabla \log p_\theta(x)\bigr] - \bar{r} \cdot \nabla KL(q \Vert p)
\end{align*}
$$

where $\bar{r} = \sum_x q(x) r(x)$, effectively an average-reward baseline.

- The first expectation term resembles a pseudo-policy gradient, increasing the log-likelihood of tokens $x$ which achieve higher-than-average rewards. (It is not an actual policy gradient, because $q \neq p$.)
- The second KL term is effectively a bias term which can destablize RL optimization. This bias increases as either the divergence $KL(q \Vert p)$ grows, or as the average reward $\bar{r}$ shifts.

## Flexible loss functions: `forward_backward_custom`

For use-cases outside of the above, we've provided the more flexible (but slower) methods `forward_backward_custom` and `forward_backward_custom_async` to compute a more general class of loss functions.

### Usage

Here's a simple example of a custom loss function:

```python
def logprob_squared_loss(data: list[Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    loss = (logprobs ** 2).sum()
    return loss, {"logprob_squared_loss": loss.item()}
```

You can call this loss function with `forward_backward_custom` like:

```python
loss, metrics = training_client.forward_backward_custom(data, logprob_squared_loss)
```

You can also define loss functions which operate on multiple sequences at a time. For example, (although practically useless), a loss function that computes the variance across the sequences can be implemented as:

```python
def variance_loss(data: list[Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    flat_logprobs = torch.cat(logprobs)
    variance = torch.var(flat_logprobs)
    return variance, {"variance_loss": variance.item()}
```

A more practical use case would be to compute a Bradley-Terry loss on pairwise comparison data -- a classic approach in RL from human feedback, as introduced/popularized by [Learning to Summarize](https://arxiv.org/abs/2009.01325). Similarly, we can also implement [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), which also computes a loss involving pairs of sequences; see the [DPO guide](/preferences/dpo-guide) for more details.

If you're using a custom loss function that you think is generally useful, please let us know, and we'll add it to the list of built-in loss functions.

We detail the `async` version of methods in the [Async and Futures](./async) of these docs.

### How `forward_backward_custom` works

---

## File: supervised-learning.mdx

import { CookbookLink } from '../components/CookbookLink'

# Supervised learning

In general, supervised learning means learning an input-output mapping from labeled data. In the context of this documentation, it'll mean that we're minimizing a weighted cross-entropy loss on token sequences, i.e., maximizing the log-probability of the specififed target tokens.

There are a few ways that supervised learning (abbreviated as SL) is commonly used in LLM fine-tuning pipelines:

- *Instruction tuning*: as the first step in post-training pipelines, applied to the *base model*, i.e., *raw pretrained model*. Typically, we do SL on a high-quality dataset that demonstrates the correct format and style, while boosting the model's skills of reasoning and instruction-following.
- *Context distillation* / *prompt distillation*: let's say we have a generic model that can do chat / instruction following / reasoning, but we want to adjust how it behaves in a certain scenario. We can add some instructions to the system message of our model. However, the system message might grow impractically long and start ignoring some of its instructions. So it's often better to create a supervised dataset on a narrow prompt distribution, with a shorter set of instructions that that are targeted at these prompts.

We'll cover these use cases in the documentation and cookbook code.

The library code implementing supervised learning can be found in the <CookbookLink path="tinker_cookbook/supervised">`supervised`</CookbookLink> directory.



---

## File: preferences.mdx

# Preferences

This overview page will cover learning from preferences.

*This is a placeholder page that needs to be written.*


---

## File: lora-primer.mdx

# LoRA Primer

Tinker supports [LoRA fine-tuning](https://arxiv.org/abs/2106.09685), which adjusts a small number of parameters, rather than full fine-tuning, which adjusts all of the parameters of the original model. Our current understanding is that LoRA has equivalent performance to full fine-tuning when doing RL or doing SL on small datasets, while it has worse performance on larger datasets. In more detail:

- For supervised fine-tuning on a small dataset, with fewer completion tokens than the number of LoRA parameters, then LoRA gives roughly the same performance as full fine-tuning.
- For supervised fine-tuning on a large dataset (like continued pre-training, or datasets with billions of tokens), LoRA will significantly underperform full fine-tuning.
- For reinforcement learning fine-tuning, LoRA performs roughly the same as full fine-tuning.
- While it has deficits when fine-tuning on larger datasets, LoRA may also typically better preserves the capabilities of the base model, according to [this study](https://arxiv.org/abs/2405.09673).

See the Connectionism blog post on [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) for more details and experimental results.

# Hyperparameters

The learning rate (LR) is usually the most important hyperparameter in your ML experiments. LoRA requires a much larger LR than full fine-tuning -- this is a mistkae people often make when porting their code to use LoRA, leading them to conclude that LoRA works poorly. Depending on the model size, the LoRA learning rate might be 20-100 times larger than the full fine-tuning learning rate.

We've provided a utility that calculates the factor you should scale the full fine-tuning LR by to get the equivalent LoRA LR.

```python
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr

model_name = "meta-llama/Llama-3.1-8B"
print(get_lora_lr_over_full_finetune_lr(model_name))
```

Note that for `Llama-3.2-1B`, the factor is 32, while for `Llama-3.1-70B`, the factor is 128.

# What is LoRA exactly?

LoRA is short for Low-Rank Adaptation. Given that the original model has a weight matrix $W$, we replace it with a new weight matrix $W'=W + BA$, where $B$ and $A$ are low-rank matrices. If $W$ is an $n \times n$ matrix, then $B$ and $A$ are $n \times r$ and $r \times n$ matrices, respectively, where $r$ is the rank of the low-rank approximation. The default $r$ used by tinker is $32$.

The fact that LoRA uses a low-rank approximation of weight matrices is not terribly important -- we prefer to think of LoRA as just a random projection of the parameter space that happens to be efficient to implement. When training with RL or small SL datasets, we are only learning a small amount of information, and this reduced set of parameters is more than enough.


# What rank to use?

The default rank used by tinker is $32$. However, if you're doing SL on a large dataset, you should use a larger rank. The largest rank currently supported is $128$, so you should use this rank if you are doing SL on a large dataset. Supporting ranks $512$ and higher is on the roadmap.

For supervised learning, as a very rough approximation, LoRA will give good results as long as the number of LoRA parameters is at least as large as the number of completion tokens (i.e., weight=1 tokens). You can calculate the number of LoRA parameters with the following utility:

```python
from tinker_cookbook.hyperparam_utils import get_lora_param_count

model_name = "meta-llama/Llama-3.1-8B"
print(get_lora_param_count(model_name, lora_rank=32))
```

For reinforcement learning, we've found that small ranks give equivalent performance to larger ranks and full fine-tuning.

Note that conveniently, the optimal learning rate does *not* depend on the LoRA rank. In fact, you can verify that if you train with SL on different ranks (but with the same LR), you'll get exactly the same learning curves for the first few steps of training.

---

## File: evals.mdx

import { Callout } from 'nextra/components'
import { CookbookLink } from '../components/CookbookLink'

# Evaluations

Our training scripts will print out training and test loss. Two common workflows for evaluations are to do inline evals during training and to do offline evals on various checkpoints from a run.

## Inline Evals

You can add inline evaluations to your training runs by configuring evaluator builders in advance for both supervised fine-tuning and RL training jobs.

### Supervised Fine-Tuning (`supervised.train`)
Add one or both of the following to your config:

- **`evaluator_builders: list[EvaluatorBuilder]`** - Runs evaluations every `eval_every` steps
- **`infrequent_evaluator_builders: list[EvaluatorBuilder]`** - Runs evaluations every `infrequent_eval_every` steps

### RL Training (`rl.train`)

Add the following to your config:

- **`evaluator_builders: list[SamplingClientEvaluator]`** - Runs evaluations every `eval_every` steps

For implementation guidance and a detailed example, see <CookbookLink path="tinker_cookbook/eval/evaluators.py">here</CookbookLink> and
 <CookbookLink path="tinker_cookbook/eval/inspect_evaluators.py">here</CookbookLink> respectively.


## Offline evals

We support and recommend several ways for creating and running your offline evaluations on your model checkpoints.

### Running Standard Evaluations with Inspect AI.

We support running many of the standard cited evaluations using the [Inspect AI library](https://github.com/UKGovernmentBEIS/inspect_ai).

We have provided a <CookbookLink path="tinker_cookbook/eval/run_inspect_evals.py">script</CookbookLink> to evaluate models using Tinker's internal sampling functionality as shown below.

```bash
MODEL_PATH=tinker://FIXME # YOUR MODEL PATH HERE
python -m tinker_cookbook.eval.run_inspect_evals \
    model_path=$MODEL_PATH \
    model_name=MODEL_NAME \ # YOUR MODEL_NAME HERE
    tasks=inspect_evals/ifeval,inspect_evals/mmlu_0_shot \
    renderer_name=RENDERER_NAME # YOUR RENDERER_NAME HERE
```

Click [here](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/evals/listing.yml) to view additional supported evaluations.

### Creating your own Sampling Evaluations

We recommend two ways to create your own evaluations:
- creating your own tasks with Inspect AI and running like above
- creating your own SamplingClientEvaluator

#### Create tasks with Inspect AI

In addition to passing in standard evaluations, you can create your own tasks using inspect ai as detailed [here](https://inspect.aisi.org.uk/tasks.html).

Here is a toy example of how to create an evaluation with an LLM-as-a-judge where we use a model produced by tinker as a grader.

```python
import tinker
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

QA_DATASET = MemoryDataset(
    name="qa_dataset",
    samples=[
        Sample(
            input="What is the capital of France?",
            target="Paris",
        ),
        Sample(
            input="What is the capital of Italy?",
            target="Rome",
        ),
    ],
)

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(
    base_model="meta-llama/Llama-3.1-8B-Instruct"
)

api = InspectAPIFromTinkerSampling(
    renderer_name="llama3",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    sampling_client=sampling_client,
    verbose=False,
)

GRADER_MODEL = InspectAIModel(api=api, config=InspectAIGenerateConfig())


@task
def example_lm_as_judge() -> Task:
    """
    Example task using LLM-as-a-judge scoring.

    Note: The grader model defaults to the model being evaluated.
    To use a different grader model, specify it with --model-grader when using inspect directly.
    """
    return Task(
        name="llm_as_judge",
        dataset=QA_DATASET,
        solver=generate(),
        scorer=model_graded_qa(
            instructions="Grade strictly against the target text as general answer key and rubric. "
            "Respond 'GRADE: C' if correct or 'GRADE: I' otherwise.",
            partial_credit=False,
            # model parameter is optional - if not specified, uses the model being evaluated
            model=GRADER_MODEL,
        ),
    )
```

Inspect also natively supports replacing our `GRADER_MODEL` with any openai-chat-completion style api (e.g. openrouter).

#### Create your own SamplingClientEvaluator

Alternatively, you can create your own SamplingClientEvaluator class instead of using Inspect AI. This is a lower
level abstraction than the above with finer-grain control over running your evaluations.

We expose this to interace to allow users more control over their datasets and metrics. To illustrate, see this
<CookbookLink path="tinker_cookbook/eval/custom_evaluators.py">custom evaluators</CookbookLink> example of how one might create their own complex SamplingClientEvaluator.

For a more illustrative toy instructive example see below.

```python
from typing import Any, Callable

import tinker
from tinker import types

from tinker_cookbook import renderers
from tinker_cookbook.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer

class CustomEvaluator(SamplingClientEvaluator):
    """
    A toy SamplingClientEvaluator that runs a custom evaluation and returns its metrics.
    """

    def __init__(
        self,
        dataset: Any,
        grader_fn: Callable[[str, str], bool],
        model_name: str,
        renderer_name: str,
    ):
        """
        Initialize the CustomEvaluator.
        Args:
            config: Configuration object containing all evaluation parameters
        """
        self.dataset = dataset
        self.grader_fn = grader_fn

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run custom evaluation on the given sampling client and return metrics.
        Args:
            sampling_client: The sampling client to evaluate
        Returns:
            Dictionary of metrics from inspect evaluation
        """

        metrics = {}

        num_examples = len(self.dataset)
        num_correct = 0

        sampling_params = types.SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        for datum in self.dataset:
            model_input: types.ModelInput = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=datum["input"])]
            )
            # Generate response
            r: types.SampleResponse = await sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )
            tokens: list[int] = r.sequences[0].tokens
            response: renderers.Message = self.renderer.parse_response(tokens)[0]
            if self.grader_fn(response["content"], datum["output"]):
                num_correct += 1

        metrics["accuracy"] = num_correct / num_examples
        return metrics
```

Here is an example of how we can use the above CustomEvaluator on a toy dataset and grader.


```python
QA_DATASET = [
    {"input": "What is the capital of France?", "output": "Paris"},
    {"input": "What is the capital of Germany?", "output": "Berlin"},
    {"input": "What is the capital of Italy?", "output": "Rome"},
]

def grader_fn(response: str, target: str) -> bool:
    return target.lower() in response.lower()

evaluator = CustomEvaluator(
    dataset=QA_DATASET,
    grader_fn=grader_fn,
    renderer_name="llama3",
    model_name="meta-llama/Llama-3.1-8B-Instruct",

)

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="meta-llama/Llama-3.1-8B-Instruct")

async def main():
    result = await evaluator(sampling_client)
    print(result)

asyncio.run(main())
```

---

## File: dev-tips.mdx

# Developer Tips

## AI-assisted development

We've provided a single-file version of the documentation that can be fed to LLMs for development: see [llms.txt](/llms.txt) and [llms-full.txt](/llms-full.txt).


---

## File: async.mdx

# Async and Futures

## Sync and Async APIs

Every method in the Tinker Python library has both a synchronous (sync) and an asynchronous (async) version. The async variants end with `_async`:

| **Client** | **Sync method** | **Async method** |
|---|---|---|
| `ServiceClient` | `create_lora_training_client()` | `create_lora_training_client_async()` |
| `TrainingClient` | `forward()` | `forward_async()` |
| `SamplingClient` | `sample()` | `sample_async()` |
| `RestClient` | `list_training_run_ids()` | `list_training_run_ids_async()` |

Tinker's `async` functionality requires an `asyncio` event loop, which you typically run like `asyncio.run(main())`.

**When to use each:**

- **Async:** Best for high-performance workflows where you need concurrency, especially when waiting on multiple network calls.
- **Sync:** Simpler for scripts and learning examples. Easier to reason about but blocks on each operation.

The Tinker Cookbook generally uses `async` for implementations where performance is critical and sync for pedagogical examples.

## Understanding Futures

Most Tinker API methods are **non-blocking**, but may take a little while to run. They return immediately with a `Future` object that acknowledges that your request has been submitted. To get the actual result, you must explicitly wait:

**Sync Python:**
```python
result = client.forward_backward_async(data, loss_fn)
result = result.result_async() # Blocks until complete
```

**Async Python (note the double await):**
```python
result = await client.forward_backward_async(data, loss_fn)
result = await result.result_async()
```

After the first `await`, you're guaranteed that the request has been submitted, which ensures that it'll be ordered correctly relative to other requests. The second `await` waits for the actual computation to finish and returns the numerical outputs. For operations like `forward_backward`, the second `await` also guarantees that operation has been applied to the model---for `forward_backward`, this means that the gradients have been accumulated in the model's optimizer state.

## Performance tips: overlap requests

For best performance, you should aim to submit your next request while the current one is running. Doing so is more important with Tinker than with other training systems because Tinker training runs on discrete [clock cycles](./under-the-hood#clock-cycles) (~10 seconds each). If you don't have a request queued when a cycle starts, you'll miss that cycle entirely.

**Example pattern for overlapping requests:**
```python
# Submit first request
future1 = await client.forward_backward_async(batch1, loss_fn)

# Submit second request immediately (don't wait for first to finish)
future2 = await client.forward_backward_async(batch2, loss_fn)

# Now retrieve results
result1 = await future1.result_async()
result2 = await future2.result_async()
```


---

## File: overview-building.mdx

# Overview (Tinker Cookbook)

The next sections provide a variety of guides for how to use the Tinker API for research and applications.

We expect people to use Tinker in a few different ways:

1. You want to define datasets and environments and plug them into existing training code from the Tinker Cookbook.
2. You want to write your own training loops from scratch, starting with the basics.
3. You want to understand the classes and other concepts in Tinker Cookbook so you can extend them to add new functionality.

Different parts of the docs will be tailored to these different approaches.

We'll start with a couple of general pages that'll be relevant to almost all of the use cases:

- [Rendering to Tokens](./rendering.mdx) -- how we convert from a conversation data structure to a list of tokens (a.k.a. chat templates).
- [LoRA Primer](./lora-primer.mdx) -- basic background of LoRA, and how to choose hyperparameters. For most fine-tuning applications, LoRA will give results that are roughly the same as full fine-tuning, however, you need to use different learning rates.



---

## File: save-load.mdx

# Saving and loading weights and optimizer state

During training, you'll need to save checkpoints for two main purposes: *sampling* (to test your model) and *resuming training* (to continue from where you left off). The `TrainingClient` provides three methods to handle these cases:

1. `save_weights_for_sampler()`: saves a copy of the model weights that can be used for sampling.
2. `save_state()`: saves the weights and the optimizer state. You can fully resume training from this checkpoint.
3. `load_state()`: load the weights and the optimizer state. You can fully resume training from this checkpoint.

Note that (1) is faster and requires less storage space than (2).

Both `save_*` functions require a `name` parameter---a string that you can set to identify the checkpoint within the current training run. For example, you can name your checkpoints `"0000"`, `"0001"`, `"step_1000"`, etc.

The return value contains a `path` field, which is a fully-qualified path, which will look something like `tinker://<model_id>/<name>`. This path is persistent and can be loaded later by a new `ServiceClient` or `TrainingClient`.

### Example: Saving for sampling

```python
# Setup
import tinker
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B", rank=32
)

# Save a checkpoint that you can use for sampling
sampling_path = training_client.save_weights_for_sampler(name="0000").result().path

# Create a sampling client with that checkpoint
sampling_client = service_client.create_sampling_client(model_path=sampling_path) #
```

**Shortcut:** Combine these steps with:

```python
sampling_client = training_client.save_weights_and_get_sampling_client(name="0000")
```

### Example: Saving to resume training

Use `save_state()` and `load_state()` when you need to pause and continue training with full optimizer state preferred:

```python
# Save a checkpoint that you can resume from
resume_path = training_client.save_state(name="0010").result().path

# Load that checkpoint
training_client.load_state(resume_path)
```

### When to use `save_state()` and `load_state()`:


- Multi-step training pipelines (e.g. supervised learning followed by reinforcement learning)
- Adjusting hyperparameters or data mid-run
- Recovery from interruptions or failures
- Any scenario where you need to preserve exact optimizer state (momentum, learning rate schedules, etc.)


---

## File: training-sampling.mdx

import { Callout } from 'nextra/components'

# Getting started with training and sampling

In this guide, we'll step you through using the Tinker Python library to do the basic operations needed for training and sampling.
[View the complete Python script →](/quickstart.py.txt)

## Creating the training client

The main object we'll be using is the `TrainingClient`, which corresponds to a fine-tuned model that we can train and sample from.

First, set your Tinker API key environment variable. In the terminal where you'll run Python, or in your `.bashrc`, put `export TINKER_API_KEY=<your key>`.

Then, create a `ServiceInterface`. This lets you find out what base models are available to be fine-tuned.

```python
import tinker
service_client = tinker.ServiceClient()
print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
```
You'll see a list of model names:
```
- meta-llama/Llama-3.1-70B
- meta-llama/Llama-3.1-8B
...
- Qwen/Qwen3-235B-A22B-Instruct-2507
- Qwen/Qwen3-30B-A3B-Base
```
We currently support models from the Qwen3 and Llama3 series. We're going to use Qwen3-30B-A3B-Base (a raw pre-trained model) for these examples. See [Available Models in Tinker](/model-lineup) for the full list.

Now we can create the `TrainingClient`:
```python
base_model = "Qwen/Qwen3-30B-A3B-Base"
training_client = service_client.create_lora_training_client(
    base_model=base_model
)
```
Note that we've specified a *base model*, which is the model we'll initialize from. In this case, it's a raw pre-trained model, but most of the "base models" in Tinker are fine-tuned for chat/instruction-following. You should check the details of the model you're using in their system cards.

## Preparing the training data

Now we can do training updates on the model. This quickstart example won't show best practices for LLM fine-tuning; it's just an API demo. Check out [Rendering](/rendering), [Supervised Fine-tuning](/supervised-learning) and the other Cookbook examples for guidance on how to use Tinker in real applications.

For this model, we'll train a model that can translate words into Pig Latin. The rules for Pig Latin are simple:
- If a word begins with a consonant, move it to the end and add "ay"
- If a word begins with a vowel, just add "way" to the end

Here are some example completions we'd like the model to perform, where the prompt is in green and the model's completion is in red:

<div className="example">
<span className="prompt">English: hello world<br/>
Pig Latin: </span><span className="completion">ello-hay orld-way</span>
</div>

Let's create some training examples and convert them to a format expected by Tinker.

```python
# Create some training examples
examples = [
    {
        "input": "banana split",
        "output": "anana-bay plit-say"
    },
    {
        "input": "quantum physics",
        "output": "uantum-qay ysics-phay"
    },
    {
        "input": "donut shop",
        "output": "onut-day op-shay"
    },
    {
        "input": "pickle jar",
        "output": "ickle-pay ar-jay"
    },
    {
        "input": "space exploration",
        "output": "ace-spay exploration-way"
    },
    {
        "input": "rubber duck",
        "output": "ubber-ray uck-day"
    },
    {
        "input": "coding wizard",
        "output": "oding-cay izard-way"
    },
]

# Convert examples into the format expected by the training client
from tinker import types

# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()

def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the input with Input/Output template
    # For most real use cases, you'll want to use a renderer / chat template,
    # (see later docs) but here, we'll keep it simple.
    prompt = f"English: {example['input']}\nPig Latin:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:] # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]

    # A datum is a single training example for the loss function.
    # It has model_input, which is the input sequence that'll be passed into the LLM,
    # loss_fn_inputs, which is a dictionary of extra inputs used by the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

processed_examples = [process_example(ex, tokenizer) for ex in examples]

# Visualize the first example for debugging purposes
datum0 = processed_examples[0]
print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
print("-" * 50)
for i, (inp, tgt, wgt) in enumerate(zip(datum0.model_input.to_ints(), datum0.loss_fn_inputs['target_tokens'].tolist(), datum0.loss_fn_inputs['weights'].tolist())):
    print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")
```

The visualization of the first example is:

```
Input                Target               Weight
--------------------------------------------------
'English'            ':'                  0.0
':'                  ' I'                 0.0
' I'                 ' love'              0.0
' love'              ' tink'              0.0
' tink'              'ering'              0.0
'ering'              '\n'                 0.0
'\n'                 'P'                  0.0
'P'                  'ig'                 0.0
'ig'                 ' Latin'             0.0
' Latin'             ':'                  0.0
':'                  ' I'                 1.0
' I'                 '-way'               1.0
'-way'               ' o'                 1.0
' o'                 've'                 1.0
've'                 '-l'                 1.0
'-l'                 'ay'                 1.0
'ay'                 ' ink'               1.0
' ink'               'ering'              1.0
'ering'              '-t'                 1.0
'-t'                 'ay'                 1.0
'ay'                 '<|endoftext|>'      1.0
```

## Performing a training update

Now we can use this data to perform a training update. We'll do 6 updates on the same batch of data. (Note that this is not typically a good way to train!)

```python
import numpy as np
for _ in range(6):
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

    # Wait for the results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    # fwdbwd_result contains the logprobs of all the tokens we put in. Now we can compute the weighted
    # average log loss per token.
    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
    print(f"Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")
```

Note that the `forward_backward` and `optim_step` functions immediately return *futures*, which acknowledge that the task has been queued up by the server. For improved speed, we submitted both operations before waiting for the result by calling `result()` on the futures.


## Sampling from the model

Now we can test our model by sampling from it. In this case, we'll translate the phrase "coffee break" into Pig Latin.

```python
# First, create a sampling client. We need to transfer weights
sampling_client = training_client.save_weights_and_get_sampling_client(name='pig-latin-model')

# Now, we can sample from the model.
prompt=types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"]) # Greedy sampling
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()
print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
```

Since sampling is nondeterministic (sadly, even with temperature=0.0, [due to batching](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)), the output will be different each time. You should see something like this:

```
Responses:
0: ' offe-bay eak-bay\n\n'
1: ' offey-coy eak-bray\n\n'
2: ' offecay eakbray\n\n'
3: ' offeec-cay eak-brcay\n\n\n'
4: ' offecay akebay\n\n'
5: ' offee-Cay ake-bay\n\n\n'
6: ' offey-pay eak-bray\n\n'
7: ' offee – cay eak – bray\n\n'
```


---

## File: rendering.mdx

import { CookbookLink } from '../components/CookbookLink'


# Rendering

Rendering is the process of converting list-of-message datatypes into their token representations and is roughly the same as [chat templates](https://huggingface.co/docs/transformers/en/chat_templating); however, the chat template functionality in other libraries is only well-suited for inference, whereas this library provides a more complete interface that handles supervised and reinforcement learning.


## Renderer Class

The Renderer class is the main interface used for rendering and can be found in <CookbookLink path="tinker_cookbook/renderers.py">`renderers.py`</CookbookLink>.

Let's use the following working example of a conversation between a user and an assistant.

```python
messages =[
    {'role': 'system', 'content': 'Answer concisely; at most one sentence per response'},
    {'role': 'user', 'content': 'What is the longest-lived rodent species?'},
    {'role': 'assistant', 'content': 'The naked mole rat, which can live over 30 years.'},
    {'role': 'user', 'content': 'How do they live so long?'},
    {'role': 'assistant', 'content': 'They evolved multiple protective mechanisms including special hyaluronic acid that prevents cancer, extremely stable proteins, and efficient DNA repair systems that work together to prevent aging.'}
]
```

## Generating messages

Our model maps tokens to tokens, but with the renderer, it can map messages to messages. To sample messages from the model, we need to use three methods from the renderer:

- `build_generation_prompt`
- `get_stop_sequences`
- `parse_response`


`build_generation_prompt` converts a conversation into a prompt that we can use to sample from the assistant. This is used during reinforcement learning and at deployment time.

Let's remove the last assistant message and call `build_generation_prompt` to get a prompt that we can use to sample an alternative response from the assistant.

```python
from tinker_cookbook import renderers, tokenizer_utils
tokenizer = tokenizer_utils.get_tokenizer('Qwen/Qwen3-30B-A3B')
renderer = renderers.get_renderer('qwen3', tokenizer)
prompt = renderer.build_generation_prompt(messages[:-1])
print(prompt)
print('-'*10)
print(tokenizer.decode(prompt.to_ints()))
```

First you can see that the prompt is a `ModelInput` object, which is a list of `EncodedTextChunk` objects (but contains different objects in multi-modal data).
```
ModelInput(chunks=[EncodedTextChunk(tokens=[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 8948, 198, 16141, 3529, 285, 974, 26, 518, 1429, 825, 11652, 817, 2033, 151645, 198, 151644, 872, 198, 3838, 374, 279, 22032, 61854, 20589, 306, 9419, 30, 151645, 198, 151644, 77091, 198, 785, 19020, 34651, 11244, 11, 892, 646, 3887, 916, 220, 18, 15, 1635, 13, 151645, 198, 151644, 872, 198, 10234, 30, 151645, 198, 151644, 77091, 198], type='encoded_text')])
----------
<|im_start|>system
Answer concisely; at most one sentence per response<|im_end|>
<|im_start|>user
What is the longest-lived rodent species?<|im_end|>
<|im_start|>assistant
The naked mole rat, which can live over 30 years.<|im_end|>
<|im_start|>user
How do they live so long?<|im_end|>
<|im_start|>assistant

```

Given that we're providing messages as input, we probably want a message output, rather than a token output. For that, we can use `parse_response`.

```python
import tinker
from tinker.types import SamplingParams
service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model='Qwen/Qwen3-30B-A3B')
stop_sequences = renderer.get_stop_sequences()
print(f"Stop sequences: {stop_sequences}")
sampling_params = SamplingParams(max_tokens=100, temperature=0.5, stop=stop_sequences)
output = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
print(f"Sampled tokens: {output.sequences[0].tokens}")
sampled_message, parse_success = renderer.parse_response(output.sequences[0].tokens)
print(f"Sampled message: {sampled_message}")
print(f"Parse success: {parse_success}")
```
We get the following output:
```
Stop sequences: [151645]
Sampled tokens: [45, 7741, 34651, 31410, 614, 4911, 76665, 11, 2670, 264, 7548, 11050, 22077, 1849, 323, 264, 1602, 3347, 40761, 4379, 11, 892, 16792, 311, 862, 57119, 13, 151645]
Sampled message: {'role': 'assistant', 'content': 'Naked mole rats have unique adaptations, including a highly efficient immune system and a very low metabolic rate, which contribute to their longevity.'}
Parse success: True
```

You can see that the there is one stop sequence, 151645, which you can verify is the `<|im_end|>` token. The output is parsed successfully into a message.


## Supervised learning

For supervised learning, along with some other algorithms like [DPO](/preferences/dpo-guide), we need different information from the renderer -- we want to provide a target assistant message, and the renderer needs to tell us which tokens are part of the prompt and completion.

To do this, we can use `build_supervised_example` as follows:

```python
tokens, weights = renderer.build_supervised_example(messages)

from tinker_cookbook.utils.format_colorized import format_colorized
print(format_colorized(tokens, weights, tokenizer))
```

We get the following output:

<div className="example">
<span className="prompt">&lt;|im_start|&gt;system↵<br />Answer concisely; at most one sentence per response&lt;|im_end|&gt;↵<br />&lt;|im_start|&gt;user↵<br />What is the longest-lived rodent species?&lt;|im_end|&gt;↵<br />&lt;|im_start|&gt;assistant↵<br />The naked mole rat, which can live over 30 years.&lt;|im_end|&gt;↵<br />&lt;|im_start|&gt;user↵<br />How do they live so long?&lt;|im_end|&gt;↵<br />&lt;|im_start|&gt;assistant↵<br /></span>
<span className="completion">They evolved multiple protective mechanisms including special hyaluronic acid that prevents cancer, extremely stable proteins, and efficient DNA repair systems that work together to prevent aging.&lt;|im_end|&gt;<br /></span>
</div>
The green text is part of the prompt (i.e. with `weight=0`) and red is part of the completion (i.e. with `weight=1`). Note that the ↵ have been inserted just to make it clear whether each newline is part of the prompt or the completion; these are not actually part of the token sequence.

## Appendix: why don't you just use jinja-formatted chat templates?

In our experience, the Jinja2 templates are harder to write than python code -- especially when we need to get the whitespace exactly right. They are also unwieldy for supervised learning, where you need to put different labels on different tokens.

---

## File: completers.mdx

import { CookbookLink } from '../components/CookbookLink'

# Completers

The concept of policies is crucial to the RL training process. In the Tinker Cookbook, policies are implemented as `Completers`. Completers are abstractions that represent models or policies that can be sampled from, providing different levels of structure depending on your use case.

## Overview of Completer Types

The Tinker Cookbook provides two main types of completers, each designed for different use cases:

1. **TokenCompleter**: Operates on tokens and is used by RL algorithms
2. **MessageCompleter**: Operates on messages and needs to be used with a renderer

The choice between these depends on whether you're working at the token level for RL training or at the message level for interacting with and evaluating the model.

### TokenCompleter

The `TokenCompleter` is the foundational interface used by RL algorithms because they work directly with tokens.

```python
class TokenCompleter:
    async def __call__(
        self, model_input: types.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
```

This interface takes:
- `model_input`: The input to the model (of type `types.ModelInput`)
- `stop`: Stop conditions, either a list of strings or token IDs (combined into a `StopCondition` class). When training with reinforcement learning, this should be defined by the `initial_observation` function of the environment.

It returns a `TokensWithLogprobs` object containing:
- `tokens`: The generated token sequence
- `maybe_logprobs`: Optional log probabilities for each token

### MessageCompleter

The `MessageCompleter` operates at a higher level with structured messages, similarly to standard chat APIs. It takes a list of messages and returns a single assistant message response.

```python
class MessageCompleter:
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
```

For training purposes the `TokenCompleter` is the class we will use for RL training as we need to optimiza the same same set of tokens during the update step that the model output during rollout. The `MessageCompleter` is useful for sampling where we need to use the model output for semantic purposes such as Judge models or multi-agent environments.

The Tinker Cookbook uses two concrete implementations of these interfaces - <CookbookLink path="tinker_cookbook/completers.py">`TinkerTokenCompleter`</CookbookLink> and <CookbookLink path="tinker_cookbook/completers.py">`TinkerMessageCompleter`</CookbookLink> which are both wrappers around a `tinker.SamplingClient`. While the TinkerTokenCompleter operates directly on tokens, the TinkerMessageCompleter needs to be instantiated with a renderer to make it compatible with the inputs expected by the samping client.


---

## File: install.mdx

# Installing Tinker

There are two packages to install:

- [tinker](https://github.com/thinking-machines-lab/tinker), which performs basic low-level operations like `forward_backward`
- [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook), which is a collection of training code and experiment tools built on top of Tinker

Both packages can be installed with `pip` from their GitHub repositories:

```bash
pip install git+https://github.com/thinking-machines-lab/tinker.git
pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git
```

However, for the Cookbook, we'd recommend doing a local editable install, as you'll probably want to browse and edit the code:

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
pip install -e .
```

Tinker Cookbook also has a few optional dependencies, which you can install with the following command in an editable install:

```bash
pip install -e ".[envs]"
```

Or, for a non-editable install:

```bash
pip install "git+https://github.com/thinking-machines-lab/tinker-cookbook.git#egg=tinker-cookbook[envs]"
```


---

## File: rl.mdx

import { CookbookLink } from '../components/CookbookLink'

# Reinforcement learning

Reinforcement learning (RL) means learning from trial and error. Whereas in supervised learning, we're given input-output pairs, in RL, we're given inputs (prompts) and reward functions (i.e., a function for scoring candidate outputs). RL algorithms need to discover what good outputs look like.

Here are a few different types of RL training that we support in the Tinker Cookbook:

- *RL with Verifiable Rewards*: this is when we do RL on a reward function that checks model outputs using a program. Typically, the reward function checks the candidate answer against a reference answer, or, in coding cases, it may check if the candidate solution passes some unit tests. RLVR is especially suitable for teaching models to do reasoning (with chain-of-thought) and multi-step tool use (e.g., debugging and iterative modification pf programs).
- *RL on Human Feedback*: here, we assume we have an objective that can't be calculated by a simple program, and it requires some human judgement. For example, we typically want to optimize our models for helpfulness, which includes being clear, informative, and interesting. For RLHF, we train a *preference model* using supervised learning to match human judgement, scoring or ranking candidate outputs. Then we do RL on the preference model's scores. See the [Preferences](/preferences) section for more details.

We'll first show how to do small RL runs in the RLVR setting, then we'll show you how to define your own RL environments and train on them, then we'll provide examples for larger-scale or more complicated training setups.


We anticipate that people will want to use tinker for RL in a few different ways:

- Creating a specialist model that's SoTA at a specific skill, which existing models haven't been trained on. In this case, you'll want to start with a post-trained model that's already strong, and then do RL on an environment you've defined. See [RL Environments](/rl/rl-envs).
- Doing research on post-training pipelines. In this case, you'll probably want to chain together SL and RL and runs with different data mixes, environments, and reward functions. See our [RLHF example](/preferences/rlhf-example).
- Doing research on RL algorithms. Here, you'll probably want to find some existing environments to use as benchmarks, and either modify our provided training code (<CookbookLink path="tinker_cookbook/rl/train.py">rl/train.py</CookbookLink>) or write your own minimal training loop. We've provided a [minimal training loop](/rl/rl-loops) that you can use as a starting point.

---

## File: model-lineup.mdx

# Available Models in Tinker

The table below shows the models that are currently available in Tinker. We plan to update this list frequently as new models are released.

## What model should I use?

- In general, use MoE models, which are more cost effective than the dense models.
- Use 🐙 Base models only if you're doing research or are running the full post-training pipeline yourself
- If you want to create a model that is good at a specific task or domain, use an existing post-trained model model, and fine-tune it on your own data or environment.
    - If you care about latency, use one of the "⚡ Instruction" models, which will start outputting tokens without a chain-of-thought.
    - If you care about intelligence and robustness, use one of the "🤔 Hybrid" models, which can use long chain-of-thought.

## Full Listing

| Model Name | Training Type | Architecture | Size |
|------------|--------------|--------------|------|
| [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) | ⚡ Instruction | 🔀 MoE | 🦖 Large |
| [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | ⚡ Instruction | 🔀 MoE | 🦅 Medium |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 🤔 Hybrid | 🔀 MoE | 🦅 Medium |
| [Qwen/Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) | 🐙 Base | 🔀 MoE | 🦅 Medium |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 🤔 Hybrid | 🧱 Dense | 🦅 Medium |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 🤔 Hybrid | 🧱 Dense | 🦆 Small |
| [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | 🐙 Base | 🧱 Dense | 🦆 Small |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | ⚡ Instruction | 🧱 Dense | 🐣 Compact |
| [meta-llama/Llama-3.3-70B](https://huggingface.co/meta-llama/Llama-3.3-70B) | ⚡ Instruction | 🧱 Dense | 🦖 Large |
| [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B) | 🐙 Base | 🧱 Dense | 🦖 Large |
| [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | 🐙 Base | 🧱 Dense | 🦆 Small |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | ⚡ Instruction | 🧱 Dense | 🦆 Small |
| [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | 🐙 Base | 🧱 Dense | 🐣 Compact |
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 🐙 Base | 🧱 Dense | 🐣 Compact |

## Legend

### Training Types
- 🐙 **Base**: Foundation models trained on raw text data, suitable for post-training research and custom fine-tuning
- ⚡ **Instruction**: Models fine-tuned for following instructions and chat, optimized for fast inference
- 🤔 **Hybrid**: Models that can operate in both thinking and non-thinking modes

### Architecture
- 🧱 **Dense**: Standard transformer architecture with all parameters active
- 🔀 **MoE**: Mixture of Experts architecture with sparse activation

### Model Sizes

- 🐣 **Compact**: 1B-4B parameters
- 🦆 **Small**: 8B parameters
- 🦅 **Medium**: 30B-32B parameters
- 🦖 **Large**: 70B+ parameters

Note that the MoE models are much more cost effective than the dense models. E.g. the Qwen3-30B-A3B model has only 3B active parameters, so it'll cost around the same as a 3B dense model for training and inference.


---

## File: preferences/dpo-guide.mdx

import { Callout } from 'nextra/components'
import { CookbookLink } from '../../components/CookbookLink'

# Direct Preference Optimization (DPO)

Direct Preference Optimization (DPO) is a method for training language models to align with human preferences without requiring a separate reward model. Instead of using reinforcement learning with human feedback (RLHF), DPO directly optimizes the model to prefer chosen responses over rejected ones using a simple classification loss.

## DPO Algorithm Details

The core DPO loss is computed as:

$$
\mathcal{L}_{\theta} = -\mathbb{E}_{x, y_\text{chosen}, y_\text{rejected} \sim \mathcal{D}}\left[\log\sigma\left(\beta\log \frac{\pi_{\theta}(y_\text{chosen}|x)}{\pi_{\text{ref}}(y_\text{chosen}|x)} - \beta\log \frac{\pi_{\theta}(y_\text{rejected}|x)}{\pi_{\text{ref}}(y_\text{rejected}|x)}\right)\right]
$$

Where:
- $\pi_{\theta}$ is the current policy
- $\pi_{\text{ref}}$ is the reference model (typically the initial model before DPO training)
- $\beta$ is the DPO beta parameter
- Where $\mathcal{D}$ is a dataset of prompts $x$, a chosen response $y_{\text{chosen}}$ and a rejected response $y_{\text{rejected}}$

This optimizes the classical constrianed RLHF objective, where the reference model constrains deviation from the initial distribution.

<Callout type="info">
**DPO vs RLHF**: DPO eliminates the need for a separate reward model by directly optimizing the policy to prefer chosen responses. This makes training simpler and computationally cheaper than classical RLHF.
</Callout>


## Running DPO Training

The implementation is in <CookbookLink path="train_dpo.py">train_dpo.py</CookbookLink> with a CLI interface in <CookbookLink path="train_dpo_cli.py">train_dpo_cli.py</CookbookLink>. You can run it from the command line:

```bash
python -m tinker_cookbook.preference.train_dpo_cli \
    log_path=/tmp/dpo-hhh-experiment \
    model_name=meta-llama/Llama-3.2-1B \
    dataset=hhh \
    renderer_name=role_colon \
    learning_rate=1e-5 \
    dpo_beta=0.1
```

### Key Parameters

- `log_relpath`: Directory where results and checkpoints are saved
- `model_name`: Base model used as initialization and for the reference policy
- `dataset`: Dataset name (`hhh`, `helpsteer3`, `ultrafeedback`)
- `renderer_name`: How conversations are formatted (see [Rendering](../rendering.mdx))
- `learning_rate`: Learning rate for optimization
- `dpo_beta`: DPO beta parameter (controls the strength of preference learning)

### Available Datasets

There are several pre-defined datasets:

- **`hhh`**: Anthropic's Helpful-Harmless-Honest dataset
- **`helpsteer3`**: NVIDIA's HelpSteer3 preference dataset
- **`ultrafeedback`**: UltraFeedback binarized preferences dataset

These are implemented as `DPODatasetBuilder` classes and you can implement a custom dataset builder following the `tinker_cookbook.preference.preference_datasets` interface.

## Training Process

During training, you'll see output like this showing the DPO metrics:

```
                   Step 50
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric                         ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ accuracy                       │ 0.568627  │
│ batch_time                     │ 27.953704 │
│ chosen_reward                  │ 0.053621  │
│ dpo_loss                       │ 0.683825  │
│ learning_rate                  │ 0.000009  │
│ margin                         │ 0.002147  │
│ num_pairs                      │ 255       │
│ num_tokens                     │ 112638    │
│ progress                       │ 0.081210  │
│ rejected_reward                │ 0.032152  │
│ test/nll                       │ 1.871778  │
└────────────────────────────────┴───────────┘
```

The key metrics are:
- **`dpo_loss`**: The DPO classification loss
- **`accuracy`**: Accuracy of the implicit reward model evaluated on the preference dataset
- **`margin`**: Average difference between chosen and rejected rewards
- **`chosen_reward`/`rejected_reward`**: Average rewards for chosen/rejected responses

## Evaluating DPO Models

After training, you can evaluate your DPO model using the inspect evaluation framework:

```bash
MODEL_PATH=tinker://YOUR_MODEL_PATH_HERE
python -m tinker_cookbook.run_inspect_evals \
    model_path=$MODEL_PATH \
    model_name=meta-llama/Llama-3.2-1B \
    tasks=ifeval,paws \
    renderer_name=role_colon
```

This will evaluate the model on various benchmarks to measure the impact of preference optimization.

## Tips for DPO Training

1. **Beta Parameter**: Start with `dpo_beta=0.1` and adjust based on your dataset.

2. **Learning Rate**: Use a lower learning rate than supervised fine-tuning (typically 1e-5 to 1e-6).

3. **Base Model**: The base model should already be in-distribution with the preference data. Either start with a ligh SFT phase or collect on-policy preferences. While training would still work. sharp distribution mis-match will create strange model behaviors.




---

## File: preferences/rlhf-example.mdx

import { CookbookLink } from '../../components/CookbookLink'

# Reinforcement Learning from Human Feedback

## Train a pairwise preference model

We can first use supervised learning to train a model to do pairwise comparisons. To do this, we format the comparisons such that the model sees the prompt and the two completions, and it learns to predict the label (which was was preferred). Here, we train on the [Anthropic HHH](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset.

```bash
model_name=meta-llama/Llama-3.1-8B-Instruct
python -m tinker_cookbook.supervised.train_cli \
    log_path=/tmp/tinker-examples/hhh-pm \
    dataset=hhh \
    model_name=$model_name \
    learning_rate=4e-4
```

After the training is done, we can make note of the path of the final checkpoint. This path is printed to the console, and it's also in the `metrics.jsonl` file.

## Do RL using the reward model

Next, we can run RL using this reward model.  In <CookbookLink path="tinker_cookbook/rl/train_cli.py">rl/train_cli.py</CookbookLink>, we've hardcoded the tinker path of a Llama-8B-Instruct model that was we trained on the dataset above. Here, we define the reward function by generating 4 completions per prompt, and doing 4x3=12 pairwise comparisons between these completions, using the reward model.

```bash
python -m tinker_cookbook.rl.train_cli \
    env=hhh \
    log_path=/tmp/tinker-examples/hhh-rl \
    groups_per_batch=256 \
    group_size=4 \
    max_tokens=400 \
    learning_rate=2e-5
```

TODO: evaluate with reward model and inspect evals

---

## File: rl/rl-basic.mdx

import { CookbookLink } from '../../components/CookbookLink'

# Your First RL Run

We've provided a minimal script that runs RL on the [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k): <CookbookLink path="tinker_cookbook/recipes/rl_basic.py">rl_basic.py</CookbookLink>. For running this script, tinker-cookbook needs to be installed with optional envs dependencies (`pip install -e ".[envs]"`). You can run the minimal RL script from the command line as follows:

```bash
python -m tinker_cookbook.recipes.rl_basic
```

This script will fine-tune the Llama-3.1-8B base (pretrained) model on this dataset with the following reward function:

$$
1[\text{answer is correct}] + 0.1 \times (1[\text{answer is formatted correctly}] - 1)
$$

The training should take about 1 minute per iteration and climb to about 63% accuracy after 15 iterations (`env/all/correct`). You can look at the printouts for some other metrics of interest:

- `ac_tokens_per_turn`: the number of each tokens in each generated completion
- `env/all/format`: the fraction of completions that are formatted correctly
- `env/all/reward/total`: mean total reward (combining format and correctness as defined above)
- `entropy`: per-token entropy (mean negative log-probability of sampled tokens)
- `kl_sample_train_{v1,v2}`: two different approximations/estimators of KL divergence between the sampler's and learner's probability distribution (contributed to by numerical differences and rounding noise)
- `progress/done_frac`: what fraction of the total number of iterations we've completed so far
- `time/...`: time for different parts of the training loop

You can also look at the `log_path` directory for more detailed metrics. There are several files of interest, which are mostly the same as in the [Supervised Learning](/supervised-learning/sl-basic) case.

---

## File: rl/rl-hyperparams.mdx

# RL Hyperparameters

This guide covers the key hyperparameters for reinforcement learning training, from core settings to advanced configurations.

## Core Hyperparameters

### Learning Rate

Similar to the [supervised learning setting](../supervised-learning/sl-hyperparams), the learning rate is the most critical hyperparameter choice. We recommend using the guidance presented there as a starting point for RL experiments as well.


### Batch and Group Sizes

As described in our [RL environments](../rl/rl-envs.mdx) documentation, we use two key parameters:

- **`batch_size`**: The number of unique environments or problems used for training
- **`group_size`**: The number of rollouts performed per unique environment

If you have limited environments or problems available for training, increase the `group_size` to generate more training data. While the total number of rollouts depends on both parameters, we recommend scaling learning rates proportionally to $\text{LR} \propto \sqrt{\text{batch\_size}}$.

## Multiple Updates per Sampling Iteration

The `num_substeps` parameter controls how many policy weight updates are performed on data sampled from the last policy iteration, similar to PPO and GRPO.

### How it works:

- **`num_substeps = 1` (default)**: Each batch of collected trajectories is used for exactly one optimizer update
- **`num_substeps > 1`**: The batch of unique environments is split into `num_substeps` mini-batches, where each environment/problem has `group_size` rollouts (we pack all rollouts for a particular environment/problem in the same minibatch). We do a single update step on each mini-batch. Note that our implementation still takes only a single epoch through the data.

### Usage Guidelines:

- The batch size must be divisible by `num_substeps`
- Our experiments show that `num_substeps = 1` already gives decent performance, but if you would like to experiment with this parameter, we recommend starting with a low value of 2-4 and using the PPO objective.
- Higher values can lead to update steps that are too out-of-distribution for the policy. Consider limiting the number of updates or decreasing the learning rate when using multiple update steps.

## Advanced Training Configurations

⚠️ **Note**: These features are experimental and may be subject to instabilities. They are currently disabled by default.

### Streaming Minibatch Training

Enable streaming minibatch training by specifying the `StreamMinibatchConfig`. This approach overlaps trajectory sampling and model training, improving overall throughput by submitting training requests as soon as enough rollouts complete, without waiting for all sampling jobs to finish.

**Configuration Parameters:**

- **`groups_per_batch`**: Same as batch size (TODO: consider getting this from the dataset builder, also general naming around these is not great now)
- **`num_minibatches`**: Number of minibatches per substep—controls how many individual forward-backward requests we submit. This controls how the work is split (TODO: Why do we need this? We should just send a group as soon as it's ready).


**Important**: This remains on-policy training and is strictly a pipeline efficiency improvement.

### Async Off-Policy Training

Async training allows the model to train on trajectories generated with slightly older model versions, enabling higher throughput at the cost of some off-policy bias. While Tinker doesn't currently support in-flight weight changes, it supports the "off-by-K" async RL approach where multiple model iterations generate data simultaneously. Configure this by setting the `AsyncConfig` object.

**Configuration Parameters:**

- **`max_steps_off_policy`**: Maximum age (in training steps) of trajectories before they're discarded. Essentially, trajectories from policy iterations older than `max_steps_off_policy` steps will not be used.
- **`groups_per_batch`**: Number of new trajectory groups to accumulate (with a `group_size` number of rollouts each) before updating the current iteration of the model. Note: This is separate from the batch size used for dataset construction.

**Usage Guidelines:**

- Async RL is appropriate for applications with long and heterogeneous rollouts, such as very long CoT models, multi-hop tool use, or agentic workflows
- Start with a small value for `max_steps_off_policy` (less than 5)



## Monitoring and Run Health

Using policy-gradient algorithms with off-policy data can significantly degrade performance or even crash the policy, making monitoring essential during training.

### KL Divergence Monitoring

The current implementation logs the KL divergence between the data generation policy and the current learner: $\mathbb{D}_{KL}[\pi_{\text{sampler}}(\cdot|x)||\pi_{\theta}(\cdot|x)]$ using two separate [estimators](http://joschu.net/blog/kl-approx.html):

- `kl_sample_train_v1`
- `kl_sample_train_v2`


A few important notes to keep in mind:
- Even with full on-policy training, the divergence between sampling and learning policies will not be [exactly zero](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) due to implementation details
- In our experience training ist stable with KL divergence below 0.01
- If KL divergence crosses the recommended threshold, this indicates a numerical instability or potential issue with the training run



---

## File: rl/rl-loops.mdx

import { CookbookLink } from '../../components/CookbookLink'

# RL Training Loop
In this section we will walk through the RL training loop using a minimum working example on the Hendrycks Math dataset.


## Dataset Construction

The Hendrycks Math environment uses the <CookbookLink path="tinker_cookbook/rl/math_env.py">`MathDatasetBuilder`</CookbookLink> as described in `Creating RL Environments` to create structured batches of mathematical problems:

```python
from tinker_cookbook.rl.math_env import MathDatasetBuilder

dataset_builder = MathDatasetBuilder(
    batch_size=64,           # 64 problem groups per training batch
    group_size=32,           # 32 solution attempts per problem
    model_name_for_tokenizer="meta-llama/Llama-3.1",
    renderer_name="llama3",
    convo_prefix="standard"  # Includes few-shot examples
)

train_dataset, test_dataset = dataset_builder()
```

Here `MathDatasetBuilder` is a `RLDatasetBuilder` that creates a `RLDataset` where we have a batch of 64 problems, each of which creates a group of 32 identical environments.

## Policy (Completer) Construction

Following the `Completers` section, the policy is implemented as a <CookbookLink path="tinker_cookbook/completers.py">`TinkerTokenCompleter`</CookbookLink> that wraps the sampling client:

```python
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter

service_client = tinker.ServiceClient()
training_client = await service_client.create_lora_training_client_async(
        base_model="meta-llama/Llama-3.1-8B", rank=32
    )

async def get_policy(step: int):
    # Save a checkpoint that you can use for sampling
    sampling_path = training_client.save_weights_for_sampler(name="xxxx").result().path

    # Create a sampling client with that checkpoint
    sampling_client = service_client.create_sampling_client(model_path=sampling_path) #

    # Wrap in completer interface for RL algorithms
    return TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=512
    )
```

At each training step the `get_policy` function will save he training weights and update the sampling client with the new weights to generate on-policy data.


## Rollout Collection Loop

The key step of the RL training loop is collecting new on-policy data after each policy update. For each training batch we have

```python
import asyncio
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout

async def generate_rollouts(step: int, policy: TokenCompleter):
    # Creates 64 builders, each building 32 MathEnv instances
    env_group_builders_P = train_dataset.get_batch(step)
    # Generate rollouts for each group of 32 environments
    trajectory_groups_P = await asyncio.gather(
        *[do_group_rollout(builder, policy) for builder in env_group_builders_P]
    )
    taglist_P = [builder.logging_tags() for builder in env_group_builders_P]
    return trajectory_groups_P, taglist_P
```

Here the `do_group_rollout` function will run the policy in the environments and return a list of of Trajectory Groups. The `TrajectoryGroup` is an object that consists of a list of trajectories (a `Trajectory` object) and, final rewards and potentially other metadata. The rollout will return a `batch_size` list of `TrajectoryGroup` objects, each of which consists of `group_size` `Trajectory` objects.


## Data Processing

After we collect new data, we need to process it for training. We compute advantages by centering rewards within each problem group. At this step we can also optionally filter out groups with all successes or all failures as these have policy gradients of zero. Finally the `assemble_training_data` function converts the trajectories into token-level training examples.

```python
from typing import List
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.rl.train import (
    _remove_constant_reward_groups,
    compute_advantages,
    assemble_training_data
)

def process_trajectory_groups(trajectory_groups_P: List[TrajectoryGroup]):
    # (Optionally) Remove groups with all successes or all failures
    filtered_trajectory_groups_P = (
        _remove_constant_reward_groups(trajectory_groups_P)
    )
    # Compute advantages for each trajectory in each group
    advantages_P = compute_advantages(filtered_trajectory_groups_P)
    # Convert trajectories to token-level training examples
    data_D, _metadata_D = assemble_training_data(filtered_trajectory_groups_P, advantages_P)
    return data_D, _metadata_D
```

This function will return a list of training data `types.Datum` objects, as well as a list of metadata dictionaries.

## Model Training Step

The model update uses importance sampling loss:

```python
from typing import List
import torch
import tinker
from tinker import types
from tinker_cookbook.rl.train import _remove_mask

async def train_step(
    data_D: List[types.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
) -> List[torch.Tensor]:
    """Train the model on collected trajectories."""
    # Forward-backward pass on all data
    fwd_bwd_future = await training_client.forward_backward_async(
        list(map(_remove_mask, data_D)), loss_fn="importance_sampling"
    )
    # Optimizer step
    adam_params = types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    optim_step_future = await training_client.optim_step_async(adam_params)
    # Wait for results
    fwd_bwd_result = await fwd_bwd_future.result_async()
    _optim_step_result = await optim_step_future.result_async()
```

Here we use the standard importance sampling loss policy gradient loss function:

```math
\mathcal{L}_{\theta} = -\mathbb{E}_{x\sim \mathcal{D}, y \sim \pi_{\text{sampler}}(\cdot|x)}\left[\textbf{sg}\left(\frac{\pi_{\theta}(y|x)}{\pi_{\text{sampler}}(y|x)}\right)A(x, y)\log\pi_{\theta}(y|x)\right]
```

where we use the group-level advantages $A(x, y)$ to scale the policy gradient. Notice that even when we have exact on-policy RL, slight differences between the trainer and sampler model (even with the same weights) can introduce off-policy issues that can (severely) negatively affect learning.








## Complete Training Loop

Putting it all together, the complete training loop iterates between the following steps:

1. Create a policy with the current model weights
2. Generate rollouts
3. Process trajectory data into training examples
4. Update model parameters



```python
import time
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics

async def rl_loop():
    """Complete RL training loop for Hendrycks Math problems."""

    # Setup dataset
    dataset_builder = MathDatasetBuilder(
        batch_size=64,
        group_size=32,
        model_name_for_tokenizer="meta-llama/Llama-3.1-8B",
        renderer_name="llama3",
        convo_prefix="standard"
    )
    train_dataset, test_dataset = dataset_builder()

    # Setup training client
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        "meta-llama/Llama-3.1-8B", rank=32
    )

    # Setup logging
    log_dir = str("~/experiments/math-rl")

    num_batches = len(train_dataset)
    learning_rate = 1e-5

    #  Main training loop
    for i_batch in range(num_batches):
        # Setup metrics for logging
        t_start = time.time()
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        # 1. Create policy with current weights
        policy = await get_policy(i_batch)

        # 2. Generate rollouts
        with timed("generate_rollouts", metrics):
            # Generate rollouts: 64 groups × 32 environments = 2,048 total rollouts
            trajectory_groups_P, taglist_P = await generate_rollouts(i_batch, policy)

        # Compute trajectory metrics
        metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

        # 3. Process trajectory data into training examples
        data_D, _metadata_D = process_trajectory_groups(trajectory_groups_P)

        # 4. Update model parameters
        await train_step(data_D, training_client, learning_rate)

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)
```


## Running RL training

To run a training job you can use the our basic RL training <CookbookLink path="tinker_cookbook/recipes/rl_basic.py">`script`</CookbookLink>.

---

## File: rl/rl-envs.mdx

import { CookbookLink } from '../../components/CookbookLink'

# RL Environments

Here, we'll explain how to create your own RL environments and train on them. First, lets look at the basic classes, which can be found in <CookbookLink path="tinker_cookbook/rl/types.py">`tinker_cookbook.rl.types`</CookbookLink>. As you can see, there's an `Env` interface, corresponding to an RL environment. To write an environment, you need to implement two methods: `initial_observation` and `step`.

```python
class Env:
    """
    Stateful environment that a single agent interacts with.
    Discard after running for one episode.
    """

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        raise NotImplementedError

    async def step(self, action: Action) -> StepResult:
        raise NotImplementedError
```

Note that this `Env` operates on *tokens*, rather than strings or messages. Why define it this way, when it's usually more natural to define the logic in terms of strings or messages? We've defined `Env` this way because this interface is what's needed by the *training* code. The training code needs to know the exact tokens that were sampled, and their logprobs. It might be more convenient to write the logic in terms of strings or messages. See [Env Adapters](/rl/env-adapters) for more on how to write adapters between our `Env` class and other libraries.

We need to write two more small classes to use this environment in the RL training code. First, since the environment is discarded after a single episode, we need to be able to instantiate new environments in the training loop. We actually build a *group* of environments at a time, which enables multi-agent training or objectives that compare multiple samples (for example, a reward model that acts on a pair of samples).

```python
class EnvGroupBuilder:
    """
    Builds a group of environments.
    """

    async def make_envs(self) -> Sequence[Env]:
        raise NotImplementedError
```

This object creates a group of environments. Often it does the trivial thing of returning a list of copies of the same environment.

Finally, we need a dataset of these EnvGroupBuilders.

```python
class RLDataset:
    """
    Dataset of EnvGroupBuilders.
    """

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        raise NotImplementedError
```


That's a lot of classes! But their combination gives us a lot of flexibility. In previous implementations (like OpenAI Gym), the dataset is implicitly part of the environment; this structure is more modular and gives us more control over the data loading.

## Building a simple example

You can find an example of writing a new RL environment in the <CookbookLink path="tinker_cookbook/recipes/twenty_questions">Twenty Questions</CookbookLink> directory.
Here, we define a multi-step environment, where we're training a question-asking agent, which asks questions to another agent to guess a hidden word.
In this case, the answerer model is fixed and is Llama-3.1-8B-Instruct.
The player model (which we fine-tune) is also based on that same model.

You can run the training script as follows:

```bash
uv run python -m tinker_cookbook.recipes.twenty_questions.train
```


---

## File: supervised-learning/sl-hyperparams.mdx

# Supervised Learning Hyperparameters

Successful fine-tuning of any LLM requires careful tuning of the hyperparameters. While the most accurate way to tune hyperparameters is to sweep over a range of values for each hyperparameter and pick the values that minimize the loss or maximize the evals, this is often time-consuming and expensive. We provide some starting recommendations for the most important hyperparameters.


## Learning rate

The most important hyperparameter is generally the learning rate (LR). Our current best estimate of optimal LR for a model $m$ is the following:

$$ LR(m) = lr_{base} · M_{LoRA} · \Big(\frac{2000}{H_m}\Big)^{P_m} $$

where $lr_{base}$ is a constant base lr, $M_{LoRA}$ is a multiplier applied when using LoRA (1 if using full-finetuning), $H_m$ is the hidden size of the model $m$, and $P_m$ is a model-specific exponent adjustment. Importantly, note that this function is independent of the LoRA rank.

Our current best estimates are the following: $lr_{base} = 5e-5$,
$M_{LoRA} = 10$, $P_m = 0.0775$ for Qwen models and $P_m = 0.781$ for Llama models.

You can use the following function to get the recommended lr for any model:
```
from tinker_cookbook.hyperparam_utils import get_lr
model_name = "meta-llama/Llama-3.2-1B"
print(get_lr(model_name))
```

We can define the regret of using any lr as the following:
$$regret(lr') = \frac{loss(lr') - min_{lr} loss(lr)}{min_{lr} loss(lr)}$$

We used our lr estimate to run SFT experiments across a variety of datasets, dataset sizes, batch_sizes and lora_ranks, and found the regret to be \<0.5%.


## Batch size

The second most important hyperparameter is the batch size. For small batch sizes, there's a phenomenon of *perfect scaling*, where the LR and batchsize should be varied together as $LR \propto \sqrt{B}$, and the learning curve only depends on $\frac{LR}{\sqrt{B}}$. See e.g., [Shallue et al. (2018)](https://arxiv.org/abs/1811.03600), in the training-from-scratch setting. When fine-tuning LLMs, we're often in a regime where smaller batch sizes give better performance, at the cost of longer training time; moreover, the $LR \propto \sqrt{B}$ scaling doesn't always hold. When doing SL fine-tuning, we recommend using smaller batch sizes like 128, depending on your tolerance for longer training time. For best results, you should aim for at least 100 steps of training (but usually get best results with 1000 or more). [NOTE: we are not confident about these recommendations and need to do further research.]


---

## File: supervised-learning/sl-basic.mdx

import { CookbookLink } from '../../components/CookbookLink'

# Basic Supervised Learning

We've provided an implementation of supervised learning in <CookbookLink path="tinker_cookbook/supervised/train.py">train_cli.py</CookbookLink>. To use this training loop, you'll need to create a `Config` object with the data and parameters.

We've provided an example of a script that starts up an SL training run in <CookbookLink path="tinker_cookbook/recipes/sl_basic.py">sl_basic.py</CookbookLink>. You can run it from the command line as follows:

```bash
python -m tinker_cookbook.recipes.sl_basic
```

This script will fine-tune the Llama-3.1-8B base (pretrained) model on a small dataset called [NoRobots](https://huggingface.co/datasets/HuggingFaceH4/no_robots), created by Hugging Face.

- Each step you should see a printout of the train and test loss, along with other stats like timing.
- The training script will also print out what the data looks like, with predicted tokens (weight=1) in green and context tokens (weight=0) in yellow.
- The training script will write various logs and checkpoint info to the `log_path` directory, which is set to `/tmp/tinker-examples/sl_basic` in the example script.

Looking at this directory, there are several files of interest:
- `metrics.jsonl`: the training metrics that also were printed to the console. You can load and plot them like this:

    ```python
    import pandas
    import matplotlib.pyplot as plt
    df = pandas.read_json("/tmp/tinker-examples/sl_basic/metrics.jsonl", lines=True)
    plt.plot(df['train_mean_nll'], label='train_loss')
    plt.plot(df['test/nll'].dropna(), label='test_loss')
    plt.legend()
    plt.show()
    ```
You should see a plot like this:
![Train and test loss as a function of steps](./images/train_test_loss.png)


- `checkpoints.jsonl`: this files stores the checkpoints that were saved during training. Recall from [Saving and Loading](/save-load) that there are (currently) two kinds of checkpoints: one that has "/sampler_weights/" in the path (used for sampling), and the other that has "/weights/" in the path (includes full optimizer state, used for resuming training). If you interrupt the training script, then run it again, it will ask you if you want to resume training. If you choose to do so, it'll load the last (full state) checkpoint from this file.
- `config.json`: the configuration that you used for training.

In the `sl_basic` script, you'll see that there's also some disabled code (under `if 0:`) that shows how to use your own dataset, specified as a JSONL file, provided in the format of <CookbookLink path="example-data/conversations.jsonl">conversations.jsonl</CookbookLink>.

---

## File: supervised-learning/prompt-distillation.mdx

import { CookbookLink } from '../../components/CookbookLink'

# Prompt Distillation

Prompt distillation is a training technique in which a model is optimized to behave as though it had been provided with a long and complex prompt, without requiring access to that prompt during inference.

At a high level, this procedure involves two main steps:
- **Creation of distillation data**: A teacher prompt, which is typically lengthy and highly detailed, provides explicit, step-by-step instructions. A teacher model uses this prompt to generate responses for a set of queries.
- **Training the student model**: A student model is then trained (or fine-tuned) on the distilled dataset, thereby learning to reproduce the essential behaviors and reasoning encoded in the teacher’s instructions.

---

## Overview

Let $f_T$ and $f_S$ denote the teacher and student models, respectively. Given an instruction prompt $P$ and a query $q_i$, the teacher model generates a response $r_i$:

$$
r_i = f_T([P, q_i])
$$

Here, the prompt $P$ and the query $q_i$ are concatenated to form the input to the teacher model $f_T$. For a dataset of queries $Q = \{q_i \mid 1 \leq i \leq D\}$, we obtain a corresponding set of teacher responses $R = \{r_i \mid 1 \leq i \leq D\}$.

The distillation training dataset is defined as the set of query–response pairs (excluding the original prompt):

$$
T = \{(q_i, r_i) \mid 1 \leq i \leq D\}.
$$

The student model $f_S$ is then trained to minimize the cross-entropy loss:

$$
\ell(f_S(q_i), r_i) = \ell(f_S(q_i), f_T([P, q_i])).
$$

---

## Example

The Tinker Cookbook provides a prompt distillation recipe tailored for a language classification task. The objective is straightforward: given a text query, the model should predict a two-character code corresponding to the language of the input. The set of possible labels is:
```
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French), hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese), zh (Chinese - Simplified), ot (Other/Unknown).
```

The recipe in <CookbookLink path="tinker_cookbook/recipes/prompt_distillation/create_data.py">recipes/prompt_distillation/create_data.py</CookbookLink> also includes handling strategies for inputs containing code, numerical content, or multiple languages.

In the example below, the same model (`Qwen/Qwen3-30B-A3B`) is used as both teacher and student, though in general they need not be identical.

---

### Step 1: Generate Training Data

Create prompt distillation data using the teacher model using <CookbookLink path="tinker_cookbook/recipes/prompt_distillation/create_data.py">recipes/prompt_distillation/create_data.py</CookbookLink>:

```bash
python -m tinker_cookbook.recipes.prompt_distillation.create_data \
  output_file=/tmp/tinker-datasets/prompt_distillation_lang.jsonl
```

This command will:
- Use the configured teacher model to generate language classification examples
- Save the distilled dataset to the specified output file
- Create diverse training examples suitable for student model fine-tuning

### Step 2: Train the Student Model

Fine-tune a student model on the distillation data using <CookbookLink path="tinker_cookbook/recipes/prompt_distillation/train.py">recipes/prompt_distillation/train.py</CookbookLink>:

```bash
python -m tinker_cookbook.recipes.prompt_distillation.train
```

The training script will:
- Load the generated distillation dataset
- Apply optimized training configurations
- Fine-tune the student model for language classification

### Step 3: Test Your Model

Once training is complete, you can test your distilled model by sampling from the trained model to verify its performance on language classification tasks.

## Advanced Configuration

The prompt distillation recipe can be customized for different scenarios:

- **Teacher model selection**: Choose different base models based on your requirements
- **Sampling strategies**: Adjust temperature and other generation parameters
- **Data volume**: Scale the number of generated examples based on your needs
- **Training hyperparameters**: Fine-tune learning rates and other training settings


---

## File: supervised-learning/sweep-case-study.mdx

import { CookbookLink } from '../../components/CookbookLink'

# Sweep case study

In [Supervised Learning Hyperparameters](./sl-hyperparams), we introduced default hyperparameters as a starting point. While defaults are useful, optimal values are often task-specific. A hyperparameter sweep (systematically testing values across a range) is a more reliable way to identify the best hyperparameter values.

This page demonstrates how to sweep over the **learning rate (LR)** to find an optimal value.


## Setup

We use the simple supervised learning training loop in
<CookbookLink path="tinker_cookbook/recipes/sl_loop.py">sl_loop.py</CookbookLink>, which trains a Llama-3.1-8B model.

To retrieve the model’s default learning rate recommendation:
```
from tinker_cookbook.hyperparam_utils import get_lr
print(get_lr("meta-llama/Llama-3.1-8B"))
```
This should output
```
0.0002856415043086949  # ≈ 2.8e-4
```
This default value provides a baseline. A common best practice is to sweep one order of magnitude above and below the default. For this case, we sweep over: $lr \in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]$



## Running the sweep
Launch experiments in separate terminal windows for each LR value. For example:
```bash
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.003 log_path=/tmp/sft-lr-sweep/lr-0.003
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.001 log_path=/tmp/sft-lr-sweep/lr-0.001
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.0003 log_path=/tmp/sft-lr-sweep/lr-0.0003
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.0001 log_path=/tmp/sft-lr-sweep/lr-0.0001
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.0003 log_path=/tmp/sft-lr-sweep/lr-0.00003
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.0001 log_path=/tmp/sft-lr-sweep/lr-0.00001
```
You can also automate this process by writing a script that spawns multiple tmux windows and launches experiments programmatically.


## Collecting Results
After the experiments are complete, you can read the `metrics.jsonl` file for all the experiments:
```python
from glob import glob
import pandas
import os
import json

data = []
for fname in sorted(glob(os.path.expanduser("/tmp/sft-lr-sweep/*/metrics.jsonl"))):
    df = pandas.read_json(fname, lines=True)
    # make sure the experiment is completed
    if len(df) == 0 or df["progress"].iloc[-1] < 0.98:
        continue
    config_fname = fname.replace("metrics.jsonl", "config.json")
    with open(config_fname, "rb") as f:
        metadata = json.load(f)
    data.append({
        "fname": fname,
        "learning_rate": metadata["learning_rate"],
        "final_loss": df["train_mean_nll"].iloc[-1].item()
    })

print(f"Read metrics for {len(data)} experiments")
```
If all the experiments are completed, the above code should print:
```
Read metrics for 6 experiments
```

## Visualizing the Sweep
Plot the `final_loss` as a function of `learning_rate`:
```python
import matplotlib.pyplot as plt
df = pandas.DataFrame(data)
plt.plot(df["learning_rate"], df["final_loss"], marker='o')
plt.axhline(y=df["final_loss"].min(), color="green", linestyle="--")
plt.ylim(1.65, 1.8)
plt.xscale("log")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Final Loss")
plt.title("Final Loss vs Learning Rate")
plt.show()
```
You should see a U-shaped curve, similar to this:
![final_loss_vs_lr](./images/lr_sweep.png)

If the full U-curve is not visible in your setting, expand the sweep range by adding more LR values.


## Determining the Optimal LR
The optimal learning rate is the one that minimizes the loss. The plot above shows that the optimal LR is `3e-4` which you can also calculate by finding the minima:
```
optimal_lr = df["learning_rate"][df["final_loss"].idxmin()]
print(f"The optimal LR is {optimal_lr:.2e}")
```
Expected output:
```
The optimal LR is 3.00e-04
```

Note that the optimal LR in our sweep (`3e-4`) is very close to the default LR (`2.8e-4`).

---

## File: supervised-learning/sl-loop.mdx

import { CookbookLink } from '../../components/CookbookLink'

# Supervised Learning Training Loop

We've provided a simple SL training loop in <CookbookLink path="tinker_cookbook/recipes/sl_loop.py">sl_loop.py</CookbookLink>, which avoids using our dataset classes and instead  defines the data loading a more self-contained way. This is for people who like to write their own training loops or learn about how things work under the hood. Our more performant implementation in <CookbookLink path="tinker_cookbook/supervised/train.py">supervised/train.py</CookbookLink> does basically the same thing, but with some performance optimizations, and with some additional features like periodic evals.

---

# PART 2: TYPE DEFINITIONS

Total types collected: 30

## Type: AdamParams

```python
class AdamParams(StrictBase):
    learning_rate: float = 0.0001
    """Learning rate for the optimizer"""

    beta1: float = 0.9
    """Coefficient used for computing running averages of gradient"""

    beta2: float = 0.95
    """Coefficient used for computing running averages of gradient square"""

    eps: float = 1e-12
    """Term added to the denominator to improve numerical stability"""
```

## Type: ComputeLogprobsResponse

```python
class ComputeLogprobsResponse(BaseModel):
    logprobs: Sequence[Optional[float]]

    type: Literal["compute_logprobs"] = "compute_logprobs"
```

## Type: CreateModelResponse

```python
class CreateModelResponse(BaseModel):
    model_id: ModelID

    type: Literal["create_model"] = "create_model"
```

## Type: Datum

```python
class Datum(StrictBase):
    loss_fn_inputs: LossFnInputs
    """Dictionary mapping field names to tensor data"""

    model_input: ModelInput

    @model_validator(mode="before")
    @classmethod
    def convert_tensors(cls, data: Any) -> Any:
        """Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction."""
        if isinstance(data, dict) and "loss_fn_inputs" in data:
            loss_fn_inputs = data["loss_fn_inputs"]
            if isinstance(loss_fn_inputs, dict):
                converted_inputs = {}
                for key, value in loss_fn_inputs.items():
                    converted_inputs[key] = cls._maybe_convert_array(key, value)
                data = dict(data)  # Make a copy
                data["loss_fn_inputs"] = converted_inputs
        return data

    @classmethod
    def _maybe_convert_array(cls, key: str, value: Any) -> Any:
        """Convert torch.Tensor, numpy array, or 1-D list to TensorData if needed."""
        if _HAVE_TORCH and isinstance(value, torch.Tensor):
            return TensorData.from_torch(value)
        elif isinstance(value, np.ndarray):
            return TensorData.from_numpy(value)
        elif isinstance(value, list):
            # assume it's 1d and infer the dtype from the key
            return TensorData(data=value, dtype=_key_to_type[key], shape=[len(value)])
        else:
            return value


_key_to_type = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
}
```

## Type: EncodedTextChunk

```python
class EncodedTextChunk(StrictBase):
    tokens: Sequence[int]
    """Array of token IDs"""

    type: Literal["encoded_text"] = "encoded_text"

    @property
    def length(self) -> int:
        return len(self.tokens)
```

## Type: ForwardBackwardInput

```python
class ForwardBackwardInput(StrictBase):
    data: List[Datum]
    """Array of input data for the forward/backward pass"""

    loss_fn: LossFnType
    """Fully qualified function path for the loss function"""
```

## Type: ForwardBackwardOutput

```python
class ForwardBackwardOutput(BaseModel):
    loss_fn_output_type: str
    """The type of the ForwardBackward output. Can be one of [...] TODO"""

    loss_fn_outputs: List[LossFnOutput]
    """Dictionary mapping field names to tensor data"""

    metrics: Dict[str, float]
    """Training metrics as key-value pairs"""
```

## Type: GetInfoResponse

```python
class GetInfoResponse(BaseModel):
    type: Optional[Literal["get_info"]] = None

    model_data: ModelData

    model_id: ModelID

    is_lora: Optional[bool] = None

    lora_rank: Optional[int] = None

    model_name: Optional[str] = None

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
```

## Type: GetServerCapabilitiesResponse

```python
class GetServerCapabilitiesResponse(BaseModel):
    supported_models: List[SupportedModel]
```

## Type: ImageAssetPointerChunk

```python
class ImageAssetPointerChunk(StrictBase):
    format: Literal["png", "jpeg"]
    """Image format"""

    height: int
    """Image height in pixels"""

    location: str
    """Path or URL to the image asset"""

    tokens: int
    """Number of tokens this image represents"""

    width: int
    """Image width in pixels"""

    type: Literal["image_asset_pointer"] = "image_asset_pointer"

    @property
    def length(self) -> int:
        return self.tokens
```

## Type: LoadWeightsResponse

```python
class LoadWeightsResponse(BaseModel):
    path: Optional[str] = None
    """A tinker URI for model weights at a specific step"""

    type: Optional[Literal["load_weights"]] = None
```

## Type: LoraConfig

```python
class LoraConfig(StrictBase):
    rank: int
    """LoRA rank (dimension of low-rank matrices)"""

    seed: Optional[int] = None
    """Seed used for initialization of LoRA weights.

    Useful if you need deterministic or reproducible initialization of weights.
    """

    train_unembed: bool = True
    """Whether to add lora to the unembedding layer"""

    train_mlp: bool = True
    """Whether to add loras to the MLP layers (including MoE layers)"""

    train_attn: bool = True
    """Whether to add loras to the attention layers"""
```

## Type: LossFnInputs

```python
LossFnInputs: TypeAlias = Dict[str, TensorData]
```

## Type: LossFnOutput

```python
LossFnOutput: TypeAlias = Dict[str, TensorData]
```

## Type: LossFnType

```python
LossFnType: TypeAlias = Literal["cross_entropy", "importance_sampling", "ppo"]
```

## Type: ModelData

```python
class ModelData(BaseModel):
    arch: Optional[str] = None

    model_name: Optional[str] = None
```

## Type: ModelID

```python
ModelID: TypeAlias = str
```

## Type: ModelInput

```python
class ModelInput(StrictBase):
    chunks: List[ModelInputChunk]
    """Sequence of input chunks (formerly TokenSequence)"""


    @classmethod
    def from_ints(cls, tokens: List[int]) -> "ModelInput":
        """
        Create a ModelInput from a list of ints (tokens).
        """
        return cls(chunks=[EncodedTextChunk(tokens=tokens)])

    def to_ints(self) -> List[int]:
        """
        Convert the ModelInput to a list of ints (tokens)
        Throws exception if there are any non-token chunks
        """
        if not all(isinstance(chunk, EncodedTextChunk) for chunk in self.chunks):
            raise ValueError(f"to_ints only supported for ModelInput with EncodedTextChunks, got {[type(chunk) for chunk in self.chunks]}")
        return [token for chunk in self.chunks for token in chunk.tokens]

    @property
    def length(self) -> int:
        """
        Return the total context length used by this ModelInput.
        """
        return sum(chunk.length for chunk in self.chunks)

    @classmethod
    def empty(cls) -> "ModelInput":
        """
        Create an empty ModelInput.
        """
        return cls(chunks=[])

    def append(self, chunk: ModelInputChunk) -> "ModelInput":
        """
        Add a new chunk, return a new ModelInput.
        """
        return ModelInput(chunks=self.chunks + [chunk])

    def append_int(self, token: int) -> "ModelInput":
        """
        Add a new token, return a new ModelInput.
        """
        return self.append(EncodedTextChunk(tokens=[token]))
```

## Type: ModelInputChunk

```python
ModelInputChunk: TypeAlias = Annotated[
    Union[EncodedTextChunk, ImageAssetPointerChunk], PropertyInfo(discriminator="type")
]
```

## Type: OptimStepResponse

```python
class OptimStepResponse(BaseModel):
    metrics: Optional[Dict[str, float]] = None
    """Optimization step metrics as key-value pairs"""
```

## Type: SampleResponse

```python
class SampleResponse(BaseModel):
    sequences: Sequence[SampledSequence]

    type: Literal["sample"] = "sample"

    prompt_logprobs: Optional[List[Optional[float]]] = None
    """
    If prompt_logprobs was set to true in the request, logprobs are computed for
    every token in the prompt. The `prompt_logprobs` response contains a float32
    value for every token in the prompt.
    """
```

## Type: SampledSequence

```python
class SampledSequence(BaseModel):
    stop_reason: StopReason
    """Reason why sampling stopped"""

    tokens: List[int]
    """List of generated token IDs"""

    logprobs: Optional[List[float]] = None
    """Log probabilities for each token (optional)"""
```

## Type: SamplingParams

```python
class SamplingParams(BaseModel):
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate"""

    seed: Optional[int] = None
    """Random seed for reproducible generation"""

    stop: Union[str, Sequence[str], Sequence[int], None] = None
    """Stop sequences for generation"""

    temperature: float = 1
    """Sampling temperature"""

    top_k: int = -1
    """Top-k sampling parameter (-1 for no limit)"""

    top_p: float = 1
    """Nucleus sampling probability"""
```

## Type: SaveWeightsForSamplerResponse

```python
class SaveWeightsForSamplerResponse(BaseModel):
    path: str
    """A tinker URI for model weights for sampling at a specific step"""

    type: Optional[Literal["save_weights_for_sampler"]] = None
```

## Type: SaveWeightsResponse

```python
class SaveWeightsResponse(BaseModel):
    path: str
    """A tinker URI for model weights at a specific step"""

    type: Optional[Literal["save_weights"]] = None
```

## Type: StopReason

```python
StopReason: TypeAlias = Literal["length", "stop"]
```

## Type: SupportedModel

```python
class SupportedModel(BaseModel):
    model_name: Optional[str] = None
```

## Type: TensorData

```python
class TensorData(StrictBase):
    data: Union[List[int], List[float]]
    """Flattened tensor data as array of numbers."""

    dtype: TensorDtype

    shape: Optional[List[int]] = None
    """Optional.

    The shape of the tensor (see PyTorch tensor.shape). The shape of a
    one-dimensional list of length N is `(N,)`. Can usually be inferred if not
    provided, and is generally inferred as a 1D tensor.
    """

    @classmethod
    def from_numpy(cls, array: npt.NDArray[Any]) -> "TensorData":
        return cls(
            data=array.flatten().tolist(),
            dtype=_convert_numpy_dtype_to_tensor(array.dtype),
            shape=list(array.shape),
        )

    @classmethod
    def from_torch(cls, tensor: "torch.Tensor") -> "TensorData":
        return cls(
            data=tensor.flatten().tolist(),
            dtype=_convert_torch_dtype_to_tensor(tensor.dtype),
            shape=list(tensor.shape),
        )

    def to_numpy(self) -> npt.NDArray[Any]:
        """Convert TensorData to numpy array."""
        numpy_dtype = _convert_tensor_dtype_to_numpy(self.dtype)
        arr = np.array(self.data, dtype=numpy_dtype)
        if self.shape is not None:
            arr = arr.reshape(self.shape)
        return arr

    def to_torch(self) -> "torch.Tensor":
        """Convert TensorData to torch tensor."""
        if not _HAVE_TORCH:
            raise ImportError("PyTorch is not installed. Cannot convert to torch tensor.")

        torch_dtype = _convert_tensor_dtype_to_torch(self.dtype)
        tensor = torch.tensor(self.data, dtype=torch_dtype)
        if self.shape is not None:
            tensor = tensor.reshape(self.shape)
        return tensor

    def tolist(self) -> List[Any]:
        return self.to_numpy().tolist()


def _convert_tensor_dtype_to_numpy(dtype: TensorDtype) -> npt.DTypeLike:
    """Convert TensorDtype to numpy dtype-like."""
    if dtype == "float32":
        return np.float32
    elif dtype == "int64":
        return np.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")


def _convert_tensor_dtype_to_torch(dtype: TensorDtype) -> "torch.dtype":
    """Convert TensorDtype to torch dtype."""
    if not _HAVE_TORCH:
        raise ImportError("PyTorch is not installed. Cannot convert to torch dtype.")
    import torch

    if dtype == "float32":
        return torch.float32
    elif dtype == "int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")


def _convert_numpy_dtype_to_tensor(dtype: np.dtype[Any]) -> TensorDtype:
    """Convert numpy dtype to TensorDtype."""
    if dtype.kind == "f":
        return "float32"
    elif dtype.kind == "i":
        return "int64"
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")


def _convert_torch_dtype_to_tensor(dtype: "torch.dtype") -> TensorDtype:
    """Convert torch dtype to TensorDtype."""
    # torch.dtype objects have .is_floating_point
    if getattr(dtype, "is_floating_point", False):
        return "float32"
    else:
        return "int64"
```

## Type: TensorDtype

```python
TensorDtype: TypeAlias = Literal["int64", "float32"]
```

## Type: UnloadModelResponse

```python
class UnloadModelResponse(BaseModel):
    model_id: ModelID

    type: Optional[Literal["unload_model"]] = None
```
