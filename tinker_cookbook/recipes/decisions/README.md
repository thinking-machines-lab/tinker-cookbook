# Decision model

Use Tinker to create a decision model that provides a probability distribution over a set of user defined choices in response to a question. This recipe also includes helpers for training any supported model as a decision model, using cross-entropy loss to fit hard labels or a custom loss on the probability distribution to fit soft labels.

A decision model may be a good fit if:

- Cost or latency is more important than additional test-time compute (reasoning)
- The decision is categorical
- A probability distribution over the choices provides more signal than a single choice

## Usage

```python
import asyncio

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.decisions import Decision, DecisionRenderer, sample_decision
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "thinkingmachines/Inkling-Small"


async def main() -> None:
    sampling_client = await tinker.ServiceClient().create_sampling_client_async(base_model=MODEL)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL), get_tokenizer(MODEL)
    )
    decision_renderer = DecisionRenderer(renderer)

    decision = Decision(
        "A user asked: How do I build a decision model with Tinker Cookbook?\n\n"
        "Which team should answer it?",
        [
            ("billing", "Charges, invoices and payments"),
            ("sdk", "The tinker Python package"),
            ("cookbook", "The Tinker Cookbook"),
            ("console", "The web console"),
        ],
    )
    print(await sample_decision(sampling_client, decision_renderer.render(decision)))
    # {"billing": 0.0, "sdk": 0.01, "cookbook": 0.99, "console": 0.0}


asyncio.run(main())
```

A `Decision` is a question plus an ordered list of `(label, description)` choices. `DecisionRenderer` turns it into the tokens the model is scored on, and `sample_decision` returns a probability per label, in choice order. Each call is one request, so `asyncio.gather` many of them to batch.

### Demo

```bash
uv run -m tinker_cookbook.recipes.decisions.demo
```

Then ask a question about Tinker and the model will assign a probability to each team that could answer it.

## How it works

Each choice is given an id of the form `choice#A`, `choice#B`, `choice#C`, etc. The system prompt message includes instructions to answer with only the choice id in response to the user prompt.

The user message includes the question and the choices, each choice with a label and optional description in addition to the choice id.

The assistant message is pre-filled as just `choice#?`, and then we make a sampling call with `max_tokens=1` and with `target_prompt_logprobs` set to the tokens `[A, B, C, ...]` at the position of `?` in the assistant message to get logprobs for each choice. Then taking a softmax over the logprobs gives the probability distribution restricted to the valid choices.

The `choice#` prefix in the assistant response is used to reduce ambiguity in the model's response. If the assistant message was merely a single letter (A, B, C, ...), the probability for `A` would include continuations such as "A reasonable answer is C."

## Train

### Hard labels

`decision_choice_datums` takes `(decision, target label)` pairs and uses the default cross-entropy loss to train the model to predict the target label. This is easiest to use when the training target is a single choice.

```python
from tinker_cookbook.recipes.decisions import decision_choice_datums

datums = decision_choice_datums(decision_renderer, [(decision, "cookbook"), ...])
fwd_bwd_future = await training_client.forward_backward_async(datums, loss_fn="cross_entropy")
optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
await fwd_bwd_future.result_async()
await optim_future.result_async()
```

### Soft labels

Use `decision_dist_datums` with `probability_loss` to define a custom loss function over the probability distribution choices. This is useful for calibrating to target a specific distribution or applying a non-cross-entropy loss (e.g. Brier).

```python
import torch
from tinker_cookbook.recipes.decisions import decision_dist_datums, probability_loss


def manhattan_loss(choice_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return (choice_probs - target_probs).abs().sum()


datums = decision_dist_datums(decision_renderer, [(decision, {"heads": 0.5, "tails": 0.5}), ...])
fwd_bwd_future = await training_client.forward_backward_custom_async(
    datums, probability_loss(manhattan_loss)
)
optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
await fwd_bwd_future.result_async()
await optim_future.result_async()
```

## Customizing the answer format

Use a `ChoiceRenderer` to customize how the choice ids are rendered.

The forecasting example below writes `choice#Y` / `choice#N` for `yes` / `no` instead of `choice#A` / `choice#B`:

```python
REPLIES = {"yes": "choice#Y", "no": "choice#N"}
ANSWER_FORMAT = "choice#?"


class YesNoChoiceRenderer(ChoiceRenderer):
    def __init__(self, renderer: Renderer):
        self.tokenizer = renderer.tokenizer

    def render_choices(self, labels: Sequence[str]) -> ChoiceRendering:
        reply_tokens = {
            label: self.tokenizer.encode(REPLIES[label], add_special_tokens=False)
            for label in labels
        }
        format_tokens = self.tokenizer.encode(ANSWER_FORMAT, add_special_tokens=False)
        return ChoiceRendering(
            # the format to show the model
            answer_tokens=format_tokens,
            # the position in the answer format to measure logprobs
            answer_position=len(format_tokens) - 1,
            # the token corresponding to each label ("yes" -> "Y", "no" -> "N")
            label_tokens={label: tokens[-1] for label, tokens in reply_tokens.items()},
        )


decision_renderer = DecisionRenderer(renderer, YesNoChoiceRenderer(renderer))
```

## Example: forecasting

To provide a comparison of a decision model against an existing Tinker benchmark for decision making, `forecasting.py` reproduces the [forecasting recipe](../forecasting/) with a decision model.

For each Prophet Arena market, the decision model is given choices `yes` or `no` for how the market will resolve, where the probability of `yes` is analogous to the original recipe's forecast. The data, temporal split, validation set and metrics (Brier reward, accuracy, AUC) are those of the original recipe.

### Configuration

The original forecasting recipe trained with 1024 questions, with 32 rollouts per question, over 2 epochs, totaling 65,536 training datums. However, because the decision model does not generate multiple rollouts for group normalization, we would only obtain 1024 training datums per epoch, or 2048 total.

To get a sense of what is possible with decision models since each training step is considerably cheaper, we train the decision model for 1 epoch using all 3584 training questions across 128 steps. The exact configuration for the run is:

```bash
uv run -m tinker_cookbook.recipes.decisions.forecasting --model_name=<Qwen/Qwen3.8-27B | zai-org/GLM-5.3:peft:262144> --max_train_questions=3584 --batch_size=28 --epochs=1 --learning_rate=1e-4 --eval_every=16 --seed=0
```

### Results

The following results show training on the decision model using the full-dataset configuration (3584 training questions, 1 epoch) compared against the original recipe's configuration (1024 training questions, 2 epochs).

With Qwen3.8-27B:

| Step | Validation accuracy | Validation AUC | Brier reward (decision model) | Brier reward (original recipe) |
| ---: | ------------------: | -------------: | ----------------------------: | -----------------------------: |
|    0 |              70.31% |         0.7738 |                        0.7734 |                         0.7998 |
|   16 |              70.31% |         0.7859 |                        0.8108 |                         0.8166 |
|   32 |              74.41% |         0.8155 |                        0.8315 |                         0.8207 |
|   48 |              71.09% |         0.7953 |                        0.8203 |                         0.8222 |
|   64 |              71.88% |         0.8131 |                        0.8274 |                         0.8262 |
|   80 |              73.63% |         0.8013 |                        0.8260 |                         0.8202 |
|   96 |              73.05% |         0.8203 |                        0.8317 |                         0.8071 |
|  112 |              75.00% |         0.8312 |                    **0.8346** |                         0.8288 |
|  128 |              75.00% |         0.8112 |                        0.8219 |                     **0.8294** |

With GLM-5.3:

| Step | Validation accuracy | Validation AUC | Brier reward (decision model) | Brier reward (original recipe) |
| ---: | ------------------: | -------------: | ----------------------------: | -----------------------------: |
|    0 |              69.92% |         0.7859 |                        0.7430 |                         0.7952 |
|   16 |              71.48% |         0.8034 |                        0.8105 |                         0.8254 |
|   32 |              70.12% |         0.7987 |                        0.8215 |                         0.8224 |
|   48 |              74.22% |         0.8211 |                        0.8353 |                         0.8058 |
|   64 |              76.95% |         0.8405 |                        0.8360 |                         0.8413 |
|   80 |              69.14% |         0.8043 |                        0.8313 |                         0.8373 |
|   96 |              76.37% |         0.8438 |                        0.8424 |                         0.8309 |
|  112 |              79.30% |         0.8535 |                    **0.8457** |                         0.8387 |
|  128 |              74.41% |         0.8298 |                        0.8360 |                     **0.8475** |

The decision model trains very close to the original recipe's performance, despite typically starting at a lower baseline and with no additional test-time compute via reasoning.

To get a sense of how the decision model compares at a similar number of training questions, at step 32, the decision model used 896 training questions, comparable to step 64 of the original recipe that used 1024 training questions. In both cases the results are comparable, with the decision model outperforming the original recipe in the Qwen3.8-27B variant, while the original recipe outperforms the decision model in the GLM-5.3 variant.

## References

- [Forecasting recipe](../forecasting/): the original RL forecasting recipe on Prophet Arena that `forecasting.py` reproduces.
- [`SamplingClient.sample`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/#sample): Tinker API reference for the sampling call used by `sample_decision`. See `target_prompt_logprobs` for getting logprobs of a specific token at a given position.

## License

Prophet Arena data is distributed under the MIT license; see the
[dataset card](https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-1200)
for terms.
