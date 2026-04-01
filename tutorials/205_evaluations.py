import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 205: Evaluations

    Tinker's evaluation system uses two abstract classes:

    - **`TrainingClientEvaluator`** -- uses the training client (forward passes) to compute metrics like NLL
    - **`SamplingClientEvaluator`** -- uses the sampling client (generation) to compute metrics like accuracy

    Both return `dict[str, float]` and plug into `train.Config.evaluator_builders` for automatic evaluation during training.

    In this tutorial you will:

    1. Implement a `TrainingClientEvaluator` that computes NLL on held-out data
    2. Implement a `SamplingClientEvaluator` that samples answers and checks correctness
    3. Wire evaluators into `train.Config` via `evaluator_builders`
    4. Learn about the Inspect AI integration for standardized benchmarks
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import tinker
    import torch
    from tinker import TensorData

    from tinker_cookbook.eval.evaluators import (
        SamplingClientEvaluator,
        TrainingClientEvaluator,
    )
    from tinker_cookbook.renderers import get_renderer, get_text_content

    return (
        SamplingClientEvaluator,
        TensorData,
        TrainingClientEvaluator,
        get_renderer,
        get_text_content,
        tinker,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The evaluator pattern

    Both evaluator types are async callables with a simple contract:

    ```python
    class TrainingClientEvaluator:
        async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
            ...

    class SamplingClientEvaluator:
        async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
            ...
    ```

    The training loop calls your evaluator periodically and logs the returned metrics. The keys become metric names (e.g., `"eval/nll"`, `"eval/accuracy"`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Create a training client and prepare some evaluation data.
    """)
    return


@app.cell
async def _(TensorData, get_renderer, tinker, torch):
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=16
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer("qwen3_instruct", tokenizer)

    # Prepare held-out SFT data for the NLL evaluator
    eval_examples = [
        "The speed of light is approximately 3 * 10^8 meters per second.",
        "Water freezes at 0 degrees Celsius under standard pressure.",
        "The Earth orbits the Sun once every 365.25 days.",
    ]

    eval_datums = []
    for text in eval_examples:
        ids = tokenizer.encode(text)
        model_input = tinker.ModelInput.from_ints(ids[:-1])
        target_tokens = ids[1:]
        w = [1.0] * len(target_tokens)
        eval_datums.append(
            tinker.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "weights": TensorData.from_torch(torch.tensor(w)),
                },
            )
        )

    print(f"Prepared {len(eval_datums)} evaluation datums")
    return (
        MODEL_NAME,
        eval_datums,
        eval_examples,
        renderer,
        service_client,
        tokenizer,
        training_client,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementing a TrainingClientEvaluator: NLL

    A `TrainingClientEvaluator` receives the current `TrainingClient` and can run forward passes to compute metrics. Here we compute the mean negative log-likelihood (NLL) on held-out data -- a standard measure of how well the model predicts the evaluation text.
    """)
    return


@app.cell
def _(TrainingClientEvaluator, tinker, torch):
    class NLLEvaluator(TrainingClientEvaluator):
        """Compute mean NLL on held-out data using forward passes."""

        def __init__(self, eval_data: list[tinker.Datum], name: str = "eval"):
            self.eval_data = eval_data
            self.name = name

        async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
            # Run a forward pass (no gradients) to get logprobs
            future = await training_client.forward_async(self.eval_data, loss_fn="cross_entropy")
            result = await future.result_async()

            # Compute weighted mean NLL
            total_nll = 0.0
            total_tokens = 0
            for datum, output in zip(self.eval_data, result.loss_fn_outputs):
                logprobs = torch.tensor(output["logprobs"])
                weights = torch.tensor(datum.loss_fn_inputs["weights"])
                total_nll += -(logprobs * weights).sum().item()
                total_tokens += weights.sum().item()

            mean_nll = total_nll / max(total_tokens, 1)
            return {f"{self.name}/nll": mean_nll}

    return (NLLEvaluator,)


@app.cell
async def _(NLLEvaluator, eval_datums, training_client):
    # Test the evaluator
    nll_evaluator = NLLEvaluator(eval_datums, name="held_out")
    nll_metrics = await nll_evaluator(training_client)
    print(f"NLL evaluation: {nll_metrics}")
    return nll_evaluator, nll_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementing a SamplingClientEvaluator: Accuracy

    A `SamplingClientEvaluator` receives a `SamplingClient` and generates text to compute metrics. Here we sample answers to simple factual questions and check if they contain the expected answer.
    """)
    return


@app.cell
def _(SamplingClientEvaluator, get_text_content, tinker):
    class AccuracyEvaluator(SamplingClientEvaluator):
        """Sample answers and check if they contain the expected string."""

        def __init__(self, questions_and_answers, renderer, tokenizer):
            self.qa_pairs = questions_and_answers
            self.renderer = renderer
            self.tokenizer = tokenizer

        async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
            correct = 0

            for question, expected_answer in self.qa_pairs:
                messages = [{"role": "user", "content": question}]
                prompt = self.renderer.build_generation_prompt(messages)
                stop = self.renderer.get_stop_sequences()

                result = await sampling_client.sample_async(
                    prompt=prompt,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=64, temperature=0.0, stop=stop
                    ),
                    num_samples=1,
                )

                tokens = result.sequences[0].tokens
                parsed, _ = self.renderer.parse_response(tokens)
                response_text = get_text_content(parsed).lower()

                if expected_answer.lower() in response_text:
                    correct += 1

            accuracy = correct / len(self.qa_pairs)
            return {"eval/accuracy": accuracy, "eval/correct": float(correct)}

    return (AccuracyEvaluator,)


@app.cell
async def _(AccuracyEvaluator, renderer, service_client, tokenizer, training_client):
    # Create a sampling client from current weights
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # Define test questions
    test_qa = [
        ("What is the capital of Japan?", "tokyo"),
        ("What is 15 + 27?", "42"),
        ("What element has the symbol 'O'?", "oxygen"),
    ]

    accuracy_evaluator = AccuracyEvaluator(test_qa, renderer, tokenizer)
    acc_metrics = await accuracy_evaluator(sampling_client)
    print(f"Accuracy evaluation: {acc_metrics}")
    return acc_metrics, accuracy_evaluator, sampling_client, test_qa


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wiring evaluators into train.Config

    The `train.Config` classes in `tinker_cookbook.supervised.train` and `tinker_cookbook.rl.train` accept an `evaluator_builders` parameter. Each builder is a zero-argument callable that returns an evaluator instance.

    The training loop calls each builder once at startup, then runs the evaluators periodically (controlled by `eval_every`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Supervised training with evaluators

    ```python
    from tinker_cookbook.supervised import train

    def make_nll_evaluator():
        # Build eval data here (or capture it from outer scope)
        return NLLEvaluator(eval_datums, name="validation")

    def make_accuracy_evaluator():
        return AccuracyEvaluator(test_qa, renderer, tokenizer)

    config = train.Config(
        log_path="~/logs/sft-with-evals",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        dataset_builder=my_dataset_builder,
        learning_rate=1e-4,

        # Evaluator builders -- called once at startup, run every eval_every steps
        evaluator_builders=[make_nll_evaluator, make_accuracy_evaluator],
        eval_every=50,  # run evaluators every 50 steps
    )
    ```

    The SFT training loop detects the evaluator type automatically:
    - `TrainingClientEvaluator` is called with the training client
    - `SamplingClientEvaluator` is called with a sampling client created from current weights
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RL training with evaluators

    The RL `train.Config` works the same way, though it only accepts `SamplingClientEvaluatorBuilder`:

    ```python
    from tinker_cookbook.rl import train

    config = train.Config(
        log_path="~/logs/rl-with-evals",
        model_name="meta-llama/Llama-3.1-8B",
        dataset_builder=my_rl_dataset_builder,

        evaluator_builders=[make_accuracy_evaluator],
        eval_every=10,
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Built-in evaluator: NLLEvaluator

    The cookbook ships a production-ready NLL evaluator at `tinker_cookbook.supervised.nll_evaluator.NLLEvaluator`. It can be constructed from a `SupervisedDataset`:

    ```python
    from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator

    # From a dataset object
    nll_eval = NLLEvaluator.from_dataset(eval_dataset, name="test")

    # Or from raw datums
    nll_eval = NLLEvaluator(data=eval_datums, name="validation")
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect AI integration

    For standardized benchmarks (MMLU, GSM8K, HumanEval, etc.), the cookbook integrates with [Inspect AI](https://inspect.ai). The `InspectEvaluator` wraps any Inspect task as a `SamplingClientEvaluator`:

    ```python
    from tinker_cookbook.eval.run_inspect_evals import Config

    # Run Inspect evals standalone
    config = Config(
        model_path="tinker://run-id/sampler_weights/final",
        # ... Inspect task configuration ...
    )
    ```

    Inspect AI provides a large library of pre-built evaluation tasks, so you can benchmark your fine-tuned model against established benchmarks without writing custom evaluation code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Evaluator type | Receives | Typical metrics | Example |
    |---|---|---|---|
    | `TrainingClientEvaluator` | `TrainingClient` | NLL, perplexity | Forward pass on held-out data |
    | `SamplingClientEvaluator` | `SamplingClient` | Accuracy, reward | Generate and grade answers |

    **Key points:**
    - Evaluators are async callables returning `dict[str, float]`
    - Wire them into training via `evaluator_builders` (list of zero-arg factories)
    - `eval_every` controls how often they run
    - Use `TrainingClientEvaluator` for metrics that only need a forward pass (fast)
    - Use `SamplingClientEvaluator` for metrics that need generation (slower but more informative)
    - For standard benchmarks, use the Inspect AI integration
    """)
    return


if __name__ == "__main__":
    app.run()
