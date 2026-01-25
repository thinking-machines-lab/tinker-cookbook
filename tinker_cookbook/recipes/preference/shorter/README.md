# Generating Shorter Responses via Comparisons

```bash
python -m tinker_cookbook.recipes.preference.shorter.train
```

`ac_tokens_per_turn` should drop significantly after 40 steps. The policy generates significantly shorter responses.

### Using the `PairwisePreferenceRLDatasetBuilder` class

We implement the `PairwisePreferenceRLDatasetBuilder` class to make it easier to learn from preference pairs, rather than scalar rewards for an individual trajectory. The key objects you need to implement are: (1) PreferenceModelBuilder, and (2) ComparisonBuilder.

**PreferenceModelBuilder** will build a *PreferenceModel* when called (via its `__call__()` method), which determines what responses are preferred. Concretely, `PreferenceModel.__call__`
- accepts a `Comparison` object, which contains (1) `prompt_conversation`: a list of input messages that the policy model receives, and (2) `completion_A` and `completion_B`, each a list of messages that the policy model generates.
- returns a floating point number; a larger number means that `completion_B` is better.

**ComparisonBuilder** will be used in our code to create a list of `Comparison` objects. We need to implement two functions
- `get_train_and_test_datasets`: which returns training and test Hugging Face `Dataset` objects
- `example_to_labeled_comparison`: which converts each datapoint (a `dict` in the `Dataset` object) to a `Comparison` object.

Note that `completion_A` and `completion_B` will NOT be used during training, and only `completion_A` will be used during `Eval`.

### Implementation of This Simple Example

We build a simple preference model that prefers shorter responses in `shorter.env.PreferenceModelShorter`. Then we implement a simple wrapper that builds it in `ShorterPreferenceModelBuilder`. To implement the `ComparisonBuilder` object, we create dummy datasets, where the `input_messages` field is a constant `Who are you?`.

### Next:

We will go through an example of the RLHF pipeline, which heavily relies on the `PairwisePreferenceRLDatasetBuilder` and `ComparisonBuilder` abstraction.
