`tinker_cookbook` shows how to use the Tinker API for common fine-tuning operations. It is intended to be readable and hackable, but also reasonably performant.

Here are some principles for the how the code is designed and organized. The codebase does not yet fully adhere to these principles, so if you see parts of the code that don't follow them, please fix them.

## Organization of training scripts

We're designing the codebase with the following goals:

1. Low barrier to entry: it should be dead simple to run something and see numbers go up. 
2. Extensible: it should be possible to pass in custom datasets and evals and control all the hyperparameters.
3. Science-friendly: it should be easy to run sweeps, and analyze the results.

To achieve this, we'll use the following structure around training scripts:

- There's a main training function, such as `sft.py` or `rl_bandit/train.py`, which contains the main loop.
    - This function contains a detailed config object (`Config`), which isn't constructable from the command line.
    - The config contains members that specify things like datasets and evals. However, these should be chz configs (with a `.build` method that constructs the actual object) or callables (we recommend using functools.partial). This way, the config is serializable, which is useful for sweeps.
- There's an auxiliary script, called something like `sft_cli.py` or `rl_bandit/train_cli.py`, which contains a smaller config object (`CLIConfig`), which is constructable from the command line. This script is useful to let people get started with the library, without digging into a lot of code and learning about new classes.

## Async

Async is very useful for RL, where it allows us to make many queries in parallel (e.g., sampling calls). For all of the interfaces used in RL (such as the `Env` class), all the methods that take nontrivial amounts of time should be async. For some of the other code, such as `sft.py`, we've chosen not to use async methods, just to make it more beginner-friendly, as many python programmers are not familiar with async.

## Typing

Please use typing wherever possible; avoid `Any` and `type: ignore`; prefer casting. However, avoid using convoluted generics or writing code that's much more verbose just to satisfy the type checker. Prefer using single types over union types.

## See also

Look at the [Notation section](README.md#notation) in the README for more details on the index notation used throughout the codebase.