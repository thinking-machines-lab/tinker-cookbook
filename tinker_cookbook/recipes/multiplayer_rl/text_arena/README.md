# Learning Tic-Tac-Toe via Self-Play

## Installation

```bash
uv pip install 'tinker-cookbook[multiplayer-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Many research studies involve training several different language model agents jointly. We cover one simple example, where the language model learns to play tic-tac-toe with itself.
We show how to coordinate the steps of two *Environment* objects such that both the winning and the losing trajectory will be used to fine-tune the weights.

## Training

```bash
python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.train
```

With the default settings (batch size 512, ~40k datapoints = 80 steps), training takes roughly 1-2 hours.

### Expected results

The key metric to watch is `test/env/all/reward/total`, which measures how well the trained policy performs against the (untrained) base model.

| Step | Train reward (self-play) | Test reward (vs base model) |
|------|--------------------------|-----------------------------|
| 0    | ~ -0.4                   | ~ -0.4                      |
| 5    | ~ 0.0                    | ~ +0.3                      |
| 20   | ~ 0.0                    | ~ +0.2                      |
| 40   | ~ 0.0                    | ~ +0.5 to +0.7              |
| 60   | ~ 0.0                    | ~ +0.4 to +0.6              |

**Why is a self-play reward of 0 good?** In tic-tac-toe, optimal play by both sides always results in a draw. The self-play training reward converging to 0 means both the "Player 0" and "Player 1" copies of the model have learned to play without making mistakes -- neither side can win. The test reward (vs the untrained base model) rises from negative to positive, meaning the trained model dominates the base model.

### Self-play dynamics over 256 steps

Running for the full 256 steps reveals interesting training dynamics:

| Phase | Steps | Self-play reward | Test reward | Description |
|-------|-------|-----------------|-------------|-------------|
| Learning | 0-5 | -0.4 → 0.0 | -0.4 → +0.3 | Model learns basic play |
| Plateau | 5-75 | ~0.0 | +0.3 to +0.7 | Stable, strong play |
| Collapse | 80-110 | **-1.0** | +0.4 → -0.1 | One side wins every game |
| Recovery | 115-130 | 0.0 | -0.2 → +0.6 | Self-play stabilizes again |

The **collapse** at step 80 happens because a small asymmetry gets amplified — if one side becomes slightly stronger, the training update reinforces that advantage, creating a feedback loop that destabilizes within a few steps. This is a known challenge with self-play RL.

The model **recovers** on its own around step 115, but the instability wastes training compute. The default of 80 steps avoids this region while capturing the performance plateau.

Potential mitigations for longer training runs include mixing self-play with games against a fixed opponent (random or base model) in each batch, or periodically freezing a copy of the weights to play against.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `model_name` | `Qwen/Qwen3-4B-Instruct-2507` | Base model to train |
| `batch_size` | `512` | Trajectories per training step |
| `num_train_datapoints` | `40960` | Total training trajectories (~80 steps) |
| `learning_rate` | `3e-5` | Adam learning rate |
| `eval_every` | `5` | Evaluate every N steps |
| `save_every` | `20` | Checkpoint every N steps |
| `test_opponent` | `base_model` | Test opponent: `base_model`, `random`, or `optimal` |
| `max_steps` | `None` | Cap training steps (None = use all data) |

## Evaluation

After training, evaluate the model against different opponents:

```bash
# Evaluate against the optimal (minimax) opponent — perfect play always draws:
python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
    checkpoint_path=<path> mode=eval opponent=optimal num_games=20

# Evaluate against a random opponent:
python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
    checkpoint_path=<path> mode=eval opponent=random num_games=20

# Evaluate against the (untrained) base model:
python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
    checkpoint_path=<path> mode=eval opponent=base_model num_games=20
```

Results before and after training (20 games each, model plays as Player 0):

| Opponent | Base model (step 0) | Trained (step 40) |
|----------|---------------------|-------------------|
| vs Random | 4W / 10D / 6L | 6W / 11D / 3L |
| vs Optimal (minimax) | 0W / 0D / **20L** | 0W / **20D** / 0L |

The untrained base model loses every game against the optimal minimax opponent and frequently loses to random. After 40 steps of self-play training, the model draws every game against optimal — confirming it learned perfect tic-tac-toe. The improvement is most dramatic against optimal: from 0% survival to 100% draws.

## Interactive play

You can also play against the trained model interactively (requires a real terminal):

```bash
# Play against the model (you go first as Player 0):
python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
    checkpoint_path=<path>

# Play as Player 1 (model goes first):
python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
    checkpoint_path=<path> human_player_id=1
```

## Background

### TextArena

The TextArena [1] already implements an environment object where two players can specify which position to play using ``[0], [1], [2] ...`` in tic-tac-toe and compute how the board changes, the observation (prompt) for each language model player, and the final reward.

Here's an example language model input:
```
[GAME] You are Player 0 in Tic Tac Toe.
Your goal is to win three in a row (horizontally, vertically, or diagonally) on the board.
On your turn, you should select the square number (0-8) you want to put your mark in next.
For example, '[4]' places your mark in the center cell of the board.

As Player 0, you will be 'O', while your opponent is 'X'.

[GAME] Current Board:

 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8

Available Moves: '[0]', '[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]'
```

If the language model wants to play in the middle, it can output `[4]`.

### Coordinators

Training an LLM to play against a fixed LLM is straightforward -- this is quite similar to our twenty-question example, where we sample the response from another LLM in the `Env.step` function.
However, in this example, we want to train on both trajectories where the language model plays on each side.
Therefore, in the `Env.step` function, we need to receive the opponent's action, which is generated in another trajectory in another `Environment` object.
This motivates the design of the `Coordinator` class, which passes the LLM-generated text between two `Environment` objects and synchronizes the two `Environment` objects to alternate taking steps.

In our implementation, the `TwoPlayerCoordinator` object is shared across two `Environment` objects, and it:
- wraps the tic-tac-toe environment from the TextArena [1],
- waits for a specific player's turn to begin, and
- allows one player to `make_move` on the board, and notifies the other player that the move is complete.

As a result, in the `Environment.step` function, we can:
- determine when to start the next move, since `TwoPlayerCoordinator` informs us when the opponent has finished.
- compute the next observation, since `TwoPlayerCoordinator` passes the move from the opponent.

### Opponent modes

The recipe supports different opponent types for evaluation:

- **`random`**: Picks a random legal move each turn. A well-trained model should rarely lose. No API calls needed.
- **`optimal`**: Perfect minimax player. Optimal vs optimal always draws, so a model that consistently draws has learned perfect play. No API calls needed.
- **`base_model`** (default for training eval): The trained model plays against the untrained base model. Measures improvement relative to the starting weights. Requires a second Tinker sampling session.

### Extension

Multi-agent training is a very active research direction with many different algorithm choices, e.g., debate [2], prover-verifier games [3], etc.
We hope Tinker can support the broader research community to explore these opportunities!

### References

[1] Guertler, L., Cheng, B., Yu, S., Liu, B., Choshen, L., & Tan, C. (2025). *TextArena*. arXiv preprint arXiv:2504.11442. https://arxiv.org/abs/2504.11442
[2] Khan, A., Hughes, J., Valentine, D., Ruis, L., Sachan, K., Radhakrishnan, A., Grefenstette, E., Bowman, S. R., Rocktäschel, T., & Perez, E. (2024). Debating with More Persuasive LLMs Leads to More Truthful Answers. Proceedings of Machine Learning Research, 235, 23662-23733.
[3] Kirchner, J. H., Chen, Y., Edwards, H., Leike, J., McAleese, N., & Burda, Y. (2024). Prover-verifier games improve legibility of LLM outputs. arXiv preprint arXiv:2407.13692. https://arxiv.org/abs/2407.13692
