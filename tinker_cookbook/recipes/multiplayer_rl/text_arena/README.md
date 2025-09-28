# Multi-Agent Training

Many research studies involve training several different language model agents jointly. We cover one simple example, where the language model learns to play tic-tac-toe with itself.
We show how to coordinate the steps of two *Environment* objects such that both the winning and the losing trajectory will be used to fine-tune the weights.

### Code
```bash
python -m recipes.multiplayer_rl.text_arena.train
```

You should reach XX performance after YY steps.


### Background

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

### Extension

Multi-agent training is a very active research direction with many different algorithm choices, e.g., debate [2], prover-verifier games [3], etc.
We hope Tinker can support the broader research community to explore these opportunities!



### References

[1] Guertler, L., Cheng, B., Yu, S., Liu, B., Choshen, L., & Tan, C. (2025). *TextArena*. arXiv preprint arXiv:2504.11442. https://arxiv.org/abs/2504.11442
[2] Khan, A., Hughes, J., Valentine, D., Ruis, L., Sachan, K., Radhakrishnan, A., Grefenstette, E., Bowman, S. R., Rocktäschel, T., & Perez, E. (2024). Debating with More Persuasive LLMs Leads to More Truthful Answers. Proceedings of Machine Learning Research, 235, 23662-23733.
[3] Kirchner, J. H., Chen, Y., Edwards, H., Leike, J., McAleese, N., & Burda, Y. (2024). Prover-verifier games improve legibility of LLM outputs. arXiv preprint arXiv:2407.13692. https://arxiv.org/abs/2407.13692
