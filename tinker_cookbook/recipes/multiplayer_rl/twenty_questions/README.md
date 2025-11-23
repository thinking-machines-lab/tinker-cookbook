# Playing Twenty Questions Against Another Language Model

```bash
python -m tinker_cookbook.recipes.multiplayer_rl.twenty_questions.train
```

The `test/env/all/reward/total` should increase from ~10% to ~20% after 20 steps.

### Background: Twenty Questions

This game involves a *player* and an *answerer*. The answerer has a *secret word* in mind, and the player needs to ask yes/no questions to guess it. Here is an example where the secret word is apple:

> **Player**: Is it an animal?
> **Answerer**: No
>
> **Player**: Is it a plant?
> **Answerer**: Yes
>
> **Player**: Does it grow on a tree?
> **Answerer**: Yes
>
> **Player**: Guess: Apple
> **Answerer**: Yes

To use Tinker to train LLMs to play twenty questions, we mainly need to implement the `TwentyQuestionsEnv` class.

### Implementing a Training Environment

Each environment object has exactly one secret word; it determines what the conversation looks like based on the player’s (policy’s) questions (actions).

The most important logic happens within the class method `TwentyQuestionsEnv.step` in [env.py](./env.py). This `step` function takes in:

* An action from the language model, which is a sequence of int tokens that can be parsed into a natural language question, e.g., “`Is it a plant?`“

This function outputs a `StepResult`, which contains:

* Reward: 1 if the player guesses the keyword “Apple“, 0 otherwise. (see `TwentyQuestionsEnv._compute_reward(content)`)
* Whether this episode should end: either when the player correctly guesses the secret word Apple, or the player has asked more than 20 questions.
* next_stop_condition: it’s usually the stop token of the policy language model.
* next_observation: what the player (policy) sees in the next turn, in token space. In twenty questions, it involves the conversation history, the last question it asked, and the answer to the last question.

To automatically compute the next_observation — what the answerer would respond — we need to sample from another language model to play the role of answerer. We will go through it next.

### Sampling from Another Language Model in `Environment.step`

Here's what the answerer receives as input and produces as output:

> **Answerer Input:** “Answer yes/no questions about your secret word. Your secret word is apple. Question: Is it a plant?“
>
> **Answerer Output:** “Yes“.

The Environment class will be responsible for preparing the Input (see `_convo_for_answerer`), which will then be handed to the answerer for sampling. You can sample from:

* a third-party LLM library that you are already familiar with, or
* the `MessageCompleter` object that we have implemented, which is an API interface similar to OpenAI Chat Completion. You can sample from any model we support, or a model you have just trained (or are still training) on Tinker!

### Extensions

Our demo is simple, involving only one LLM other than the policy and a very simple data format (a single string as an answer). One can easily add more LLMs into this environment, e.g., using a language model as a judge and sampling from `judge_message_completer`. One can also easily extend it to more complicated datapoint types, including rubrics  [1] for LLM-judges [2], conversation starters, reference answers, or private information for human simulators [3, 4]. There are many other possibilities, and we look forward to seeing what you come up with using Tinker!

### Next

In this example, we play against a static language model answerer, which does not update during training.
In recipes.multiplayer_rl.text_arena, we will demonstrate an example (tic-tac-toe), which updates the weights of both players in a game.

[1] Checklists Are Better Than Reward Models For Aligning Language Models
Viswanathan, V., Sun, Y., Ma, S., Kong, X., Cao, M., Neubig, G., & Wu, T. (2025). Checklists are better than reward models for aligning language models. arXiv. https://arxiv.org/abs/2507.18624
[2] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-bench and chatbot arena. arXiv. https://arxiv.org/abs/2306.05685
[3] SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks
Zhou, Y., Jiang, S., Tian, Y., Weston, J., Levine, S., Sukhbaatar, S., & Li, X. (2025). SWEET-RL: Training multi-turn LLM agents on collaborative reasoning tasks. arXiv. https://arxiv.org/abs/2503.15478
[4] CollabLLM: From Passive Responders to Active Collaborators
Wu, S., Galley, M., Peng, B., Cheng, H., Li, G., Dou, Y., Cai, W., Zou, J., Leskovec, J., & Gao, J. (2025). CollabLLM: From passive responders to active collaborators. arXiv. https://arxiv.org/abs/2502.00640
