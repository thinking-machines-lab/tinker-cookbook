# Multiturn Training

Often we not only want large language models (LLMs) to generate a single response, but also to perform well across multiple turns of interaction.
To help Tinker users easily customize their own training, we provide the *Environment* abstraction.

We cover three examples, with increasing complexity.

1. [Guess the number](./guess_number/): where the policy learns to guess the target number with multiple tries, given feedback on whether the number is too high or low.
2. [Twenty Questions](./twenty_questions): where the policy learns to guess an underlying object by asking yes/no questions.
3. [Tic-Tac-Toe](./text_arena): where the policy learns by playing against itself.

The first example is the simplest, since the user turn can be programmed with simple python statements.
The second is more complicated, since we need a language model to answer yes/no questions to the policy.
The third one is the most complicated, since we need to train multiple LLMs at the same time.
Fortunately, our `Environment` abstraction can handle all of them, and we will show examples of how to implement each one.
