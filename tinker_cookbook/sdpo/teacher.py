"""Teacher prompt construction and logprob computation for SDPO.

The "teacher" in SDPO is the reference model conditioned on a successful
solution. By prepending the solution to the prompt, the teacher sees the
answer in-context before scoring each response token. The difference between
teacher and student logprobs (teacher can see the answer, student cannot)
becomes the per-token advantage for training.
"""

import re

import tinker
import torch

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv
from tinker_cookbook.sdpo.data import build_full_sequence


def strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_teacher_prompt(
    env: ProblemEnv,
    solution_text: str,
    reprompt_suffix: str,
) -> tinker.ModelInput:
    """Build a teacher prompt by conditioning on a successful solution.

    The teacher conversation is::

        [few-shot prefix]
        User: <question>
        Assistant: <successful solution>
        User: "Correctly solve the original question."
        Assistant: ...  <- model generates / scores from here

    The key idea: the reference model sees the correct answer in-context
    before it scores the response tokens, giving it an informational
    advantage over the student (which only sees the question).
    """
    teacher_convo: list[renderers.Message] = env.convo_prefix + [
        {"role": "user", "content": env.get_question()},
        {"role": "assistant", "content": solution_text},
        {"role": "user", "content": reprompt_suffix},
    ]
    return env.renderer.build_generation_prompt(teacher_convo)


async def compute_teacher_logprobs(
    reference_client: tinker.SamplingClient,
    teacher_ob: tinker.ModelInput,
    response_tokens: list[int],
) -> torch.Tensor:
    """Score response tokens using the reference model conditioned on the teacher prompt.

    Sends the full sequence (teacher_prompt + response) to the reference model,
    then extracts only the logprobs for response token positions. These represent
    "how likely is each response token if you already know the solution?"

    Returns a tensor of length ``len(response_tokens)``.
    """
    teacher_full = build_full_sequence(teacher_ob, response_tokens)
    all_logprobs = await reference_client.compute_logprobs_async(teacher_full)
    # Extract only the logprobs for response token positions.
    # compute_logprobs[i] = log P(token_i | tokens_0..i-1)
    # The first element may be None (no conditioning context); response
    # positions start at teacher_prompt_len which is always >= 1, so we
    # won't encounter None, but we guard defensively.
    teacher_prompt_len = teacher_ob.length
    response_logprobs = [lp if lp is not None else 0.0 for lp in all_logprobs[teacher_prompt_len:]]
    return torch.tensor(response_logprobs, dtype=torch.float32)
