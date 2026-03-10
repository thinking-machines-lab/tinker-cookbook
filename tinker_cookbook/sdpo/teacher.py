"""Teacher prompt construction and logprob computation for SDPO."""

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
    """Build a teacher prompt following the paper's format.

    The teacher conversation is:
      [few-shot prefix] + user question + assistant solution + user re-prompt
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
    """Compute reference model logprobs for response tokens under the teacher prompt.

    Returns a tensor of length len(response_tokens) containing the log probability
    of each response token conditioned on the teacher prompt and preceding response
    tokens.
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
