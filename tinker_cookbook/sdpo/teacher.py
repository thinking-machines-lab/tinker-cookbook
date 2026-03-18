"""Teacher prompt construction and logprob computation for SDPO.

The "teacher" in SDPO is the reference model conditioned on additional
information that the student doesn't have — a successful solution from
another rollout in the group, and/or environment feedback (e.g. compiler
errors) from the current trajectory. The difference between teacher and
student logprobs becomes the per-token advantage for training.
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
    reprompt_suffix: str,
    solution_text: str | None = None,
    feedback_text: str | None = None,
    solution_template: str = "Correct solution:\n\n{successful_previous_attempt}\n",
    feedback_template: str = (
        "The following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n"
    ),
) -> tinker.ModelInput:
    """Build a teacher prompt by conditioning on solutions and/or environment feedback.

    The teacher conversation is::

        [few-shot prefix]
        User: <question>
        Assistant: <content with solution and/or feedback>
        User: "Correctly solve the original question."
        Assistant: ...  <- model generates / scores from here

    The content can include:

    - **A successful solution** from another rollout in the group (the primary
      SDPO signal for verifiable tasks like math and MCQ).
    - **Environment feedback** from the current trajectory's execution (e.g.
      compiler errors, failing test cases). This is especially useful for code
      tasks where error messages provide rich diagnostic information.
    - **Both** — the paper (Table 6) shows these are complementary: solution
      alone gets 42.6%, feedback alone gets 39.9%, both together get 48.3%.

    The key idea: the reference model sees additional information in-context
    before scoring the response tokens, giving it an informational advantage
    over the student (which only sees the question).
    """
    # Assemble the conditioning content following the paper's template:
    # {prompt}{solution}{feedback}\n\nCorrectly solve the original question.
    content_parts: list[str] = []
    if solution_text is not None:
        content_parts.append(solution_template.format(successful_previous_attempt=solution_text))
    if feedback_text is not None:
        content_parts.append(feedback_template.format(feedback_raw=feedback_text))

    conditioning_content = "\n".join(content_parts) if content_parts else ""

    teacher_convo: list[renderers.Message] = env.convo_prefix + [
        {"role": "user", "content": env.get_question()},
    ]
    if conditioning_content:
        teacher_convo.append({"role": "assistant", "content": conditioning_content})
    teacher_convo.append({"role": "user", "content": reprompt_suffix})

    return env.renderer.build_generation_prompt(teacher_convo)


def build_teacher_prompt_from_messages(
    convo_prefix: list[renderers.Message],
    question: str,
    renderer: renderers.Renderer,
    reprompt_suffix: str,
    solution_text: str | None = None,
    feedback_text: str | None = None,
    solution_template: str = "Correct solution:\n\n{successful_previous_attempt}\n",
    feedback_template: str = (
        "The following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n"
    ),
) -> tinker.ModelInput:
    """Build a teacher prompt from raw conversation components.

    This is a more generic version of ``build_teacher_prompt`` that doesn't
    require a ``ProblemEnv``. It works with any environment type (including
    tool-use environments like code_rl) as long as you supply the conversation
    prefix, question, and renderer.
    """
    content_parts: list[str] = []
    if solution_text is not None:
        content_parts.append(solution_template.format(successful_previous_attempt=solution_text))
    if feedback_text is not None:
        content_parts.append(feedback_template.format(feedback_raw=feedback_text))

    conditioning_content = "\n".join(content_parts) if content_parts else ""

    teacher_convo: list[renderers.Message] = list(convo_prefix) + [
        {"role": "user", "content": question},
    ]
    if conditioning_content:
        teacher_convo.append({"role": "assistant", "content": conditioning_content})
    teacher_convo.append({"role": "user", "content": reprompt_suffix})

    return renderer.build_generation_prompt(teacher_convo)


async def compute_teacher_logprobs(
    reference_client: tinker.SamplingClient,
    teacher_ob: tinker.ModelInput,
    response_tokens: list[int],
    max_context_length: int = 32768,
) -> torch.Tensor:
    """Score response tokens using the reference model conditioned on the teacher prompt.

    Sends the full sequence (teacher_prompt + response) to the reference model,
    then extracts only the logprobs for response token positions. These represent
    "how likely is each response token if you already know the solution?"

    If the teacher prompt + response exceeds ``max_context_length``, the response
    is truncated from the end so that the sequence fits. This can happen because
    the teacher prompt is longer than the student prompt (it includes the solution
    and/or environment feedback). The returned tensor will be shorter than
    ``len(response_tokens)`` in this case; the caller (``build_sdpo_datum``)
    already handles mismatched lengths via slicing.

    Returns a tensor of length ``min(len(response_tokens), available_space)``.
    """
    teacher_prompt_len = teacher_ob.length
    available_for_response = max_context_length - teacher_prompt_len
    if available_for_response <= 0:
        # Teacher prompt alone exceeds context — return empty tensor.
        # build_sdpo_datum will pad advantages with 0.0, so this trajectory
        # contributes no gradient.
        return torch.tensor([], dtype=torch.float32)
    truncated_response = response_tokens[:available_for_response]

    teacher_full = build_full_sequence(teacher_ob, truncated_response)
    all_logprobs = await reference_client.compute_logprobs_async(teacher_full)
    # Extract only the logprobs for response token positions.
    # compute_logprobs[i] = log P(token_i | tokens_0..i-1)
    # The first element may be None (no conditioning context); response
    # positions start at teacher_prompt_len which is always >= 1, so we
    # won't encounter None, but we guard defensively.
    response_logprobs = [lp if lp is not None else 0.0 for lp in all_logprobs[teacher_prompt_len:]]
    return torch.tensor(response_logprobs, dtype=torch.float32)
