"""Datum construction for SDPO.

Builds datums compatible with tinker's ``importance_sampling`` loss. Each datum
contains per-token advantages (teacher_lp - student_lp) that encode the SDPO
training signal. See ``sdpo/train.py`` for how these datums are used.
"""

import tinker
import torch

from tinker_cookbook.rl.types import Trajectory
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)


def extract_response_tokens(traj: Trajectory) -> list[int]:
    """Extract all action tokens from a trajectory."""
    tokens: list[int] = []
    for transition in traj.transitions:
        tokens.extend(transition.ac.tokens)
    return tokens


def extract_response_logprobs(traj: Trajectory) -> list[float]:
    """Extract sampled logprobs for all action tokens from a trajectory."""
    logprobs: list[float] = []
    for transition in traj.transitions:
        logprobs.extend(transition.ac.logprobs)
    return logprobs


def extract_feedback(traj: Trajectory) -> str | None:
    """Extract environment feedback from a trajectory's logs.

    Looks for a ``"feedback"`` key in the last transition's logs. Environments
    that produce diagnostic output (e.g. compiler errors, failing test cases)
    should store it under this key for SDPO to use as a conditioning signal.
    """
    if not traj.transitions:
        return None
    last_logs = traj.transitions[-1].logs
    feedback = last_logs.get("feedback")
    if isinstance(feedback, str) and feedback.strip():
        return feedback.strip()
    return None


def build_full_sequence(ob: tinker.ModelInput, response_tokens: list[int]) -> tinker.ModelInput:
    """Append response tokens to an observation ModelInput."""
    chunks = list(ob.chunks) + [tinker.EncodedTextChunk(tokens=response_tokens)]
    return tinker.ModelInput(chunks=chunks)


def build_sdpo_datum(
    ob: tinker.ModelInput,
    response_tokens: list[int],
    sampled_logprobs: list[float],
    teacher_logprobs: torch.Tensor,
) -> tinker.Datum:
    """Build a datum encoding the SDPO training signal.

    The datum is structured for ``forward_backward(..., loss_fn="importance_sampling")``:

      - **target_tokens**: Next-token prediction targets (left-shifted).
      - **logprobs**: Student's sampled logprobs. The importance_sampling loss uses
        these to compute the importance weight ``exp(current_lp - sampled_lp)``,
        which corrects for any drift between the current policy and the sampling
        policy.
      - **advantages**: ``teacher_lp - student_lp`` for response tokens, 0 for
        prompt tokens. This is the SDPO signal — tokens where the teacher (which
        can see the solution) is more confident get positive advantage.

    Prompt positions have logprobs=0 and advantages=0, so they don't contribute
    to the loss.

    Args:
        ob: The prompt ModelInput (observation from rollout).
        response_tokens: Generated response tokens.
        sampled_logprobs: Student logprobs at sampling time (one per response token).
        teacher_logprobs: Reference model logprobs under teacher prompt (one per response token).
    """
    full_seq = build_full_sequence(ob, response_tokens)
    input_mi, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        list(full_seq.chunks)
    )

    prompt_len = ob.length
    response_len = len(response_tokens)

    # Logprobs: 0 for prompt positions, sampled logprobs for response positions.
    # Length = prompt_len + response_len - 1 (same as target_tokens).
    all_logprobs = [0.0] * (prompt_len - 1) + list(sampled_logprobs)
    all_logprobs = all_logprobs[: len(target_tokens)]

    # Advantages: teacher_lp - student_lp for response tokens, 0 for prompt.
    # This encodes the SDPO signal: push student toward teacher.
    # teacher_logprobs may be shorter than response_len if the teacher prompt
    # + response exceeded the model's context window and was truncated.
    # Positions beyond the teacher's coverage get advantage=0 (no gradient).
    teacher_lps = teacher_logprobs[:response_len].tolist()
    student_lps = list(sampled_logprobs[:response_len])
    response_advantages = [t - s for t, s in zip(teacher_lps, student_lps)]
    # Pad to response_len so all positions are covered.
    response_advantages += [0.0] * (response_len - len(response_advantages))
    all_advantages = [0.0] * (prompt_len - 1) + response_advantages
    all_advantages = all_advantages[: len(target_tokens)]

    return tinker.Datum(
        model_input=input_mi,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=target_tokens, dtype="int64", shape=[len(target_tokens)]
            ),
            "logprobs": tinker.TensorData(
                data=all_logprobs, dtype="float32", shape=[len(all_logprobs)]
            ),
            "advantages": tinker.TensorData(
                data=all_advantages, dtype="float32", shape=[len(all_advantages)]
            ),
        },
    )
