"""Datum construction for SDPO.

Builds datums compatible with tinker's ``importance_sampling`` loss (IS-only mode)
or ``forward_backward_custom`` (combined CE + IS mode). See ``sdpo/train.py``
for how these datums are used.
"""

from __future__ import annotations

import asyncio
import logging
from typing import cast

import tinker
import torch

from tinker_cookbook.rl.types import Trajectory
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)

logger = logging.getLogger(__name__)


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
    """Build a datum encoding the SDPO training signal (IS-only mode).

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
    """
    full_seq = build_full_sequence(ob, response_tokens)
    input_mi, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        list(full_seq.chunks)
    )

    prompt_len = ob.length
    response_len = len(response_tokens)

    # Logprobs: 0 for prompt positions, sampled logprobs for response positions.
    all_logprobs = [0.0] * (prompt_len - 1) + list(sampled_logprobs)
    all_logprobs = all_logprobs[: len(target_tokens)]

    # Advantages: teacher_lp - student_lp for response tokens, 0 for prompt.
    teacher_lps = teacher_logprobs[:response_len].tolist()
    student_lps = list(sampled_logprobs[:response_len])
    response_advantages = [t - s for t, s in zip(teacher_lps, student_lps)]
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


# ---------------------------------------------------------------------------
# Combined CE + IS datum construction for forward_backward_custom
# ---------------------------------------------------------------------------


def _extract_completion_tokens_from_datum(
    datum: tinker.Datum,
    teacher_prompt: tinker.ModelInput,
    max_context_length: int,
) -> tuple[list[int], int, int, bool]:
    """Extract student completion tokens and compute teacher prompt length.

    Returns (completion_tokens, teacher_prompt_len, completion_start, was_truncated).
    """
    mask = datum.loss_fn_inputs["mask"].to_torch()
    completion_mask_indices = torch.where(mask > 0)[0]
    teacher_prompt_len = teacher_prompt.length

    if len(completion_mask_indices) == 0:
        return [], teacher_prompt_len, 0, False

    student_full = datum.model_input.append_int(
        cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])
    )
    student_full_tokens = student_full.to_ints()
    completion_start = int(completion_mask_indices[0].item()) + 1
    completion_tokens = student_full_tokens[completion_start:]

    available = max_context_length - teacher_prompt_len
    if available <= 0:
        return [], teacher_prompt_len, completion_start, True
    truncated = len(completion_tokens) > available
    if truncated:
        completion_tokens = completion_tokens[:available]

    return completion_tokens, teacher_prompt_len, completion_start, truncated


async def build_sdpo_combined_datums(
    data_D: list[tinker.Datum],
    metadata_D: list[dict[str, int]],
    teacher_client: tinker.SamplingClient,
    teacher_prompts_P: list[tinker.ModelInput],
    topk: int = 20,
    max_context_length: int = 32768,
    vocab_size: int | None = None,
    skip_first_n_tokens: int = 0,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build datums for combined CE + IS SDPO loss via forward_backward_custom.

    Each datum has ``(N, K+1)``-shaped ``target_tokens``:
    - Columns ``0..K-1``: teacher's top-K token IDs
    - Column ``K``: the sampled token (for IS term)

    Additional loss_fn_inputs stored in each datum:
    - ``teacher_weights``: ``(N, K+1)`` — renormalized teacher probs for top-K,
      0 for the sampled-token column
    - ``is_advantages``: ``(N,)`` — teacher_lp(sampled) - student_lp(sampled)
    - ``is_old_logprobs``: ``(N,)`` — student logprobs from sampling time
    - ``mask``: ``(N,)`` — 1 for completion positions, 0 for prompt

    These are consumed by :func:`sdpo_combined_loss`.
    """
    K1 = topk + 1  # K top tokens + 1 sampled token

    # Step 1: Build teacher-forced sequences and extract completion info.
    teacher_forced_D: list[tinker.ModelInput] = []
    teacher_prompt_lens_D: list[int] = []
    completion_lens_D: list[int] = []
    truncated_count = 0

    for i, datum in enumerate(data_D):
        group_idx = metadata_D[i]["group_idx"]
        teacher_prompt = teacher_prompts_P[group_idx]

        comp_tokens, t_prompt_len, _, was_truncated = _extract_completion_tokens_from_datum(
            datum, teacher_prompt, max_context_length
        )
        if was_truncated:
            truncated_count += 1

        if not comp_tokens:
            teacher_forced_D.append(teacher_prompt)
            teacher_prompt_lens_D.append(t_prompt_len)
            completion_lens_D.append(0)
            continue

        # Build teacher-forced: teacher_prompt + completion tokens
        teacher_forced = teacher_prompt
        for tok in comp_tokens:
            teacher_forced = teacher_forced.append_int(tok)
        teacher_forced_D.append(teacher_forced)
        teacher_prompt_lens_D.append(t_prompt_len)
        completion_lens_D.append(len(comp_tokens))

    # Step 2: One sample_async call per datum gets both prompt_logprobs and topk.
    responses_D = await asyncio.gather(
        *[
            teacher_client.sample_async(
                prompt=tf,
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=1),
                include_prompt_logprobs=True,
                topk_prompt_logprobs=topk,
            )
            for tf in teacher_forced_D
        ]
    )

    # Step 3: Build (N, K+1) datums.
    new_datums: list[tinker.Datum] = []
    total_completion_tokens = 0
    total_teacher_entropy = 0.0
    total_advantage_sum = 0.0
    total_mask_sum = 0.0

    for i, datum in enumerate(data_D):
        mask = datum.loss_fn_inputs["mask"].to_torch()
        completion_mask_indices = torch.where(mask > 0)[0]
        N = datum.model_input.length
        completion_len = completion_lens_D[i]
        teacher_prompt_len = teacher_prompt_lens_D[i]

        target_tokens_NK1 = torch.zeros(N, K1, dtype=torch.long)
        teacher_weights_NK1 = torch.zeros(N, K1, dtype=torch.float32)
        is_advantages = torch.zeros(N, dtype=torch.float32)
        is_old_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()  # from sampling

        if completion_len > 0 and len(completion_mask_indices) > 0:
            resp = responses_D[i]
            topk_all = resp.topk_prompt_logprobs
            prompt_lps_all = resp.prompt_logprobs  # teacher logprob of each actual token

            num_tokens = min(completion_len, len(completion_mask_indices))
            for t in range(num_tokens):
                if t < skip_first_n_tokens:
                    continue

                teacher_pos = teacher_prompt_len + t
                student_pos = int(completion_mask_indices[t].item())

                # --- Top-K entries (for CE term) ---
                if topk_all is not None and teacher_pos < len(topk_all):
                    topk_entries = topk_all[teacher_pos]
                    if topk_entries is not None:
                        filtered = [
                            (tok_id, lp)
                            for tok_id, lp in topk_entries[:topk]
                            if vocab_size is None or tok_id < vocab_size
                        ]
                        if filtered:
                            k_actual = len(filtered)
                            tok_ids = torch.tensor(
                                [tid for tid, _ in filtered], dtype=torch.long
                            )
                            logps = torch.tensor(
                                [lp for _, lp in filtered], dtype=torch.float32
                            )
                            # Renormalize over top-K
                            logps -= torch.logsumexp(logps, dim=0)
                            probs = logps.exp()

                            target_tokens_NK1[student_pos, :k_actual] = tok_ids
                            teacher_weights_NK1[student_pos, :k_actual] = probs

                            total_teacher_entropy += -(probs * logps).sum().item()

                # --- Sampled token (for IS term) ---
                # The sampled token is the original target at this position.
                sampled_tok = int(datum.loss_fn_inputs["target_tokens"].data[student_pos])
                target_tokens_NK1[student_pos, topk] = sampled_tok

                # Teacher logprob of the sampled token.
                teacher_lp_sampled = 0.0
                if prompt_lps_all is not None and teacher_pos < len(prompt_lps_all):
                    lp_val = prompt_lps_all[teacher_pos]
                    if lp_val is not None:
                        teacher_lp_sampled = lp_val

                # IS advantage = teacher_lp(sampled) - student_lp(sampled)
                student_lp_sampled = float(is_old_logprobs[student_pos].item())
                is_advantages[student_pos] = teacher_lp_sampled - student_lp_sampled

                total_advantage_sum += is_advantages[student_pos].item()
                total_mask_sum += 1.0

            total_completion_tokens += num_tokens

        new_datum = tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(target_tokens_NK1),
                "teacher_weights": tinker.TensorData.from_torch(teacher_weights_NK1),
                "is_advantages": tinker.TensorData.from_torch(is_advantages),
                "is_old_logprobs": tinker.TensorData.from_torch(is_old_logprobs),
                "mask": tinker.TensorData.from_torch(mask),
            },
        )
        new_datums.append(new_datum)

    metrics: dict[str, float] = {
        "sdpo/teacher_truncated_count": float(truncated_count),
        "sdpo/num_datums": float(len(data_D)),
        "sdpo/topk": float(topk),
    }
    if total_mask_sum > 0:
        metrics["sdpo/mean_advantage"] = total_advantage_sum / total_mask_sum
    if total_completion_tokens > 0:
        metrics["sdpo/total_completion_tokens"] = float(total_completion_tokens)
        metrics["sdpo/mean_teacher_entropy"] = total_teacher_entropy / total_completion_tokens

    return new_datums, metrics


def sdpo_combined_loss(
    data: list[tinker.Datum],
    logprobs: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined SDPO loss: 0.5 * CE (forward KL) + 0.5 * IS (reverse KL).

    Called by ``forward_backward_custom``. Receives ``(N, K+1)``-shaped logprobs
    where columns ``0..K-1`` are the student's logprobs for the teacher's top-K
    tokens and column ``K`` is the student's logprob for the sampled token.

    The CE term pushes the student toward the teacher's top-K distribution.
    The IS term is a REINFORCE estimator for reverse KL, using the sampled token.
    """
    total_ce = torch.tensor(0.0)
    total_is = torch.tensor(0.0)

    for datum, lp in zip(data, logprobs):
        teacher_weights = datum.loss_fn_inputs["teacher_weights"].to_torch()  # (N, K+1)
        is_advantages = datum.loss_fn_inputs["is_advantages"].to_torch()  # (N,)
        is_old_logprobs = datum.loss_fn_inputs["is_old_logprobs"].to_torch()  # (N,)
        mask = datum.loss_fn_inputs["mask"].to_torch()  # (N,)

        K = teacher_weights.shape[1] - 1

        # CE term: -sum_k teacher_prob_k * student_logprob_k (forward KL)
        student_lp_topk = lp[:, :K]  # (N, K)
        teacher_w_topk = teacher_weights[:, :K]  # (N, K)
        ce_per_pos = -(teacher_w_topk * student_lp_topk).sum(dim=1)  # (N,)
        total_ce = total_ce + (ce_per_pos * mask).sum()

        # IS term: -ratio * advantage (reverse KL via REINFORCE)
        student_lp_sampled = lp[:, K]  # (N,) — current logprob of sampled token
        ratio = torch.exp(student_lp_sampled - is_old_logprobs)
        is_per_pos = -(ratio * is_advantages)  # (N,)
        total_is = total_is + (is_per_pos * mask).sum()

    loss = 0.5 * total_ce + 0.5 * total_is

    return loss, {
        "sdpo/ce_loss": total_ce.item(),
        "sdpo/is_loss": total_is.item(),
        "sdpo/combined_loss": loss.item(),
    }
