"""SDPO loss computation.

From Proposition 2.1 of the paper, the gradient of the SDPO objective is:

    nabla L = E[ sum_t (log pi_student - log pi_teacher) * nabla log pi_student ]

We implement this as:

    loss = mean_t [ (student_lp - teacher_lp).detach() * student_lp ]

The .detach() on the log-ratio makes it act as a per-token advantage that
doesn't receive gradients, so only the student log-probs are differentiated.
"""

import tinker
import torch


def compute_sdpo_loss(
    data: list[tinker.Datum],
    student_logprobs_list: list[torch.Tensor],
    teacher_logprobs_list: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the SDPO token-level reverse-KL loss.

    Args:
        data: Student datums with "weights" in loss_fn_inputs (1 for response, 0 for prompt).
        student_logprobs_list: Per-token logprobs from forward_backward_custom (one tensor per datum).
        teacher_logprobs_list: Pre-computed teacher logprobs for response tokens only.

    Returns:
        (loss, metrics) tuple suitable for forward_backward_custom.
    """
    losses: list[torch.Tensor] = []
    log_ratio_sum = 0.0
    token_count = 0

    for datum, student_lps, teacher_response_lps in zip(
        data, student_logprobs_list, teacher_logprobs_list, strict=True
    ):
        weights = torch.tensor(
            datum.loss_fn_inputs["weights"].data, dtype=torch.float32
        )

        # Response tokens start where weights become 1.
        response_start = int((weights == 0).sum().item())
        response_len = int(weights.sum().item())

        student_response = student_lps[response_start : response_start + response_len]

        # Align lengths in case of minor truncation differences.
        min_len = min(len(student_response), len(teacher_response_lps))
        if min_len == 0:
            continue

        s = student_response[:min_len].float()
        t = teacher_response_lps[:min_len].float()

        # Per-token advantage: how much the student overestimates vs teacher.
        log_ratio = (s - t).detach()
        per_token_loss = log_ratio * s
        losses.append(per_token_loss.mean())

        log_ratio_sum += log_ratio.sum().item()
        token_count += min_len

    if not losses:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, {"sdpo/loss": 0.0, "sdpo/mean_log_ratio": 0.0}

    loss = torch.stack(losses).mean()
    metrics = {
        "sdpo/loss": loss.item(),
        "sdpo/mean_log_ratio": log_ratio_sum / max(token_count, 1),
    }
    return loss, metrics
