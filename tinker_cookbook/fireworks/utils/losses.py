"""Loss functions for DPO, ORPO, and SFT training.

RL losses (GRPO, DAPO, GSPO) live in ``utils.rl``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Callable

import torch
import tinker
import torch.nn.functional as F


def _zero_loss(logprobs_list: List[torch.Tensor]) -> torch.Tensor:
    if logprobs_list:
        return logprobs_list[0].new_tensor(0.0)
    return torch.tensor(0.0)


def _validate_microbatch_sizes(
    total_items: int,
    microbatch_sizes: List[int] | None,
) -> List[int] | None:
    if microbatch_sizes is None:
        return None
    if not microbatch_sizes:
        raise ValueError("microbatch_sizes must not be empty when provided.")
    if any(size <= 0 for size in microbatch_sizes):
        raise ValueError(f"microbatch_sizes must be positive, got {microbatch_sizes}")
    if sum(microbatch_sizes) != total_items:
        raise ValueError(
            f"microbatch_sizes must sum to {total_items}, got {sum(microbatch_sizes)}"
        )
    return microbatch_sizes


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x < 0.

    Uses two branches for stability:
      - x < -log(2): log1p(-exp(x))      (exp(x) is small)
      - x >= -log(2): log(-expm1(x))     (1 - exp(x) is small)
    """
    mask = x < -0.6931471805599453  # -log(2)
    return torch.where(
        mask,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def make_orpo_loss_fn(
    response_start: int,
    orpo_lambda: float = 1.0,
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Single-pair ORPO loss wrapper over the batched implementation."""
    return make_batch_orpo_loss_fn([response_start], orpo_lambda=orpo_lambda)


def make_batch_orpo_loss_fn(
    response_starts: List[int],
    orpo_lambda: float = 1.0,
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Batched ORPO loss over multiple preference pairs.

    Expects ``2 * N`` datums arranged as
    ``[chosen_0, rejected_0, chosen_1, rejected_1, ...]``.

    The returned loss tensor sums the per-pair losses so a single fused
    ``forward_backward_custom`` preserves the previous "one pair per
    accumulation slot" behavior while still reducing RPC count.
    """
    n_pairs = len(response_starts)
    if n_pairs <= 0:
        raise ValueError("ORPO batching requires at least one pair.")

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(logprobs_list) == 2 * n_pairs, (
            f"Expected {2 * n_pairs} logprobs (2 per pair), got {len(logprobs_list)}"
        )

        total_loss = _zero_loss(logprobs_list)
        total_orpo_loss = 0.0
        total_sft_loss = 0.0
        total_or_loss = 0.0
        total_log_odds_ratio = 0.0
        total_chosen_avg_lp = 0.0
        total_rejected_avg_lp = 0.0
        total_accuracy = 0.0

        for i, response_start in enumerate(response_starts):
            lp_start = max(0, response_start - 1)
            chosen_lp = logprobs_list[2 * i][lp_start:]
            rejected_lp = logprobs_list[2 * i + 1][lp_start:]

            n_chosen = max(len(chosen_lp), 1)
            n_rejected = max(len(rejected_lp), 1)

            sft_loss = -chosen_lp.sum() / n_chosen
            avg_chosen_lp = chosen_lp.sum() / n_chosen
            avg_rejected_lp = rejected_lp.sum() / n_rejected

            avg_chosen_lp = torch.clamp(avg_chosen_lp, min=-50.0, max=-1e-6)
            avg_rejected_lp = torch.clamp(avg_rejected_lp, min=-50.0, max=-1e-6)

            log_odds_chosen = avg_chosen_lp - _log1mexp(avg_chosen_lp)
            log_odds_rejected = avg_rejected_lp - _log1mexp(avg_rejected_lp)
            log_odds_ratio = log_odds_chosen - log_odds_rejected
            or_loss = -F.logsigmoid(log_odds_ratio)
            pair_loss = sft_loss + orpo_lambda * or_loss

            total_loss = total_loss + pair_loss

            with torch.no_grad():
                total_orpo_loss += pair_loss.item()
                total_sft_loss += sft_loss.item()
                total_or_loss += or_loss.item()
                total_log_odds_ratio += log_odds_ratio.item()
                total_chosen_avg_lp += avg_chosen_lp.item()
                total_rejected_avg_lp += avg_rejected_lp.item()
                total_accuracy += 1.0 if log_odds_ratio.item() > 0 else 0.0

        with torch.no_grad():
            metrics = {
                "orpo_loss": total_orpo_loss / n_pairs,
                "sft_loss": total_sft_loss / n_pairs,
                "or_loss": total_or_loss / n_pairs,
                "log_odds_ratio": total_log_odds_ratio / n_pairs,
                "orpo_lambda": orpo_lambda,
                "chosen_avg_lp": total_chosen_avg_lp / n_pairs,
                "rejected_avg_lp": total_rejected_avg_lp / n_pairs,
                "accuracy": total_accuracy / n_pairs,
                "batch_pairs": n_pairs,
            }
        return total_loss, metrics

    return loss_fn


def make_batch_dpo_loss_fn(
    ref_chosen_list: List[List[float]],
    ref_rejected_list: List[List[float]],
    response_starts: List[int],
    beta: float,
    microbatch_sizes: List[int] | None = None,
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Batched DPO loss over multiple preference pairs.

    Expects ``2 * N`` datums arranged as
    ``[chosen_0, rejected_0, chosen_1, rejected_1, ...]`` and computes the
    mean DPO loss across all pairs.

    Args:
        ref_chosen_list: Per-pair reference logprobs for chosen sequences.
        ref_rejected_list: Per-pair reference logprobs for rejected sequences.
        response_starts: Per-pair token index where the response begins.
        beta: DPO temperature parameter.
        microbatch_sizes: Optional per-microbatch sizes for fused accumulation.
    """
    n_pairs = len(ref_chosen_list)
    assert len(ref_rejected_list) == n_pairs
    assert len(response_starts) == n_pairs
    microbatch_sizes = _validate_microbatch_sizes(n_pairs, microbatch_sizes)

    ref_chosen_ts = [torch.tensor(r, dtype=torch.float32) for r in ref_chosen_list]
    ref_rejected_ts = [torch.tensor(r, dtype=torch.float32) for r in ref_rejected_list]

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(logprobs_list) == 2 * n_pairs, (
            f"Expected {2 * n_pairs} logprobs (2 per pair), got {len(logprobs_list)}"
        )

        pair_losses: list[torch.Tensor] = []
        pair_margins: list[float] = []
        pair_accuracies: list[float] = []
        pair_chosen_rewards: list[float] = []
        pair_rejected_rewards: list[float] = []

        for i in range(n_pairs):
            rs = response_starts[i]
            # logprobs[j] predicts token[j+1], so the first response token
            # at full-sequence index `rs` has its logprob at index `rs - 1`.
            # Slicing [rs:] silently drops that logprob — which is often the
            # *only* token that differs between chosen and rejected and
            # collapses the DPO margin to zero. Matches make_batch_orpo_loss_fn.
            lp_start = max(0, rs - 1)
            pi_chosen = logprobs_list[2 * i][lp_start:].sum()
            pi_rejected = logprobs_list[2 * i + 1][lp_start:].sum()
            rc = ref_chosen_ts[i][lp_start:].sum()
            rr = ref_rejected_ts[i][lp_start:].sum()

            margin = (pi_chosen - rc) - (pi_rejected - rr)
            pair_loss = -F.logsigmoid(beta * margin)
            pair_losses.append(pair_loss)

            with torch.no_grad():
                pair_margins.append(margin.item())
                pair_accuracies.append(1.0 if margin.item() > 0 else 0.0)
                pair_chosen_rewards.append(beta * (pi_chosen.item() - rc.item()))
                pair_rejected_rewards.append(beta * (pi_rejected.item() - rr.item()))

        if microbatch_sizes is None:
            total_loss = torch.stack(pair_losses).sum()
            avg_loss = total_loss / n_pairs
            margin_metric = sum(pair_margins) / n_pairs
            accuracy_metric = sum(pair_accuracies) / n_pairs
            chosen_reward_metric = sum(pair_chosen_rewards) / n_pairs
            rejected_reward_metric = sum(pair_rejected_rewards) / n_pairs
            loss_to_backprop = avg_loss
        else:
            total_loss = _zero_loss(logprobs_list)
            total_microbatch_loss = 0.0
            total_microbatch_margin = 0.0
            total_microbatch_accuracy = 0.0
            total_microbatch_chosen_reward = 0.0
            total_microbatch_rejected_reward = 0.0
            cursor = 0

            for microbatch_size in microbatch_sizes:
                next_cursor = cursor + microbatch_size
                loss_chunk = pair_losses[cursor:next_cursor]
                margin_chunk = pair_margins[cursor:next_cursor]
                accuracy_chunk = pair_accuracies[cursor:next_cursor]
                chosen_reward_chunk = pair_chosen_rewards[cursor:next_cursor]
                rejected_reward_chunk = pair_rejected_rewards[cursor:next_cursor]

                microbatch_loss = torch.stack(loss_chunk).mean()
                total_loss = total_loss + microbatch_loss
                total_microbatch_loss += sum(loss.item() for loss in loss_chunk) / microbatch_size
                total_microbatch_margin += sum(margin_chunk) / microbatch_size
                total_microbatch_accuracy += sum(accuracy_chunk) / microbatch_size
                total_microbatch_chosen_reward += sum(chosen_reward_chunk) / microbatch_size
                total_microbatch_rejected_reward += sum(rejected_reward_chunk) / microbatch_size
                cursor = next_cursor

            microbatch_count = len(microbatch_sizes)
            avg_loss = total_loss / microbatch_count
            margin_metric = total_microbatch_margin / microbatch_count
            accuracy_metric = total_microbatch_accuracy / microbatch_count
            chosen_reward_metric = total_microbatch_chosen_reward / microbatch_count
            rejected_reward_metric = total_microbatch_rejected_reward / microbatch_count
            loss_to_backprop = total_loss

        with torch.no_grad():
            metrics = {
                "dpo_loss": avg_loss.item(),
                "margin": margin_metric,
                "accuracy": accuracy_metric,
                "chosen_reward": chosen_reward_metric,
                "rejected_reward": rejected_reward_metric,
                "batch_pairs": n_pairs,
            }
            if microbatch_sizes is not None:
                metrics["microbatch_count"] = len(microbatch_sizes)
        return loss_to_backprop, metrics

    return loss_fn


def make_sft_loss_fn(
    response_start: int,
    target_tokens: List[int],
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Cross-entropy loss over response tokens only (single-sample)."""
    targets = torch.tensor(target_tokens, dtype=torch.long)

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        lp = logprobs_list[0]
        # Same off-by-one correction as DPO/ORPO/batch-SFT: logprob[j] predicts
        # token[j+1], so the first response token (at full-sequence index
        # response_start) has its logprob at index response_start - 1.
        lp_start = max(0, response_start - 1)
        resp_lp = lp[lp_start:]
        resp_t = targets[lp_start:]
        n = max(len(resp_t), 1)
        ce = -resp_lp.sum() / n
        with torch.no_grad():
            ppl = torch.exp(ce).item()
        return ce, {"ce_loss": ce.item(), "ppl": ppl, "response_tokens": len(resp_t)}

    return loss_fn


def make_batch_sft_loss_fn(
    prompt_token_counts: List[int],
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Cross-entropy loss over response tokens for a *batch* of samples.

    Each sample may have a different prompt length. The loss is averaged across
    all response tokens in the batch (token-level mean), matching the behaviour
    of ``train_sft_tinker_sdk.py``.

    Args:
        prompt_token_counts: Per-sample prompt token counts. Tokens before this
            boundary are masked (no gradient contribution).
    """

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(data) == len(logprobs_list)
        assert len(prompt_token_counts) == len(logprobs_list)

        total_loss = torch.tensor(0.0)
        total_response_tokens = 0
        total_nll = 0.0

        for i, logprobs in enumerate(logprobs_list):
            response_start = max(0, prompt_token_counts[i] - 1)
            response_logprobs = logprobs[response_start:]
            n = len(response_logprobs)
            if n == 0:
                continue
            sample_nll = -response_logprobs.sum()
            total_loss = total_loss + sample_nll
            total_response_tokens += n
            with torch.no_grad():
                total_nll += sample_nll.item()

        if total_response_tokens > 0:
            avg_loss = total_loss / total_response_tokens
        else:
            avg_loss = total_loss

        with torch.no_grad():
            avg_nll = total_nll / total_response_tokens if total_response_tokens > 0 else 0.0
            ppl = torch.exp(torch.tensor(avg_nll)).item()

        return avg_loss, {
            "ce_loss": avg_nll,
            "ce_loss_sum": total_nll,
            "ppl": ppl,
            "response_tokens": total_response_tokens,
            "batch_size": len(logprobs_list),
        }

    return loss_fn


def make_batch_weighted_sft_loss_fn(
    microbatch_sizes: List[int] | None = None,
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Cross-entropy loss over per-token supervised weights stored on each datum.

    This is the renderer-safe path for multi-turn SFT. Each datum must include
    ``loss_fn_inputs["weights"]`` aligned with ``target_tokens``.
    """

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(data) == len(logprobs_list)

        microbatch_sizes_local = _validate_microbatch_sizes(len(data), microbatch_sizes)

        def _compute_sample_nll(
            datum: tinker.Datum,
            logprobs: torch.Tensor,
        ) -> tuple[torch.Tensor | None, float]:
            weights_td = datum.loss_fn_inputs.get("weights")
            if weights_td is None:
                raise ValueError("Weighted SFT expects each datum to include loss_fn_inputs['weights'].")

            weights = weights_td.to_torch().to(device=logprobs.device, dtype=logprobs.dtype)
            if len(weights) != len(logprobs):
                raise ValueError(
                    f"weights/logprobs length mismatch: {len(weights)} != {len(logprobs)}"
                )

            sample_weight = float(weights.sum().item())
            if sample_weight <= 0:
                return None, 0.0
            return -(logprobs * weights).sum(), sample_weight

        total_weight = 0.0
        total_nll = 0.0

        if microbatch_sizes_local is None:
            total_loss = _zero_loss(logprobs_list)

            for datum, logprobs in zip(data, logprobs_list, strict=True):
                sample_nll, sample_weight = _compute_sample_nll(datum, logprobs)
                if sample_nll is None:
                    continue

                total_loss = total_loss + sample_nll
                total_weight += sample_weight
                with torch.no_grad():
                    total_nll += float(sample_nll.item())

            loss_to_backprop = total_loss / total_weight if total_weight > 0 else total_loss
        else:
            loss_to_backprop = _zero_loss(logprobs_list)
            cursor = 0

            for microbatch_size in microbatch_sizes_local:
                next_cursor = cursor + microbatch_size
                microbatch_loss = _zero_loss(logprobs_list)
                microbatch_weight = 0.0

                for datum, logprobs in zip(
                    data[cursor:next_cursor],
                    logprobs_list[cursor:next_cursor],
                    strict=True,
                ):
                    sample_nll, sample_weight = _compute_sample_nll(datum, logprobs)
                    if sample_nll is None:
                        continue

                    microbatch_loss = microbatch_loss + sample_nll
                    microbatch_weight += sample_weight
                    total_weight += sample_weight
                    with torch.no_grad():
                        total_nll += float(sample_nll.item())

                if microbatch_weight > 0:
                    loss_to_backprop = loss_to_backprop + (microbatch_loss / microbatch_weight)
                cursor = next_cursor

        with torch.no_grad():
            avg_nll = total_nll / total_weight if total_weight > 0 else 0.0
            ppl = torch.exp(torch.tensor(avg_nll)).item()

        return loss_to_backprop, {
            "ce_loss": avg_nll,
            "ce_loss_sum": total_nll,
            "ppl": ppl,
            "response_tokens": total_weight,
            "weighted_tokens": total_weight,
            "batch_size": len(logprobs_list),
            "microbatch_count": len(microbatch_sizes_local) if microbatch_sizes_local is not None else 1,
        }

    return loss_fn
