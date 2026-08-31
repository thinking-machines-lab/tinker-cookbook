import torch
import tinker


def _tensor_from_loss_input(
    datum: tinker.Datum,
    key: str,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    """Convert a Datum loss input to a torch tensor matching `like`."""
    if key not in datum.loss_fn_inputs:
        raise KeyError(f"FPGB loss requires datum.loss_fn_inputs[{key!r}]")

    value = datum.loss_fn_inputs[key]
    if hasattr(value, "to_torch"):
        tensor = value.to_torch()
    else:
        tensor = torch.as_tensor(value.data)

    return tensor.to(device=like.device, dtype=like.dtype)


def compute_fpgb_loss(
    data: list[tinker.Datum],
    current_logprobs: list[torch.Tensor],
    *,
    beta: float,
    reference_data: list[tinker.Datum],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the FPGB residual-regression loss.

    For sampled action tokens, fit the current policy change

        delta_logp = log pi_theta(a|s) - log pi_k(a|s)

    to the FPGB target

        beta * A_k.

    `data` contains only the loss inputs accepted by Tinker's custom-loss
    API (`target_tokens` and `weights`). FPGB-specific rollout statistics are
    supplied separately in `reference_data`, which is captured client-side by
    the custom-loss closure.

    The rollout-time log-probabilities in `reference_data` are treated as the
    frozen stage policy pi_k. `current_logprobs` are differentiable target-token
    log-probabilities supplied by Tinker's custom-loss API.

    The objective is

        mean_masked[(delta_logp - beta * advantage)^2].

    Notes:
        This is a practical log-probability-space realization of the FPGB
        residual fit. Tinker's custom loss API currently exposes target-token
        log-probabilities rather than full vocabulary logits.
    """
    if beta <= 0:
        raise ValueError(f"FPGB beta must be positive, got {beta}")

    if len(data) != len(current_logprobs):
        raise ValueError(
            "FPGB custom loss received mismatched batch sizes: "
            f"{len(data)=}, {len(current_logprobs)=}"
        )
    if len(reference_data) != len(current_logprobs):
        raise ValueError(
            "FPGB custom loss received mismatched reference batch sizes: "
            f"{len(reference_data)=}, {len(current_logprobs)=}"
        )

    if not data:
        raise ValueError("FPGB custom loss received an empty batch")

    total_sq_error: torch.Tensor | None = None
    total_weight: torch.Tensor | None = None

    active_delta: list[torch.Tensor] = []
    active_target: list[torch.Tensor] = []

    for i, (datum, reference_datum, logp) in enumerate(
        zip(data, reference_data, current_logprobs, strict=True)
    ):
        # `datum` is the sanitized object sent through Tinker's custom-loss API
        # and therefore contains only target_tokens / weights.
        # FPGB-specific rollout statistics are kept client-side in
        # `reference_data` and captured by the Python closure.
        old_logp = _tensor_from_loss_input(reference_datum, "logprobs", like=logp)
        advantages = _tensor_from_loss_input(reference_datum, "advantages", like=logp)
        mask = _tensor_from_loss_input(reference_datum, "mask", like=logp)

        if old_logp.shape != logp.shape:
            raise ValueError(
                f"Datum {i}: rollout logprobs shape {tuple(old_logp.shape)} "
                f"does not match current logprobs shape {tuple(logp.shape)}"
            )
        if advantages.shape != logp.shape:
            raise ValueError(
                f"Datum {i}: advantages shape {tuple(advantages.shape)} "
                f"does not match current logprobs shape {tuple(logp.shape)}"
            )
        if mask.shape != logp.shape:
            raise ValueError(
                f"Datum {i}: mask shape {tuple(mask.shape)} "
                f"does not match current logprobs shape {tuple(logp.shape)}"
            )

        if not torch.isfinite(logp).all():
            raise ValueError(f"Datum {i}: current logprobs contain non-finite values")
        if not torch.isfinite(old_logp).all():
            raise ValueError(f"Datum {i}: rollout logprobs contain non-finite values")
        if not torch.isfinite(advantages).all():
            raise ValueError(f"Datum {i}: advantages contain non-finite values")
        if not torch.isfinite(mask).all():
            raise ValueError(f"Datum {i}: mask contains non-finite values")

        # Treat mask as a nonnegative weight. In the standard RL data pipeline
        # it is binary (1 for sampled action tokens, 0 for prompt/masked tokens).
        if torch.any(mask < 0):
            raise ValueError(f"Datum {i}: FPGB mask contains negative values")

        delta_logp = logp - old_logp
        target = beta * advantages
        residual = delta_logp - target

        weighted_sq_error = residual.square() * mask
        datum_sq_error = weighted_sq_error.sum()
        datum_weight = mask.sum()

        if total_sq_error is None:
            total_sq_error = datum_sq_error
            total_weight = datum_weight
        else:
            total_sq_error = total_sq_error + datum_sq_error
            assert total_weight is not None
            total_weight = total_weight + datum_weight

        active = mask > 0
        if torch.any(active):
            active_delta.append(delta_logp[active].detach())
            active_target.append(target[active].detach())

    assert total_sq_error is not None
    assert total_weight is not None

    if float(total_weight.detach()) <= 0.0:
        raise ValueError("FPGB batch contains no active action tokens (mask sum is zero)")

    loss = total_sq_error / total_weight

    delta = torch.cat(active_delta)
    target = torch.cat(active_target)
    residual = delta - target

    # unbiased=False avoids NaN when a tiny smoke-test batch has only one
    # active token.
    delta_std = delta.std(unbiased=False)
    target_std = target.std(unbiased=False)

    delta_norm = torch.linalg.vector_norm(delta)
    target_norm = torch.linalg.vector_norm(target)
    denom = delta_norm * target_norm
    if float(denom) > 0.0:
        cosine = torch.dot(delta, target) / denom
    else:
        cosine = torch.zeros((), device=delta.device, dtype=delta.dtype)

    metrics = {
        "fpgb/loss": loss.detach().item(),
        "fpgb/delta_mean": delta.mean().item(),
        "fpgb/delta_std": delta_std.item(),
        "fpgb/target_mean": target.mean().item(),
        "fpgb/target_std": target_std.item(),
        "fpgb/residual_rmse": residual.square().mean().sqrt().item(),
        "fpgb/delta_target_cosine": cosine.item(),
        "fpgb/num_active_tokens": float(total_weight.detach().item()),
    }

    return loss, metrics
