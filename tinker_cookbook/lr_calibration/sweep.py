"""Run the LoRA learning-rate sweep for one model on plain text windows.

Every learning rate runs as its own LoRA training client, concurrently, on the
same windows in the same order. Training uses ``cross_entropy`` on every token
after the first, a linear decay from the peak LR to zero, and AdamW. Each run ends
with a forward pass on held-out windows, reported in bits per byte.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import tinker
import torch

from tinker_cookbook.lr_calibration.coverage import base_model_name
from tinker_cookbook.supervised.common import compute_bpb, datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

logger = logging.getLogger(__name__)

# First-time imports of transformers and tml_renderers from many threads at once can deadlock.
_TOKENIZER_LOAD_LOCK = threading.Lock()

# A step is bad if its training bits per byte exceed DIVERGED_RATIO * first + DIVERGED_MARGIN;
# a run is diverged after DIVERGED_STEPS bad steps in a row.
DIVERGED_RATIO = 1.5
DIVERGED_MARGIN = 0.2
DIVERGED_STEPS = 3

_PendingStep = tuple[
    list[tinker.Datum],
    tinker.APIFuture[tinker.ForwardBackwardOutput],
    tinker.APIFuture[tinker.OptimStepResponse],
]


@dataclass(frozen=True)
class SweepConfig:
    """Training setup shared by every run of a calibration sweep, read from the table's
    ``protocol`` block.

    Attributes:
        batch_size (int): Windows per optimizer step.
        steps (int): Optimizer steps per run.
        lora_rank (int): LoRA rank.
        n_eval (int): Held-out windows (taken from the start of the corpus).
        adam_beta1 (float): AdamW beta1.
        adam_beta2 (float): AdamW beta2.
        adam_eps (float): AdamW epsilon.
        seed (int): LoRA initialization seed.
    """

    batch_size: int
    steps: int
    lora_rank: int
    n_eval: int
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    seed: int = 0


@dataclass
class RunResult:
    """Outcome of one learning rate."""

    lr: float
    diverged: bool
    eval_bpb: float
    train_bpb: list[float] = field(default_factory=list)
    tokens: int = 0


def _make_examples(texts: list[str], tokenizer: Tokenizer) -> list[tinker.Datum]:
    """One datum per text, with loss on every token after the first."""
    out = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        model_input = tinker.ModelInput.from_ints(tokens)
        out.append(datum_from_model_input_weights(model_input, torch.ones(len(tokens))))
    return out


def _bits_per_byte(
    outputs: list[tinker.types.LossFnOutput], data: list[tinker.Datum], tokenizer: Tokenizer
) -> float:
    return compute_bpb(
        [output["logprobs"] for output in outputs],
        [datum.loss_fn_inputs["weights"] for datum in data],
        [datum.loss_fn_inputs["target_tokens"] for datum in data],
        tokenizer,
    )


async def _evaluate(
    tc: tinker.TrainingClient, data: list[tinker.Datum], tokenizer: Tokenizer, chunk: int = 16
) -> float:
    parts = [data[i : i + chunk] for i in range(0, len(data), chunk)]
    futures = [tc.forward(part, "cross_entropy") for part in parts]
    outputs = [o for fut in futures for o in (await fut).loss_fn_outputs]
    return _bits_per_byte(outputs, data, tokenizer)


async def _run_one(
    service_client: tinker.ServiceClient,
    model_name: str,
    lr: float,
    config: SweepConfig,
    train: list[tinker.Datum],
    evals: list[tinker.Datum],
    tokenizer: Tokenizer,
) -> RunResult:
    tc = await service_client.create_lora_training_client(
        base_model=model_name, rank=config.lora_rank, seed=config.seed
    )
    result = RunResult(lr=lr, diverged=False, eval_bpb=math.nan)
    first_bpb: float | None = None
    bad_steps = 0
    pending: _PendingStep | None = None

    async def finish(p: _PendingStep) -> None:
        nonlocal first_bpb, bad_steps
        batch, fb, opt = p
        bpb = _bits_per_byte((await fb).loss_fn_outputs, batch, tokenizer)
        await opt
        result.train_bpb.append(bpb)
        result.tokens += sum(d.model_input.length for d in batch)
        first_bpb = bpb if first_bpb is None else first_bpb
        bad = not math.isfinite(bpb) or bpb > DIVERGED_RATIO * first_bpb + DIVERGED_MARGIN
        bad_steps = bad_steps + 1 if bad else 0

    for step in range(config.steps):
        batch = train[step * config.batch_size : (step + 1) * config.batch_size]
        fb = tc.forward_backward(batch, "cross_entropy")
        opt = tc.optim_step(
            tinker.AdamParams(
                learning_rate=lr * compute_schedule_lr_multiplier("linear", step, config.steps),
                beta1=config.adam_beta1,
                beta2=config.adam_beta2,
                eps=config.adam_eps,
            )
        )
        if pending is not None:
            await finish(pending)
        pending = (batch, fb, opt)
        if bad_steps >= DIVERGED_STEPS:
            result.diverged = True
            break
    if pending is not None:
        await finish(pending)
    result.diverged = result.diverged or bad_steps >= DIVERGED_STEPS
    if not result.diverged:
        result.eval_bpb = await _evaluate(tc, evals, tokenizer)
        result.tokens += sum(d.model_input.length for d in evals)
        result.diverged = not math.isfinite(result.eval_bpb)
    logger.info(
        "%s lr=%.3g eval_bpb=%.4f diverged=%s", model_name, lr, result.eval_bpb, result.diverged
    )
    return result


def _load_tokenizer(model_name: str) -> Tokenizer:
    with _TOKENIZER_LOAD_LOCK:
        return get_tokenizer(base_model_name(model_name))


async def run_sweep(
    service_client: tinker.ServiceClient,
    model_name: str,
    lrs: list[float],
    windows: list[str],
    config: SweepConfig,
    on_result: Callable[[RunResult], None] | None = None,
) -> list[RunResult]:
    """Train one LoRA run per learning rate, concurrently, and return their results.

    Args:
        service_client (tinker.ServiceClient): Client to create training clients from.
        model_name (str): Tinker model id to calibrate.
        lrs (list[float]): Peak learning rates.
        windows (list[str]): Corpus windows; the first ``config.n_eval`` are held out.
        config (SweepConfig): Shared training setup.
        on_result (Callable[[RunResult], None] | None): Called as each run finishes, so
            callers can save results before the slowest run is done.

    Returns:
        list[RunResult]: One result per learning rate, in the order given.
    """
    needed = config.n_eval + config.batch_size * config.steps
    if len(windows) < needed:
        raise ValueError(f"need {needed} windows, got {len(windows)}")
    # Loading and running a tokenizer takes seconds; threads keep other models' sweeps moving.
    tokenizer = await asyncio.to_thread(_load_tokenizer, model_name)
    evals, train = await asyncio.gather(
        asyncio.to_thread(_make_examples, windows[: config.n_eval], tokenizer),
        asyncio.to_thread(_make_examples, windows[config.n_eval : needed], tokenizer),
    )

    async def run(lr: float) -> RunResult:
        result = await _run_one(service_client, model_name, lr, config, train, evals, tokenizer)
        if on_result is not None:
            on_result(result)
        return result

    return list(await asyncio.gather(*[run(lr) for lr in lrs]))
