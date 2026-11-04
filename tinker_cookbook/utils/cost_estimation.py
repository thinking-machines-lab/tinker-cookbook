"""Running token and USD accounting for the training loops.

This gives per-step numbers: what one training step sampled, prefilled and
trained, and what that cost. For a run's total cost use
``RestClient.get_billing_usage``, which reports the usage Tinker attributed to
each session in hourly buckets, per kind, with its estimated cost and
effective rates:
https://tinker-docs.thinkingmachines.ai/tinker/api-reference/restclient/#get_billing_usage.
An hourly bucket cannot be split back into steps, which is why the trainers
keep this count from what they send and receive. (It also needs billing-view
access and can lag by a few hours.)

Tinker bills every token a trainer touches, and in RL most tokens are billed
more than once: a sampled token is paid for at the sample rate when it is
generated and again at the train rate when the trajectory is trained on, and
the prompt is paid for at the prefill rate on every turn (at the cheaper
cached-prefill rate where it hits the prefix cache). Estimating from sampled
tokens alone undercounts a run by roughly half, so the trainers log the
counts, and a USD estimate when the model's rates are known, alongside the
other per-step metrics.

Metrics are logged once per step and cumulatively::

    tokens/sampled               tokens returned by sampling requests this step
    tokens/prefill_cached        prompt tokens reported as prefix-cache hits
    tokens/prefill_uncached      the rest of the prompt tokens, at the full prefill rate
    tokens/trained               tokens in the datums sent to forward_backward
    tokens/cumulative/<kind>     running totals since the tracker was created
    cost/step_usd                see :meth:`TokenCostRates.estimate_usd`
    cost/cumulative_usd

Supervised training only has trained tokens, so its sampling and prefill
counts are zero.

Cumulative totals are saved with each checkpoint record by both trainers: the
RL ``loop_state`` carries them as ``cumulative_tokens``, the supervised one as
``elapsed_tokens``. ``loop_state`` is a cookbook-side dict, written flat into
the run's ``checkpoints.jsonl`` record next to the ``tinker://`` paths; it is
not part of the saved weights or optimizer state. Automatic resume from the
latest record in ``log_path`` restores the totals. A fresh run that starts from
``load_checkpoint_path`` imports weights only, so its totals start at zero.

Rates come from the machine-readable pricing file published next to the
Models & Pricing page (``models.json``, see :func:`fetch_model_pricing`). It
is downloaded once when the trainer process starts and kept in memory for the
life of the process; nothing is written to disk, and a restart fetches again.
USD is estimated only when the model's entry carries all four prices
(``prefill``, ``cached_prefill``, ``sample`` and ``train``). If the fetch
fails, the model is not listed, or its entry is incomplete, a warning is
logged once and the run logs token counts only, for the whole process; there
is never a partial USD figure. Training never depends on the fetch.

The estimate is deliberately simple. A step's ``tokens/`` and
``cost/step_usd`` cover the batch that step consumed: its rollouts, its
training, and its reference-KL and post-update-KL scoring requests (taken
from what each response reports). Trajectory groups that completed but were
dropped before training for having a constant reward are counted apart, with
:meth:`CostTracker.record_discarded_sampling`: they go into the cumulative
totals at once and are reported as ``tokens/discarded/<kind>`` and
``cost/discarded_usd``, never as part of a step. It is not a complete
ledger: it does not see evaluation rollouts, requests that failed before
returning a response, or rollouts still in flight when training stops.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from tinker_cookbook import __version__

logger = logging.getLogger(__name__)

TOKENS_PER_MILLION = 1_000_000

PRICING_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
"""The Training table of the Models & Pricing page, one object per model."""


@dataclass(frozen=True)
class TokenUsage:
    """Billed token counts by kind, in the same four kinds as :class:`TokenCostRates`."""

    sampled: int = 0
    prefill_cached: int = 0
    prefill_uncached: int = 0
    trained: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            sampled=self.sampled + other.sampled,
            prefill_cached=self.prefill_cached + other.prefill_cached,
            prefill_uncached=self.prefill_uncached + other.prefill_uncached,
            trained=self.trained + other.trained,
        )

    def metrics(self, prefix: str) -> dict[str, int]:
        """The counts as ``<prefix>/<kind>`` metrics."""
        return {f"{prefix}/{kind}": count for kind, count in asdict(self).items()}


@dataclass(frozen=True)
class TokenCostRates:
    """USD per million tokens for each kind of token the trainer is billed for.

    The fields match the ``prefill``, ``cached_prefill``, ``sample`` and
    ``train`` columns of the Models & Pricing page.
    """

    prefill_usd_per_million: float
    cached_prefill_usd_per_million: float
    sample_usd_per_million: float
    train_usd_per_million: float

    def estimate_usd(self, usage: TokenUsage) -> float:
        """Estimate the USD cost of *usage*.

        ``usd = (prefill_uncached * prefill + prefill_cached * cached_prefill
        + sampled * sample + trained * train) / 1e6``, each rate in USD per
        million tokens.
        """
        return (
            usage.prefill_uncached * self.prefill_usd_per_million
            + usage.prefill_cached * self.cached_prefill_usd_per_million
            + usage.sampled * self.sample_usd_per_million
            + usage.trained * self.train_usd_per_million
        ) / TOKENS_PER_MILLION


def _parse_price(value: object) -> float | None:
    """``"$1.87"`` -> ``1.87``; a missing or unparseable price -> ``None``."""
    try:
        return float(str(value).strip().lstrip("$").replace(",", ""))
    except ValueError:
        return None


def rates_from_pricing_entry(entry: Mapping[str, object]) -> TokenCostRates | None:
    """Rates from one ``models.json`` object (prices are strings like ``"$1.87"``).

    ``None`` when any of the four prices is missing or unparseable.
    """
    prefill = _parse_price(entry.get("prefill"))
    cached_prefill = _parse_price(entry.get("cached_prefill"))
    sample = _parse_price(entry.get("sample"))
    train = _parse_price(entry.get("train"))
    if prefill is None or cached_prefill is None or sample is None or train is None:
        return None
    return TokenCostRates(
        prefill_usd_per_million=prefill,
        cached_prefill_usd_per_million=cached_prefill,
        sample_usd_per_million=sample,
        train_usd_per_million=train,
    )


def _download_pricing(url: str, timeout: float) -> list[dict[str, object]]:
    # The docs site rejects urllib's default User-Agent with 403.
    request = urllib.request.Request(url, headers={"User-Agent": f"tinker-cookbook/{__version__}"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        data = json.load(resp)
    if not isinstance(data, list) or not all(isinstance(entry, dict) for entry in data):
        raise ValueError(f"expected a JSON list of objects at {url}")
    return data


# The pricing file for this process, fetched once; None after a failed fetch.
_pricing_entries: list[dict[str, object]] | None = None
_pricing_fetched = False


def load_model_pricing(
    *, url: str = PRICING_URL, timeout: float = 5.0
) -> list[dict[str, object]] | None:
    """The pricing file, downloaded on the first call and kept in memory afterwards.

    Returns ``None`` when the download failed; the failure is logged once and
    not retried, so a run that started without pricing stays without it until
    the process restarts. Nothing is read from or written to disk.
    """
    global _pricing_entries, _pricing_fetched
    if not _pricing_fetched:
        _pricing_fetched = True
        try:
            _pricing_entries = _download_pricing(url, timeout)
        except Exception as e:  # network or parse; training must not depend on it
            logger.warning(
                "Could not fetch model pricing from %s (%s: %s); logging token counts only",
                url,
                type(e).__name__,
                e,
            )
    return _pricing_entries


def fetch_model_pricing(
    model_name: str, *, url: str = PRICING_URL, timeout: float = 5.0
) -> TokenCostRates | None:
    """Complete rates for *model_name* from the published pricing file, or ``None``.

    The model is matched on its exact ``tinker_id``, including any
    ``:peft:<ctx>`` suffix, because long-context variants have their own
    prices. The entry must carry all four prices; when the model is unlisted
    or its entry is incomplete, a warning is logged and ``None`` is returned
    so the run logs token counts only. Never raises.
    """
    entries = load_model_pricing(url=url, timeout=timeout)
    if entries is None:
        return None
    for entry in entries:
        if entry.get("tinker_id") == model_name:
            rates = rates_from_pricing_entry(entry)
            if rates is None:
                logger.warning(
                    "Pricing entry for %s in %s is incomplete (%s); logging token counts only",
                    model_name,
                    url,
                    entry,
                )
                return None
            note = entry.get("note")
            logger.info("Pricing for %s: %s%s", model_name, rates, f" ({note})" if note else "")
            return rates
    logger.warning("Model %s is not listed in %s; logging token counts only", model_name, url)
    return None


class CostTracker:
    """Accumulate per-step token usage and turn it into ``tokens/`` and ``cost/`` metrics.

    Args:
        rates (TokenCostRates | None): Rates for the USD estimate (see
            :func:`fetch_model_pricing`). ``None`` logs token counts only.
        cumulative (TokenUsage): Usage already accounted for before this
            tracker was created, i.e. the previous process's
            :attr:`cumulative` read back from its checkpoint, so totals
            continue across a resume.
        discarded (TokenUsage): Likewise the previous process's
            :attr:`discarded`.

    Example::

        tracker = CostTracker(fetch_model_pricing("Qwen/Qwen3-8B"))
        metrics.update(tracker.record_step(TokenUsage(trained=4096)))
    """

    def __init__(
        self,
        rates: TokenCostRates | None = None,
        *,
        cumulative: TokenUsage = TokenUsage(),
        discarded: TokenUsage = TokenUsage(),
    ):
        self.rates = rates
        self.cumulative = cumulative
        """Running totals, discarded sampling included; persisted for resume."""
        self.discarded = discarded
        """The part of :attr:`cumulative` that was sampled for groups dropped before training."""
        self.step = TokenUsage()
        """The usage of the latest step recorded."""

    def record_discarded_sampling(self, usage: TokenUsage) -> None:
        """Count the sampling of a group that was dropped before training.

        It goes into :attr:`cumulative` and :attr:`discarded` at once; no step
        reports it (see :meth:`cumulative_metrics`).
        """
        self.cumulative += usage
        self.discarded += usage

    def cumulative_metrics(self) -> dict[str, float]:
        """The running totals only: ``tokens/cumulative/<kind>``, ``tokens/discarded/<kind>``
        and their USD. These are levels, so repeating them is harmless."""
        metrics: dict[str, float] = {
            **self.cumulative.metrics("tokens/cumulative"),
            **self.discarded.metrics("tokens/discarded"),
        }
        if self.rates is not None:
            metrics["cost/cumulative_usd"] = self.rates.estimate_usd(self.cumulative)
            metrics["cost/discarded_usd"] = self.rates.estimate_usd(self.discarded)
        return metrics

    def record_step(self, usage: TokenUsage) -> dict[str, float]:
        """Add one step's usage and return the metrics to log for it: ``tokens/<kind>``,
        ``tokens/cumulative/<kind>``, and ``cost/step_usd`` and ``cost/cumulative_usd``
        when the tracker has rates."""
        self.step = usage
        self.cumulative += usage
        return self._step_metrics()

    def extend_step(self, usage: TokenUsage) -> dict[str, float]:
        """Add usage that belongs to the latest step but was only known after it was
        recorded (the post-update KL scoring runs after the step's checkpoint), and
        return the step's complete metrics again."""
        self.step += usage
        self.cumulative += usage
        return self._step_metrics()

    def _step_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            **self.step.metrics("tokens"),
            **self.cumulative.metrics("tokens/cumulative"),
        }
        if self.rates is not None:
            metrics["cost/step_usd"] = self.rates.estimate_usd(self.step)
            metrics["cost/cumulative_usd"] = self.rates.estimate_usd(self.cumulative)
        return metrics
