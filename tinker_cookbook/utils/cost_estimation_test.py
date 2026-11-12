from __future__ import annotations

import io
import json
import logging
import os
import pathlib
import urllib.request

import pytest

from tinker_cookbook.utils import cost_estimation
from tinker_cookbook.utils.cost_estimation import (
    CostTracker,
    TokenCostRates,
    TokenUsage,
    fetch_model_pricing,
    rates_from_pricing_entry,
)

# Per-million rates in the shape of the Models & Pricing page.
RATES = TokenCostRates(
    prefill_usd_per_million=2.0,
    cached_prefill_usd_per_million=0.4,
    sample_usd_per_million=4.0,
    train_usd_per_million=6.0,
)

# One entry of models.json, as published (prices are "$" strings).
INKLING_ENTRY = {
    "name": "Inkling",
    "tinker_id": "thinkingmachines/Inkling",
    "prefill": "$1.87",
    "cached_prefill": "$0.374",
    "sample": "$4.68",
    "train": "$5.61",
}
INKLING_LONG_ENTRY = {
    **INKLING_ENTRY,
    "tinker_id": "thinkingmachines/Inkling:peft:262144",
    "train": "$11.23",
    "note": "temporary discount",
    "original_train": "$12.00",
}
PRICING = [INKLING_ENTRY, INKLING_LONG_ENTRY]


def test_estimate_usd_bills_each_kind_at_its_own_rate():
    # 1M uncached prefill @ $2 + 1M cached @ $0.4 + 0.5M sampled @ $4 + 1.5M trained @ $6
    usd = RATES.estimate_usd(
        TokenUsage(
            prefill_uncached=1_000_000,
            prefill_cached=1_000_000,
            sampled=500_000,
            trained=1_500_000,
        )
    )
    assert usd == pytest.approx(2.0 + 0.4 + 2.0 + 9.0)


def test_estimate_usd_counts_sampled_tokens_twice_when_they_are_also_trained():
    # An RL step where every sampled token is trained on: the sample rate and the
    # train rate both apply, so the estimate is not just sampled * sample_rate.
    sampled = 1_000_000
    usd = RATES.estimate_usd(TokenUsage(sampled=sampled, trained=sampled))
    assert usd == pytest.approx(4.0 + 6.0)


def test_token_usage_adds_by_kind_and_names_its_metrics():
    total = TokenUsage(sampled=1, prefill_cached=2) + TokenUsage(prefill_cached=3, trained=4)
    assert total == TokenUsage(sampled=1, prefill_cached=5, prefill_uncached=0, trained=4)
    assert total.metrics("tokens") == {
        "tokens/sampled": 1,
        "tokens/prefill_cached": 5,
        "tokens/prefill_uncached": 0,
        "tokens/trained": 4,
    }


def test_rates_from_pricing_entry_parses_dollar_strings():
    assert rates_from_pricing_entry(INKLING_LONG_ENTRY) == TokenCostRates(
        prefill_usd_per_million=1.87,
        cached_prefill_usd_per_million=0.374,
        sample_usd_per_million=4.68,
        train_usd_per_million=11.23,
    )


def test_rates_from_pricing_entry_is_none_unless_all_four_prices_parse():
    assert rates_from_pricing_entry({**INKLING_ENTRY, "train": "n/a"}) is None
    assert (
        rates_from_pricing_entry({k: v for k, v in INKLING_ENTRY.items() if k != "sample"}) is None
    )


def test_download_sends_a_cookbook_user_agent(monkeypatch):
    # The docs site answers urllib's default User-Agent with 403.
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        requests.append(request)
        return io.BytesIO(json.dumps(PRICING).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert cost_estimation._download_pricing(cost_estimation.PRICING_URL, 5.0) == PRICING
    (request,) = requests
    assert request.full_url == cost_estimation.PRICING_URL
    assert request.get_header("User-agent", "").startswith("tinker-cookbook/")


@pytest.mark.skipif(
    os.environ.get("TINKER_COOKBOOK_LIVE_TESTS") != "1",
    reason="needs network; set TINKER_COOKBOOK_LIVE_TESTS=1",
)
def test_published_pricing_file_is_reachable_and_complete():
    entries = cost_estimation._download_pricing(cost_estimation.PRICING_URL, 10.0)
    assert entries, "models.json is empty"
    for entry in entries:
        assert "tinker_id" in entry, entry
        assert rates_from_pricing_entry(entry) is not None, entry


@pytest.fixture
def unfetched(monkeypatch):
    """Start each test as a fresh process: nothing fetched yet, no disk access allowed."""
    monkeypatch.setattr(cost_estimation, "_pricing_fetched", False)
    monkeypatch.setattr(cost_estimation, "_pricing_entries", None)

    def no_disk(*args, **kwargs):
        raise AssertionError("pricing must not touch the filesystem")

    monkeypatch.setattr(pathlib.Path, "read_text", no_disk)
    monkeypatch.setattr(pathlib.Path, "write_text", no_disk)
    monkeypatch.setattr(cost_estimation, "open", no_disk, raising=False)


def _downloads(monkeypatch, result):
    calls: list[tuple[str, float]] = []

    def fake_download(url: str, timeout: float):
        calls.append((url, timeout))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(cost_estimation, "_download_pricing", fake_download)
    return calls


def test_pricing_is_fetched_once_per_process_and_kept_in_memory(unfetched, monkeypatch, caplog):
    calls = _downloads(monkeypatch, PRICING)

    long_ctx = fetch_model_pricing("thinkingmachines/Inkling:peft:262144")
    assert long_ctx is not None and long_ctx.train_usd_per_million == 11.23
    short = fetch_model_pricing("thinkingmachines/Inkling")
    assert short is not None and short.train_usd_per_million == 5.61
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.utils.cost_estimation"):
        assert fetch_model_pricing("Qwen/Qwen3-8B") is None
    assert any("not listed" in r.message for r in caplog.records)
    assert calls == [(cost_estimation.PRICING_URL, 5.0)]


def test_failed_fetch_is_not_retried_and_leaves_token_counts_only(unfetched, monkeypatch, caplog):
    calls = _downloads(monkeypatch, OSError("no network"))
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.utils.cost_estimation"):
        rates = fetch_model_pricing("thinkingmachines/Inkling")
        assert rates is None
        assert fetch_model_pricing("thinkingmachines/Inkling") is None
    assert calls == [(cost_estimation.PRICING_URL, 5.0)]
    assert sum("Could not fetch" in r.message for r in caplog.records) == 1

    tracker = CostTracker(rates)
    for _ in range(2):
        metrics = tracker.record_step(TokenUsage(sampled=10, trained=10))
        assert "cost/step_usd" not in metrics
    assert tracker.cumulative == TokenUsage(sampled=20, trained=20)


def test_complete_entry_prices_the_run(unfetched, monkeypatch):
    _downloads(monkeypatch, PRICING)
    rates = fetch_model_pricing("thinkingmachines/Inkling")
    assert rates == rates_from_pricing_entry(INKLING_ENTRY)
    metrics = CostTracker(rates).record_step(TokenUsage(sampled=1_000_000, trained=1_000_000))
    assert metrics["cost/step_usd"] == pytest.approx(4.68 + 5.61)


@pytest.mark.parametrize(
    "entry",
    [
        {k: v for k, v in INKLING_ENTRY.items() if k != "cached_prefill"},  # missing a price
        {**INKLING_ENTRY, "cached_prefill": "n/a"},  # malformed price
    ],
)
def test_incomplete_entry_turns_usd_off_for_the_process(entry, unfetched, monkeypatch, caplog):
    _downloads(monkeypatch, [entry])
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.utils.cost_estimation"):
        rates = fetch_model_pricing("thinkingmachines/Inkling")
    assert rates is None
    assert sum("incomplete" in r.message for r in caplog.records) == 1

    # The missing rate's bucket being zero makes no difference, and USD never appears
    # later when that bucket does get tokens.
    tracker = CostTracker(rates)
    first = tracker.record_step(TokenUsage(sampled=10, trained=10))
    later = tracker.record_step(TokenUsage(sampled=10, prefill_cached=5, trained=10))
    assert not any(k.startswith("cost/") for k in first)
    assert not any(k.startswith("cost/") for k in later)
    assert later["tokens/cumulative/prefill_cached"] == 5


def test_tracker_logs_step_and_cumulative_metrics():
    tracker = CostTracker(RATES)
    first = tracker.record_step(
        TokenUsage(sampled=100, prefill_cached=200, prefill_uncached=300, trained=400)
    )
    first_usd = (300 * 2.0 + 200 * 0.4 + 100 * 4.0 + 400 * 6.0) / 1e6
    assert first == {
        "tokens/sampled": 100,
        "tokens/prefill_cached": 200,
        "tokens/prefill_uncached": 300,
        "tokens/trained": 400,
        "tokens/cumulative/sampled": 100,
        "tokens/cumulative/prefill_cached": 200,
        "tokens/cumulative/prefill_uncached": 300,
        "tokens/cumulative/trained": 400,
        "cost/step_usd": pytest.approx(first_usd),
        "cost/cumulative_usd": pytest.approx(first_usd),
    }
    second = tracker.record_step(TokenUsage(sampled=50, prefill_uncached=100, trained=150))
    assert second["tokens/sampled"] == 50
    assert second["tokens/cumulative/sampled"] == 150
    assert second["tokens/cumulative/prefill_cached"] == 200
    assert second["tokens/cumulative/prefill_uncached"] == 400
    assert second["tokens/cumulative/trained"] == 550
    assert second["cost/cumulative_usd"] == pytest.approx(first_usd + second["cost/step_usd"])
    assert tracker.cumulative == TokenUsage(
        sampled=150, prefill_cached=200, prefill_uncached=400, trained=550
    )


def test_tracker_without_rates_logs_token_counts_only():
    metrics = CostTracker(None).record_step(TokenUsage(sampled=1, prefill_uncached=2, trained=3))
    assert set(metrics) == {
        "tokens/sampled",
        "tokens/prefill_cached",
        "tokens/prefill_uncached",
        "tokens/trained",
        "tokens/cumulative/sampled",
        "tokens/cumulative/prefill_cached",
        "tokens/cumulative/prefill_uncached",
        "tokens/cumulative/trained",
    }


def test_tracker_continues_from_resumed_usage():
    # Supervised training bills only trained tokens; its total is seeded from the
    # resumed run and the sampling counts stay at zero.
    tracker = CostTracker(RATES, cumulative=TokenUsage(trained=1_000_000))
    metrics = tracker.record_step(TokenUsage(trained=500_000))
    assert metrics["tokens/trained"] == 500_000
    assert metrics["tokens/cumulative/trained"] == 1_500_000
    assert metrics["tokens/sampled"] == 0
    assert metrics["cost/step_usd"] == pytest.approx(3.0)
    assert metrics["cost/cumulative_usd"] == pytest.approx(9.0)
