"""Tests for the benchmark evaluation framework."""

import pytest

from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig, BenchmarkResult


class TestBenchmarkResult:
    def test_construction(self):
        r = BenchmarkResult(name="test", score=0.8, num_examples=100, num_correct=80)
        assert r.score == 0.8
        assert r.num_correct == 80

    def test_default_metrics(self):
        r = BenchmarkResult(name="test", score=0.5, num_examples=10, num_correct=5)
        assert r.metrics == {}


class TestBenchmarkConfig:
    def test_defaults(self):
        c = BenchmarkConfig()
        assert c.max_examples is None
        assert c.concurrency == 64
        assert c.max_tokens == 32768
        assert c.temperature == 0.6
        assert c.save_dir is None
        assert c.save_every == 50


class TestGSM8KImport:
    def test_gsm8k_registered(self):
        # Importing the module triggers registration
        import tinker_cookbook.eval.benchmarks.gsm8k  # noqa: F401
        from tinker_cookbook.eval.benchmarks import REGISTRY

        assert "gsm8k" in REGISTRY
        assert REGISTRY["gsm8k"].name == "gsm8k"

    def test_answer_extraction(self):
        from tinker_cookbook.eval.benchmarks.gsm8k import extract_gsm8k_answer

        assert extract_gsm8k_answer("The answer is 42") == "42"
        assert extract_gsm8k_answer("\\boxed{123}") == "123"
        assert extract_gsm8k_answer("#### 7") == "7"
        assert extract_gsm8k_answer("So we get 3.14 as the result") == "3.14"

    def test_check_gsm8k(self):
        from tinker_cookbook.eval.benchmarks.gsm8k import check_gsm8k

        assert check_gsm8k("The answer is 42.", "42")
        assert check_gsm8k("\\boxed{42}", "42")
        assert not check_gsm8k("The answer is 43.", "42")


class TestDefaultAggregate:
    def test_accuracy(self):
        builder = type("B", (BenchmarkBuilder,), {
            "name": "test",
            "make_envs": lambda self, r, c: [],
        })()
        result = builder.aggregate([1.0, 0.0, 1.0, 1.0], [{}, {}, {}, {}])
        assert result.score == 0.75
        assert result.num_correct == 3
        assert result.num_examples == 4

    def test_empty(self):
        builder = type("B", (BenchmarkBuilder,), {
            "name": "test",
            "make_envs": lambda self, r, c: [],
        })()
        result = builder.aggregate([], [])
        assert result.score == 0.0
        assert result.num_examples == 0
