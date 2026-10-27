import pytest

from tinker_cookbook.eval.benchmarks._common import get_sandbox_factory
from tinker_cookbook.eval.benchmarks._types import BenchmarkConfig


def test_get_sandbox_factory_returns_configured_factory():
    async def factory():
        raise NotImplementedError

    config = BenchmarkConfig(sandbox_factory=factory)
    assert get_sandbox_factory(config, packages=["git"]) is factory


def test_get_sandbox_factory_builds_sail_default():
    pytest.importorskip("sail")
    config = BenchmarkConfig(sandbox_backend="sail")
    assert callable(get_sandbox_factory(config, packages=["git"]))
