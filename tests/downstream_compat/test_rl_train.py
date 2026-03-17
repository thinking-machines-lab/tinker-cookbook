"""Downstream compatibility tests for tinker_cookbook.rl.train and rl.data_processing.

Validates that RL training entry points and data processing functions remain stable.
"""

import inspect

from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    trajectory_to_data,
)
from tinker_cookbook.rl.train import Config, main

# ---------------------------------------------------------------------------
# rl.train
# ---------------------------------------------------------------------------


class TestRLTrainConfig:
    def test_config_exists(self):
        assert Config is not None

    def test_main_exists(self):
        assert callable(main)

    def test_main_is_async(self):
        assert inspect.iscoroutinefunction(main)


# ---------------------------------------------------------------------------
# rl.data_processing
# ---------------------------------------------------------------------------


class TestRLDataProcessing:
    def test_compute_advantages_callable(self):
        assert callable(compute_advantages)

    def test_trajectory_to_data_callable(self):
        assert callable(trajectory_to_data)

    def test_assemble_training_data_callable(self):
        assert callable(assemble_training_data)

    def test_compute_advantages_signature(self):
        sig = inspect.signature(compute_advantages)
        params = list(sig.parameters.keys())
        assert len(params) >= 1  # takes trajectory_groups


# ---------------------------------------------------------------------------
# rl.metrics (used by tibo training code)
# ---------------------------------------------------------------------------


class TestRLMetrics:
    def test_metrics_importable(self):
        from tinker_cookbook.rl.metrics import (
            compute_kl_sample_train,
            discounted_future_sum_vectorized,
        )

        assert callable(compute_kl_sample_train)
        assert callable(discounted_future_sum_vectorized)

    def test_metric_util_importable(self):
        from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics

        assert RLTestSetEvaluator is not None
        assert callable(compute_trajectory_metrics)


# ---------------------------------------------------------------------------
# rl.rollouts (used by web_search_tasks)
# ---------------------------------------------------------------------------


class TestRLRollouts:
    def test_do_single_rollout_importable(self):
        from tinker_cookbook.rl.rollouts import do_single_rollout

        assert callable(do_single_rollout)
