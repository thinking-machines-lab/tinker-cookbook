"""Tests for package-level __init__.py re-exports.

Validates that core types are importable from the package level
(e.g., ``from tinker_cookbook.supervised import ChatDatasetBuilder``)
and that __all__ matches the actual exports.
"""

import importlib

import pytest

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# ---------------------------------------------------------------------------
# Helper: verify __all__ is consistent with actual exports
# ---------------------------------------------------------------------------


def _check_all_exports(module_path: str) -> None:
    """Verify every name in __all__ is actually importable from the module."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "__all__"), f"{module_path} has no __all__"
    for name in mod.__all__:
        assert hasattr(mod, name), (
            f"{module_path}.__all__ lists '{name}' but it is not an attribute"
        )


# ---------------------------------------------------------------------------
# supervised
# ---------------------------------------------------------------------------


class TestSupervisedExports:
    def test_all_consistent(self):
        _check_all_exports("tinker_cookbook.supervised")

    def test_types_importable(self):
        from tinker_cookbook.supervised import (
            ChatDatasetBuilder,
            ChatDatasetBuilderCommonConfig,
            SupervisedDataset,
            SupervisedDatasetBuilder,
        )

        assert issubclass(ChatDatasetBuilder, SupervisedDatasetBuilder)
        assert ChatDatasetBuilderCommonConfig is not None
        assert SupervisedDataset is not None

    def test_data_importable(self):
        from tinker_cookbook.supervised import (
            FromConversationFileBuilder,
            StreamingSupervisedDatasetFromHFDataset,
            SupervisedDatasetFromHFDataset,
            conversation_to_datum,
        )

        assert callable(conversation_to_datum)
        assert FromConversationFileBuilder is not None
        assert SupervisedDatasetFromHFDataset is not None
        assert StreamingSupervisedDatasetFromHFDataset is not None

    def test_helpers_importable(self):
        from tinker_cookbook.supervised import compute_mean_nll, datum_from_model_input_weights

        assert callable(compute_mean_nll)
        assert callable(datum_from_model_input_weights)


# ---------------------------------------------------------------------------
# rl
# ---------------------------------------------------------------------------


class TestRLExports:
    def test_all_consistent(self):
        _check_all_exports("tinker_cookbook.rl")

    def test_core_types_importable(self):
        from tinker_cookbook.rl import (
            Action,
            Env,
            EnvGroupBuilder,
            Logs,
            Metrics,
            Observation,
            RLDataset,
            RLDatasetBuilder,
            RolloutError,
            StepResult,
            Trajectory,
            TrajectoryGroup,
            Transition,
        )

        assert Env is not None
        assert EnvGroupBuilder is not None
        assert RLDataset is not None
        assert RLDatasetBuilder is not None
        assert Trajectory is not None
        assert TrajectoryGroup is not None
        assert StepResult is not None
        assert Transition is not None
        assert RolloutError is not None
        assert Action is not None
        assert Observation is not None
        assert Metrics is not None
        assert Logs is not None

    def test_rollout_strategies_importable(self):
        from tinker_cookbook.rl import FailFast, RetryOnFailure, RolloutStrategy

        assert issubclass(FailFast, RolloutStrategy)
        assert issubclass(RetryOnFailure, RolloutStrategy)


# ---------------------------------------------------------------------------
# preference
# ---------------------------------------------------------------------------


class TestPreferenceExports:
    def test_all_consistent(self):
        _check_all_exports("tinker_cookbook.preference")

    def test_types_importable(self):
        from tinker_cookbook.preference import (
            Comparison,
            ComparisonDatasetBuilder,
            LabeledComparison,
        )

        assert Comparison is not None
        assert LabeledComparison is not None
        assert ComparisonDatasetBuilder is not None


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


class TestEvalExports:
    def test_all_consistent(self):
        _check_all_exports("tinker_cookbook.eval")

    def test_types_importable(self):
        from tinker_cookbook.eval import (
            Evaluator,
            EvaluatorBuilder,
            SamplingClientEvaluator,
            SamplingClientEvaluatorBuilder,
            TrainingClientEvaluator,
        )

        assert SamplingClientEvaluator is not None
        assert TrainingClientEvaluator is not None
        assert EvaluatorBuilder is not None
        assert SamplingClientEvaluatorBuilder is not None
        assert Evaluator is not None


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


class TestWeightsExports:
    def test_all_consistent(self):
        _check_all_exports("tinker_cookbook.weights")

    def test_core_functions_importable(self):
        from tinker_cookbook.weights import build_hf_model, build_lora_adapter, download

        assert callable(build_hf_model)
        assert callable(build_lora_adapter)
        assert callable(download)


# ---------------------------------------------------------------------------
# renderers: concrete classes not in __all__
# ---------------------------------------------------------------------------


class TestRendererExports:
    def test_all_consistent(self):
        _check_all_exports("tinker_cookbook.renderers")

    def test_concrete_renderers_not_in_all(self):
        assert "DeepSeekV3ThinkingRenderer" not in renderers.__all__
        assert "GptOssRenderer" not in renderers.__all__
        assert "Qwen3Renderer" not in renderers.__all__

    def test_vl_renderer_without_image_processor_succeeds_for_text(self):
        """VL renderers should construct without image_processor for text-only use."""
        from tinker_cookbook.renderers.base import RenderContext

        tokenizer = get_tokenizer("Qwen/Qwen2.5-VL-3B-Instruct")
        for name in ("qwen3_vl", "qwen3_vl_instruct"):
            renderer = renderers.get_renderer(name, tokenizer)
            ctx = RenderContext(idx=0, is_last=True)
            result = renderer.render_message({"role": "user", "content": "hello"}, ctx)
            assert len(result.output) > 0
