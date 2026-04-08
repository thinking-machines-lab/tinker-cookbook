"""Downstream compatibility tests for tinker_cookbook.recipes.

Validates that recipe modules used by downstream remain importable and have
the expected API surface.
"""

import inspect

import pytest

# ---------------------------------------------------------------------------
# recipes.math_rl
# ---------------------------------------------------------------------------


class TestMathRL:
    def test_math_env_importable(self):
        math_env = pytest.importorskip(
            "tinker_cookbook.recipes.math_rl.math_env",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        assert math_env is not None

    def test_arithmetic_env_importable(self):
        arithmetic_env = pytest.importorskip(
            "tinker_cookbook.recipes.math_rl.arithmetic_env",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        assert arithmetic_env is not None

    def test_get_math_dataset_builder(self):
        math_env = pytest.importorskip(
            "tinker_cookbook.recipes.math_rl.math_env",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        assert callable(math_env.get_math_dataset_builder)

    def test_math_env_classes(self):
        math_env = pytest.importorskip(
            "tinker_cookbook.recipes.math_rl.math_env",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        assert math_env.Gsm8kDatasetBuilder is not None
        assert math_env.MathDatasetBuilder is not None

    def test_math_grading_functions(self):
        math_grading = pytest.importorskip(
            "tinker_cookbook.recipes.math_rl.math_grading",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        assert callable(math_grading.grade_answer)
        assert callable(math_grading.normalize_answer)
        assert callable(math_grading.extract_boxed)

    def test_safe_grade(self):
        math_env = pytest.importorskip(
            "tinker_cookbook.recipes.math_rl.math_env",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        assert callable(math_env.safe_grade)


# ---------------------------------------------------------------------------
# recipes.code_rl
# ---------------------------------------------------------------------------


class TestCodeRL:
    def test_code_env_importable(self):
        from tinker_cookbook.recipes.code_rl.code_env import DeepcoderDatasetBuilder

        assert DeepcoderDatasetBuilder is not None


# ---------------------------------------------------------------------------
# recipes.chat_sl
# ---------------------------------------------------------------------------


class TestChatSL:
    def test_chat_datasets_importable(self):
        from tinker_cookbook.recipes.chat_sl import chat_datasets

        assert chat_datasets is not None

    def test_tulu3_builder_exists(self):
        from tinker_cookbook.recipes.chat_sl.chat_datasets import Tulu3Builder

        assert Tulu3Builder is not None


# ---------------------------------------------------------------------------
# recipes.preference
# ---------------------------------------------------------------------------


class TestPreference:
    def test_dpo_train_importable(self):
        from tinker_cookbook.recipes.preference.dpo.train import CLIConfig, cli_main

        assert CLIConfig is not None
        assert callable(cli_main)

    def test_preference_datasets_importable(self):
        from tinker_cookbook.recipes.preference.datasets import HHHComparisonBuilder

        assert HHHComparisonBuilder is not None


# ---------------------------------------------------------------------------
# recipes.rl_basic and sl_basic (used by config_utils)
# ---------------------------------------------------------------------------


class TestBasicRecipes:
    def test_rl_basic_build_config(self):
        rl_basic = pytest.importorskip(
            "tinker_cookbook.recipes.rl_basic",
            reason="math-rl dependencies not installed",
            exc_type=ImportError,
        )
        assert callable(rl_basic.build_config_blueprint)

    def test_sl_basic_build_config(self):
        from tinker_cookbook.recipes.sl_basic import build_config_blueprint

        assert callable(build_config_blueprint)


# ---------------------------------------------------------------------------
# eval.evaluators
# ---------------------------------------------------------------------------


class TestEvaluators:
    def test_sampling_client_evaluator_importable(self):
        from tinker_cookbook.eval.evaluators import SamplingClientEvaluator

        assert SamplingClientEvaluator is not None

    def test_training_client_evaluator_importable(self):
        from tinker_cookbook.eval.evaluators import TrainingClientEvaluator

        assert TrainingClientEvaluator is not None

    def test_evaluator_builder_importable(self):
        from tinker_cookbook.eval.evaluators import EvaluatorBuilder

        assert EvaluatorBuilder is not None


# ---------------------------------------------------------------------------
# distillation.datasets (used by tibo)
# ---------------------------------------------------------------------------


class TestDistillation:
    def test_prompt_only_env_importable(self):
        from tinker_cookbook.distillation.datasets import PromptOnlyEnv

        assert PromptOnlyEnv is not None

    def test_load_tulu3_prompts_importable(self):
        from tinker_cookbook.distillation.datasets import load_tulu3_prompts

        assert callable(load_tulu3_prompts)

    def test_distillation_package_exports(self):
        import tinker_cookbook.distillation as dist

        # Submodules
        assert hasattr(dist, "train_on_policy")
        assert hasattr(dist, "train_off_policy")
        assert hasattr(dist, "sdft")

        # Each submodule has Config and main
        assert hasattr(dist.train_on_policy, "Config")
        assert hasattr(dist.train_on_policy, "main")
        assert hasattr(dist.train_off_policy, "Config")
        assert hasattr(dist.train_off_policy, "main")
        assert hasattr(dist.sdft, "Config")
        assert hasattr(dist.sdft, "main")

        # Shared dataset types
        assert dist.DistillationDatasetConfig is not None
        assert dist.TeacherConfig is not None
        assert dist.PromptOnlyDatasetBuilder is not None

        # SDFT protocol and utilities
        assert dist.SDFTBatchProvider is not None
        assert callable(dist.build_sdft_teacher_prompt)

    def test_distillation_all_complete(self):
        import tinker_cookbook.distillation as dist

        expected = {
            "train_on_policy",
            "train_off_policy",
            "sdft",
            "DistillationDatasetConfig",
            "TeacherConfig",
            "PromptOnlyDatasetBuilder",
            "SDFTBatchProvider",
            "build_sdft_teacher_prompt",
        }
        assert set(dist.__all__) == expected


# ---------------------------------------------------------------------------
# preference.types (used by rl_cli)
# ---------------------------------------------------------------------------


class TestPreferenceTypes:
    def test_preference_model_builder_importable(self):
        from tinker_cookbook.preference.types import PreferenceModelBuilderFromChatRenderer

        assert PreferenceModelBuilderFromChatRenderer is not None


# ---------------------------------------------------------------------------
# supervised.train (entry point)
# ---------------------------------------------------------------------------


class TestSupervisedTrain:
    def test_config_exists(self):
        from tinker_cookbook.supervised.train import Config

        assert Config is not None

    def test_main_exists(self):
        from tinker_cookbook.supervised.train import main

        assert callable(main)

    def test_main_is_async(self):
        from tinker_cookbook.supervised.train import main

        assert inspect.iscoroutinefunction(main)


# ---------------------------------------------------------------------------
# utils.lr_scheduling
# ---------------------------------------------------------------------------


class TestLRScheduling:
    def test_lr_schedule_importable(self):
        from tinker_cookbook.utils.lr_scheduling import LRSchedule

        assert LRSchedule is not None
