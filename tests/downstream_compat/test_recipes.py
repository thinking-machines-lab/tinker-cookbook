"""Downstream compatibility tests for tinker_cookbook.recipes.

Validates that recipe modules used by downstream remain importable and have
the expected API surface.
"""

import inspect

# ---------------------------------------------------------------------------
# recipes.math_rl
# ---------------------------------------------------------------------------


class TestMathRL:
    def test_math_env_importable(self):
        from tinker_cookbook.recipes.math_rl import math_env

        assert math_env is not None

    def test_arithmetic_env_importable(self):
        from tinker_cookbook.recipes.math_rl import arithmetic_env

        assert arithmetic_env is not None

    def test_get_math_dataset_builder(self):
        from tinker_cookbook.recipes.math_rl.math_env import get_math_dataset_builder

        assert callable(get_math_dataset_builder)

    def test_math_env_classes(self):
        from tinker_cookbook.recipes.math_rl.math_env import (
            Gsm8kDatasetBuilder,
            MathDatasetBuilder,
        )

        assert Gsm8kDatasetBuilder is not None
        assert MathDatasetBuilder is not None

    def test_math_grading_functions(self):
        from tinker_cookbook.recipes.math_rl.math_grading import (
            extract_boxed,
            grade_answer,
            normalize_answer,
        )

        assert callable(grade_answer)
        assert callable(normalize_answer)
        assert callable(extract_boxed)

    def test_safe_grade(self):
        from tinker_cookbook.recipes.math_rl.math_env import safe_grade

        assert callable(safe_grade)


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
        from tinker_cookbook.recipes.rl_basic import build_config_blueprint

        assert callable(build_config_blueprint)

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
