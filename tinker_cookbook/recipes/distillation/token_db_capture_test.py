"""Light tests for the distillation recipes' token DB capture wiring.

Asserts the capture call shape (which attrs each recipe stamps onto captured
sample rows) against an in-memory writer, without running real generation.
"""

import pytest

pytest.importorskip("pyarrow")

from tinker_cookbook.recipes.distillation import on_policy_distillation, on_policy_multi_teacher
from tinker_cookbook.tokendb.capture import capture_samples
from tinker_cookbook.tokendb.capture_test import ListWriter
from tinker_cookbook.tokendb.sample_capture_test import make_completer, run_completer


class TestOnPolicyDistillation:
    def test_attrs_shape(self):
        cfg = on_policy_distillation.CLIConfig()
        assert on_policy_distillation._token_db_attrs(cfg) == {
            "student_model": cfg.model_name,
            "teacher_model": cfg.teacher_model,
            "dataset": cfg.dataset,
        }

    def test_captured_rows_carry_recipe_attrs(self):
        cfg = on_policy_distillation.CLIConfig()
        writer = ListWriter()
        with capture_samples(writer, attrs=on_policy_distillation._token_db_attrs(cfg)):
            run_completer(make_completer())
        (row,) = writer.rows
        assert row.source == "sample"
        assert row.attrs["student_model"] == cfg.model_name
        assert row.attrs["teacher_model"] == cfg.teacher_model
        assert row.attrs["dataset"] == cfg.dataset


class TestOnPolicyMultiTeacher:
    def test_attrs_shape(self):
        cfg = on_policy_multi_teacher.CLIConfig()
        assert on_policy_multi_teacher._token_db_attrs(cfg) == {
            "student_model": cfg.model_name,
            "deepmath_teacher_model": cfg.deepmath_teacher_model,
            "tulu3_teacher_model": cfg.tulu3_teacher_model,
            "dataset": "deepmath+tulu3",
        }


class TestHarborMultiTurn:
    def test_attrs_shape(self):
        mod = pytest.importorskip(
            "tinker_cookbook.recipes.distillation.on_policy_distillation_harbor_multi_turn",
            reason="requires modal",
            exc_type=ImportError,
        )
        cfg = mod.CLIConfig()
        assert mod._token_db_attrs(cfg) == {
            "student_model": cfg.model_name,
            "teacher_model": cfg.teacher_model,
            "dataset": cfg.task_name,
        }
