"""Light test for the SDFT recipe's token DB capture wiring.

Asserts the capture call shape (which attrs the recipe stamps onto captured
sample rows) against an in-memory writer, without running real generation.
"""

import pytest

pytest.importorskip("pyarrow")

from tinker_cookbook.recipes.sdft import train as sdft_train
from tinker_cookbook.tokendb.capture import capture_samples
from tinker_cookbook.tokendb.capture_test import ListWriter
from tinker_cookbook.tokendb.sample_capture_test import make_completer, run_completer


def test_attrs_shape_self_distillation():
    cfg = sdft_train.CLIConfig()
    attrs = sdft_train._token_db_attrs(cfg)
    # Self-distillation: teacher and student are the same model.
    assert attrs == {
        "student_model": cfg.model_name,
        "teacher_model": cfg.model_name,
        "dataset": cfg.dataset,
    }


def test_captured_rows_carry_recipe_attrs():
    cfg = sdft_train.CLIConfig()
    writer = ListWriter()
    with capture_samples(writer, attrs=sdft_train._token_db_attrs(cfg)):
        run_completer(make_completer())
    (row,) = writer.rows
    assert row.source == "sample"
    assert row.attrs["teacher_model"] == cfg.model_name
    assert row.attrs["dataset"] == cfg.dataset
