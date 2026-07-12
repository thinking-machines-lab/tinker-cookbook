"""Tests for rubric env token DB capture metadata."""

from typing import Any, cast

from tinker_cookbook.recipes.rubric.data import Rubric, RubricBasedDatapoint
from tinker_cookbook.recipes.rubric.env import RubricGradedEnvGroupBuilder


def test_metadata_reports_num_rubrics():
    datapoint = RubricBasedDatapoint(
        convo=[{"role": "user", "content": "hi"}],
        rubric_items=[Rubric(rubric_str="clear"), Rubric(rubric_str="correct")],
    )
    builder = RubricGradedEnvGroupBuilder(
        renderer=cast(Any, None),
        datapoint=datapoint,
        grader_llm=cast(Any, None),
        group_size=2,
    )
    assert builder.metadata() == {"num_rubrics": 2}
