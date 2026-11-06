from __future__ import annotations

import json

import pytest

from tinker_cookbook.recipes.code_rl.code_env import _normalize_tests
from tinker_cookbook.recipes.code_rl.code_grading import (
    postprocess_lcb_sample,
    taco_to_lcb_format,
)


@pytest.mark.parametrize(
    "fn_name, inputs, outputs",
    [
        # Reduced function-call rows from DeepCoder's TACO train split.
        (
            "is_anagram",
            [["foefet", "toffee"], ["Buckethead", "DeathCubeK"], ["dumble", "bumble"]],
            [[True], [True], [False]],
        ),
        ("solve_for_x", [["x - 5 = 20"]], [[25]]),
        ("make_acronym", [["My aunt sally"]], [["MAS"]]),
        # A list-valued argument must remain one argument, not be flattened.
        ("nested_list", [[[[7, 2, 1], [6, 4, 2]], True]], [[[7, 6]]]),
        # Non-list inputs represent a single positional argument.
        ("single_argument", ['a "quoted" string\nwith a newline'], [["result"]]),
    ],
    ids=["anagram", "integer", "string", "nested-list", "non-list-input"],
)
def test_taco_functional_tests_match_grader_json_contract(
    fn_name: str, inputs: list[object], outputs: list[list[object]]
) -> None:
    raw_tests = {"fn_name": fn_name, "inputs": inputs, "outputs": outputs}
    normalized = _normalize_tests(json.dumps(raw_tests), {})
    assert len(normalized) == len(inputs)
    for case in normalized:
        assert case["testtype"] == "functional"
        assert case["metadata"] == {"func_name": fn_name}

    # Decode exactly as the call-based LiveCodeBench runner does, without executing code.
    packed = json.loads(postprocess_lcb_sample(normalized)["input_output"])
    assert packed["fn_name"] == fn_name
    assert len(packed["inputs"]) == len(inputs)
    assert len(packed["outputs"]) == len(outputs)
    for encoded_input, encoded_output, inp, out in zip(
        packed["inputs"], packed["outputs"], inputs, outputs, strict=True
    ):
        expected_args = inp if isinstance(inp, list) else [inp]
        assert [json.loads(line) for line in encoded_input.split("\n")] == expected_args
        decoded_output = json.loads(encoded_output)
        assert decoded_output == out[0]
        assert type(decoded_output) is type(out[0])


def test_taco_stdin_stdout_tests_are_unchanged() -> None:
    inp = "5 3 10\n1 2 3 4 5\nRGBRR\n"
    out = "4\n"
    raw_tests = {"inputs": [inp], "outputs": [out]}
    assert taco_to_lcb_format(raw_tests) == [
        {"input": inp, "output": out, "testtype": "stdin_stdout", "metadata": {}}
    ]
    assert _normalize_tests(json.dumps(raw_tests), {}) == [
        {
            "input": inp,
            "output": out,
            "testtype": "stdin_stdout",
            "metadata": {"func_name": None},
        }
    ]
