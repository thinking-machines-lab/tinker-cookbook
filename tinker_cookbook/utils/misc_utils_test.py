from __future__ import annotations

import math

from tinker_cookbook.utils.misc_utils import dict_mean


def test_dict_mean_basic():
    assert dict_mean([{"a": 1.0}, {"a": 3.0}]) == {"a": 2.0}


def test_dict_mean_missing_keys_averaged_over_present_only():
    # keys present in only some dicts average over just those entries
    assert dict_mean([{"a": 1, "b": 4}, {"a": 3}]) == {"a": 2.0, "b": 4.0}


def test_dict_mean_skips_none_values():
    # a None value (e.g. a reward's binary verdict on a trial that died before
    # grading) must be skipped, not averaged in — it previously crashed with
    # "unsupported operand type(s) for +: 'float' and 'NoneType'".
    assert dict_mean([{"resolved": 1.0}, {"resolved": None}, {"resolved": 0.0}]) == {
        "resolved": 0.5
    }


def test_dict_mean_all_none_key_dropped():
    # a key whose only values are None is dropped entirely (no verdict to report)
    assert dict_mean([{"resolved": None}, {"resolved": None}]) == {}


def test_dict_mean_mixed_none_and_missing():
    out = dict_mean([{"a": 2.0, "b": None}, {"a": None}, {"b": 4.0}])
    assert out == {"a": 2.0, "b": 4.0}


def test_dict_mean_empty():
    assert dict_mean([]) == {}


def test_dict_mean_returns_python_floats():
    out = dict_mean([{"a": 1}, {"a": 2}])
    assert isinstance(out["a"], float)
    assert math.isclose(out["a"], 1.5)
