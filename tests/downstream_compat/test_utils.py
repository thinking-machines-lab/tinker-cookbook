"""Downstream compatibility tests for tinker_cookbook.utils.

Validates that logging, tracing, and misc utilities remain stable.
"""

import inspect

from tinker_cookbook.utils.misc_utils import (
    all_same,
    concat_lists,
    dict_mean,
    not_none,
    safezip,
    split_list,
    timed,
)
from tinker_cookbook.utils.ml_log import (
    JsonLogger,
    Logger,
    MultiplexLogger,
    PrettyPrintLogger,
    configure_logging_module,
    dump_config,
    setup_logging,
)

# ---------------------------------------------------------------------------
# ml_log
# ---------------------------------------------------------------------------


class TestLoggerHierarchy:
    def test_logger_is_abstract(self):
        assert inspect.isabstract(Logger)

    def test_json_logger_is_subclass(self):
        assert issubclass(JsonLogger, Logger)

    def test_pretty_print_logger_is_subclass(self):
        assert issubclass(PrettyPrintLogger, Logger)

    def test_multiplex_logger_is_subclass(self):
        assert issubclass(MultiplexLogger, Logger)

    def test_setup_logging_callable(self):
        assert callable(setup_logging)

    def test_configure_logging_module_callable(self):
        assert callable(configure_logging_module)

    def test_dump_config_callable(self):
        assert callable(dump_config)


# ---------------------------------------------------------------------------
# misc_utils
# ---------------------------------------------------------------------------


class TestMiscUtils:
    def test_dict_mean(self):
        result = dict_mean([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}])
        assert result["a"] == 2.0
        assert result["b"] == 3.0

    def test_all_same_true(self):
        assert all_same([1, 1, 1]) is True

    def test_all_same_false(self):
        assert all_same([1, 2, 1]) is False

    def test_split_list(self):
        result = split_list([1, 2, 3, 4], 2)
        assert len(result) == 2

    def test_concat_lists(self):
        result = concat_lists([[1, 2], [3, 4]])
        assert result == [1, 2, 3, 4]

    def test_not_none(self):
        assert not_none(42) == 42

    def test_safezip(self):
        result = list(safezip([1, 2], [3, 4]))
        assert result == [(1, 3), (2, 4)]

    def test_timed_is_context_manager(self):
        assert callable(timed)


# ---------------------------------------------------------------------------
# trace (import-only check — used by tibo training code)
# ---------------------------------------------------------------------------


class TestTraceImports:
    def test_trace_importable(self):
        from tinker_cookbook.utils.trace import scope, trace_init, update_scope_context

        assert callable(scope)
        assert callable(trace_init)
        assert callable(update_scope_context)

    def test_logtree_importable(self):
        from tinker_cookbook.utils import logtree

        assert logtree is not None
