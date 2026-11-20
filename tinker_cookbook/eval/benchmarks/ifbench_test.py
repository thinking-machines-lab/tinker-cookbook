"""Tests for the IFBench benchmark integration."""

import importlib
import sys
import types


def test_imports_checker_registry_from_ifbench_package(monkeypatch):
    checkers = {"test:constraint": object}
    ifbench_package = types.ModuleType("ifbench")
    ifbench_package.__dict__["__path__"] = []
    registry = types.ModuleType("ifbench.instructions_registry")
    registry.__dict__["INSTRUCTION_DICT"] = checkers

    monkeypatch.setitem(sys.modules, "ifbench", ifbench_package)
    monkeypatch.setitem(sys.modules, "ifbench.instructions_registry", registry)
    monkeypatch.delitem(sys.modules, "instructions_registry", raising=False)

    module = importlib.import_module("tinker_cookbook.eval.benchmarks.ifbench")
    module = importlib.reload(module)

    assert module._IFBENCH_CHECKERS is checkers
