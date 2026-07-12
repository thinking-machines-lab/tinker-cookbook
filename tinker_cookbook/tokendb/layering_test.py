"""Enforce the one-way dependency between tokendb and tokendb_studio.

The persistence layer (``tinker_cookbook.tokendb``) must be importable and
fully functional without the studio app or its ``aiohttp`` dependency; the
studio (``tinker_cookbook.tokendb_studio``) imports tokendb through its
public API, never the other way around. The single sanctioned exception is
``tokendb/serve.py``, a deprecated forwarder kept so
``python -m tinker_cookbook.tokendb.serve`` still works.
"""

import ast
import builtins
import subprocess
import sys
from pathlib import Path

TOKENDB_DIR = Path(__file__).parent

#: The compat forwarder is the one module allowed to import the studio.
ALLOWED_FORWARDERS = {"serve.py"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_tokendb_never_imports_the_studio():
    offenders: list[str] = []
    for path in sorted(TOKENDB_DIR.glob("*.py")):
        if path.name in ALLOWED_FORWARDERS:
            continue
        for module in _imported_modules(path):
            if module == "tinker_cookbook.tokendb_studio" or module.startswith(
                "tinker_cookbook.tokendb_studio."
            ):
                offenders.append(f"{path.name}: {module}")
    assert not offenders, (
        "tinker_cookbook.tokendb must not depend on the studio app; "
        f"found imports of tokendb_studio in: {offenders}"
    )


def test_tokendb_never_imports_aiohttp():
    """The persistence layer's only third-party deps are pyarrow and duckdb."""
    offenders: list[str] = []
    for path in sorted(TOKENDB_DIR.glob("*.py")):
        if path.name in ALLOWED_FORWARDERS or path.name.endswith("_test.py"):
            continue
        for module in _imported_modules(path):
            if module == "aiohttp" or module.startswith("aiohttp."):
                offenders.append(f"{path.name}: {module}")
    assert not offenders, f"tokendb modules import aiohttp: {offenders}"


def test_tokendb_imports_with_aiohttp_blocked():
    """``import tinker_cookbook.tokendb`` works when aiohttp is uninstallable.

    Runs in a subprocess so the block applies to a fresh interpreter with no
    cached modules.
    """
    code = (
        "import sys\n"
        "sys.modules['aiohttp'] = None  # import raises ImportError\n"
        "import tinker_cookbook.tokendb\n"
        "from tinker_cookbook.tokendb import TokenDB, TokenDbWriter, RegistryBackend\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_module_level_import_of_tokendb_does_not_pull_studio(monkeypatch):
    """Importing tokendb modules must not (even indirectly) import the studio."""
    for name in list(sys.modules):
        if name.startswith("tinker_cookbook.tokendb"):
            monkeypatch.delitem(sys.modules, name)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("tinker_cookbook.tokendb_studio"), (
            f"import of {name} while importing tinker_cookbook.tokendb"
        )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    import tinker_cookbook.tokendb  # noqa: F401
