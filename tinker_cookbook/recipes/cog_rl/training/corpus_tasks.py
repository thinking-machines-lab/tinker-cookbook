"""Build verified Cog tasks from an existing programming-problem corpus (MBPP).

This is the scalable alternative to hand-authoring a Cog reference per task. The grader is
language-agnostic on the expected side: it runs the model's Cog ``solve`` on hidden inputs
and compares the printed output to a stored string. We produce that string by running the
corpus's **Python** reference on the corpus's own test inputs — so any problem that ships a
reference + tests becomes a Cog task with zero hand-written Cog.

Pipeline per problem:
  1. Parse the ``assert f(args) == expected`` tests into ``(args, expected)`` pairs.
  2. Execute the Python reference and confirm it reproduces each pair (drops broken refs).
  3. Keep only pairs whose inputs and outputs are Cog-representable (int / text / flag /
     list), via ``cog_format.supported`` — Cog has no float / dict / set / tuple.
  4. Require >= 2 distinct expected outputs (a constant can't pass; mirrors the hidden-input
     design of the hand-authored families).
  5. Emit a ``CogTask`` whose prompt is the problem's natural-language spec plus a
     ``forge solve(<params>)`` instruction, and whose tests are ``(cog_args, cog_repr)``.

Shape coverage (top-k, grouping, two-pointer, ...) then comes from the corpus's diversity,
not from us enumerating families. Train/eval are split by problem so eval problems are held
out entirely.
"""

from __future__ import annotations

import ast
import random
import warnings

from tinker_cookbook.recipes.cog_rl.training.cog_format import cog_args, cog_repr, supported
from tinker_cookbook.recipes.cog_rl.training.tasks import CogTask

_MBPP_CONFIG = "full"
_MBPP_SPLITS = ("train", "test", "validation", "prompt")


def _func_def(code: str) -> tuple[str, list[str]] | None:
    """Return (function name, param names) for the first top-level def in ``code``."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            a = node.args
            if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
                return None
            return node.name, [p.arg for p in a.args]
    return None


def _parse_assert(stmt: str, func: str) -> tuple[tuple, object] | None:
    """Extract ``(args, expected)`` from ``assert func(args) == expected`` (or a bare
    ``assert func(args)`` meaning expected True). Returns None if it isn't that shape or the
    args/expected aren't Python literals."""
    try:
        tree = ast.parse(stmt.strip())
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        return None
    test = tree.body[0].test

    def call_args(call: ast.Call):
        if not isinstance(call, ast.Call) or call.keywords:
            return None
        if not (isinstance(call.func, ast.Name) and call.func.id == func):
            return None
        try:
            return tuple(ast.literal_eval(a) for a in call.args)
        except (ValueError, SyntaxError):
            return None

    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        args = call_args(test.left)
        if args is None:
            return None
        try:
            expected = ast.literal_eval(test.comparators[0])
        except (ValueError, SyntaxError):
            return None
        return args, expected
    if isinstance(test, ast.Call):
        args = call_args(test)
        if args is None:
            return None
        return args, True
    return None


def _build_one(row: dict) -> CogTask | None:
    code = row.get("code") or ""
    sig = _func_def(code)
    if sig is None:
        return None
    func, params = sig
    if not params:
        return None

    # Execute the reference so we can validate (and so broken refs are dropped). Many MBPP
    # refs contain unescaped regex strings that emit SyntaxWarnings on compile; mute them.
    ns: dict = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for imp in row.get("test_imports") or []:
                exec(imp, ns)
            exec(code, ns)
    except Exception:
        return None
    fn = ns.get(func)
    if not callable(fn):
        return None

    tests: list[tuple[str, str]] = []
    for stmt in row.get("test_list") or []:
        parsed = _parse_assert(stmt, func)
        if parsed is None:
            return None  # an unparseable test means we can't trust coverage; skip the problem
        args, expected = parsed
        if len(args) != len(params):
            return None
        if not all(supported(a) for a in args) or not supported(expected):
            return None
        try:
            got = fn(*args)
        except Exception:
            return None
        if got != expected or not supported(got):
            return None
        tests.append((cog_args(args), cog_repr(expected)))

    if len(tests) < 2 or len({e for _, e in tests}) < 2:
        return None

    nl = (row.get("prompt") or row.get("text") or "").strip().rstrip(".")
    prompt = (
        f"{nl}. Define `forge solve({', '.join(params)})` that does this. It is tested on "
        f"hidden inputs; do not hardcode the answer."
    )
    return CogTask(
        name=f"mbpp_{row.get('task_id')}", family="corpus", prompt=prompt, tests=tuple(tests)
    )


def _load_rows() -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    for split in _MBPP_SPLITS:
        try:
            ds = load_dataset("mbpp", _MBPP_CONFIG, split=split)
        except Exception:
            continue
        rows.extend(dict(r) for r in ds)
    return rows


def _he_description(prompt: str) -> str:
    """Natural-language task description from a HumanEval stub: its docstring text."""
    try:
        tree = ast.parse(prompt)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node) or ""
                # Drop doctest-style example lines; keep the prose.
                lines = [
                    ln.strip()
                    for ln in doc.splitlines()
                    if ln.strip() and not ln.strip().startswith(">>>")
                ]
                return " ".join(lines)
    except SyntaxError:
        pass
    return ""


def _load_humaneval_rows() -> list[dict]:
    """HumanEval as MBPP-shaped rows: check(candidate) asserts -> test_list with the entry
    point substituted, reference = prompt + canonical_solution, NL prompt = docstring."""
    from datasets import load_dataset

    try:
        ds = load_dataset("openai_humaneval", split="test")
    except Exception:
        return []
    rows: list[dict] = []
    for r in ds:
        entry = r["entry_point"]
        desc = _he_description(r["prompt"])
        if not desc:
            continue
        # Pull `assert candidate(...) == ...` lines out of check() and rename the callee.
        tests: list[str] = []
        try:
            tree = ast.parse(r["test"])
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                stmt = ast.unparse(node)
                if "candidate(" in stmt:
                    tests.append(stmt.replace("candidate(", f"{entry}("))
        if len(tests) < 2:
            continue
        rows.append(
            {
                "task_id": f"he_{r['task_id'].split('/')[-1]}",
                "prompt": desc,
                "code": r["prompt"] + r["canonical_solution"],
                "test_imports": [],
                "test_list": tests,
            }
        )
    return rows


def build_corpus_tasks(
    seed: int = 0,
    eval_frac: float = 0.15,
    max_tasks: int | None = None,
    include_humaneval: bool = False,
) -> tuple[list[CogTask], list[CogTask]]:
    """Return (train, eval) Cog tasks generated from MBPP (optionally + HumanEval), split by
    problem.

    ``eval_frac`` of the usable problems are held out for eval. ``max_tasks`` caps the total
    (after filtering) for quick runs. ``include_humaneval`` appends HumanEval-derived tasks
    to the TRAIN side only — the MBPP shuffle/split is computed first and unchanged, so the
    held-out eval set stays identical to the MBPP-only benchmark.
    """
    rows = _load_rows()
    tasks: list[CogTask] = []
    # Many MBPP refs/tests embed unescaped regex strings; parsing/compiling them emits
    # SyntaxWarnings that are noise here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in rows:
            t = _build_one(r)
            if t is not None:
                tasks.append(t)
    # Deterministic order, then shuffle by seed so train/eval split is stable per seed.
    tasks.sort(key=lambda t: t.name)
    random.Random(seed).shuffle(tasks)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    n_eval = max(1, int(len(tasks) * eval_frac))
    eval_, train = tasks[:n_eval], tasks[n_eval:]
    if include_humaneval:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            he = [t for t in (_build_one(r) for r in _load_humaneval_rows()) if t is not None]
        he.sort(key=lambda t: t.name)
        train = train + he
    return train, eval_


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build/inspect MBPP-derived Cog tasks.")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()
    train, eval_ = build_corpus_tasks(max_tasks=args.max_tasks)
    print(f"usable MBPP tasks: {len(train)} train + {len(eval_)} eval")
    for t in train[: args.show] + eval_[: args.show]:
        print(f"\n[{t.name}] {t.prompt}")
        print(f"  tests: {t.tests}")
