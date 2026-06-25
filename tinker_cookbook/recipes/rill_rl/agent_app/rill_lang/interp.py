"""
RILL interpreter — a deterministic tree-walking evaluator.

Public API:
    run_rill(src: str, *, max_steps: int = 100_000) -> Result

Result is a dataclass with:
    ok:     bool                # ran to completion without error
    output: str                 # everything `emit`-ed, newline-joined
    error:  str | None          # category:message, e.g. "parse:..." / "runtime:..." / "budget:..."
    steps:  int                 # statements executed (for cost accounting)

Design notes for RL use:
    * Fully deterministic: no clocks, no RNG, no I/O except captured `emit`.
    * Hard step budget: guards against infinite loops eating rollout compute.
    * Error categories are coarse + stable so they can drive reward shaping.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from lark import Lark, Tree, Token

_GRAMMAR = (Path(__file__).parent / "rill.lark").read_text()
_PARSER = Lark(_GRAMMAR, parser="lalr", maybe_placeholders=True)


# ---- control-flow signals (used internally, never leak to the user) ----
class _Give(Exception):
    def __init__(self, value): self.value = value
class _Halt(Exception): pass
class _Skip(Exception): pass


class RillError(Exception):
    """Runtime error inside a RILL program."""


@dataclass
class Result:
    ok: bool
    output: str
    error: str | None
    steps: int


class _Func:
    __slots__ = ("name", "params", "body", "closure")
    def __init__(self, name, params, body, closure):
        self.name, self.params, self.body, self.closure = name, params, body, closure


class _Env:
    __slots__ = ("vars", "parent")
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def get(self, name):
        e = self
        while e is not None:
            if name in e.vars:
                return e.vars[name]
            e = e.parent
        raise RillError(f"unbound name '{name}'")

    def set(self, name, value):
        # assignment walks up to an existing binding, else binds locally
        e = self
        while e is not None:
            if name in e.vars:
                e.vars[name] = value
                return
            e = e.parent
        self.vars[name] = value


class Interpreter:
    def __init__(self, max_steps: int):
        self.out: list[str] = []
        self.steps = 0
        self.max_steps = max_steps

    # ---- the driver ----
    def run(self, tree: Tree):
        root = _Env()
        # hoist function definitions so order of declaration doesn't matter
        for stmt in tree.children:
            if isinstance(stmt, Tree) and stmt.data == "forge_stmt":
                self._exec(stmt, root)
        for stmt in tree.children:
            if not (isinstance(stmt, Tree) and stmt.data == "forge_stmt"):
                self._exec(stmt, root)

    def _tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise RillError("step budget exceeded")  # caught & retagged as budget

    # ---- statement execution ----
    def _exec(self, node: Tree, env: _Env):
        self._tick()
        kind = node.data

        if kind == "bind":
            value = self._eval(node.children[0], env)
            name = str(node.children[1])
            env.set(name, value)

        elif kind == "emit_stmt":
            self.out.append(_to_str(self._eval(node.children[0], env)))

        elif kind == "expr_stmt":
            self._eval(node.children[0], env)

        elif kind == "give_stmt":
            arg = node.children[0]
            raise _Give(None if arg is None else self._eval(arg, env))

        elif kind == "halt_stmt":
            raise _Halt()

        elif kind == "skip_stmt":
            raise _Skip()

        elif kind == "when_stmt":
            self._exec_when(node, env)

        elif kind == "sustain_stmt":
            cond_node, block = node.children[0], node.children[1]
            while _truthy(self._eval(cond_node, env)):
                self._tick()
                try:
                    self._exec_block(block, _Env(env))
                except _Halt:
                    break
                except _Skip:
                    continue

        elif kind == "walk_stmt":
            var = str(node.children[0])
            seq = self._eval(node.children[1], env)
            block = node.children[2]
            if not isinstance(seq, list):
                raise RillError("'walk across' expects a list")
            for item in seq:
                self._tick()
                loop_env = _Env(env)
                loop_env.vars[var] = item
                try:
                    self._exec_block(block, loop_env)
                except _Halt:
                    break
                except _Skip:
                    continue

        elif kind == "forge_stmt":
            name = str(node.children[0])
            params_node = node.children[1]
            param_names = (
                [str(t) for t in params_node.children] if params_node is not None else []
            )
            body = node.children[2]
            env.vars[name] = _Func(name, param_names, body, env)

        else:
            raise RillError(f"unknown statement: {kind}")

    def _exec_when(self, node: Tree, env: _Env):
        children = node.children
        cond, block = children[0], children[1]
        if _truthy(self._eval(cond, env)):
            return self._exec_block(block, _Env(env))
        for child in children[2:]:
            if child is None:
                continue
            if child.data == "elsewhen":
                if _truthy(self._eval(child.children[0], env)):
                    return self._exec_block(child.children[1], _Env(env))
            elif child.data == "otherwise":
                return self._exec_block(child.children[0], _Env(env))

    def _exec_block(self, block: Tree, env: _Env):
        for stmt in block.children:
            self._exec(stmt, env)

    # ---- expression evaluation ----
    def _eval(self, node, env: _Env):
        if isinstance(node, Token):
            raise RillError(f"stray token {node!r}")
        d = node.data

        if d == "int":   return int(node.children[0])
        if d == "float": return float(node.children[0])
        if d == "string": return _unescape(str(node.children[0])[1:-1])
        if d == "true":  return True
        if d == "false": return False
        if d == "nil":   return None
        if d == "var":   return env.get(str(node.children[0]))

        if d == "list":
            return [self._eval(c, env) for c in node.children if c is not None]

        if d == "neg":  return -_num(self._eval(node.children[0], env))
        if d == "flip": return not _truthy(self._eval(node.children[0], env))

        if d == "add":
            a, b = self._eval(node.children[0], env), self._eval(node.children[1], env)
            if isinstance(a, str) or isinstance(b, str):
                return _to_str(a) + _to_str(b)        # '+' also concatenates strings
            if isinstance(a, list) and isinstance(b, list):
                return a + b                          # ... and joins lists
            return _num(a) + _num(b)
        if d == "sub": return _num(self._eval(node.children[0], env)) - _num(self._eval(node.children[1], env))
        if d == "mul": return _num(self._eval(node.children[0], env)) * _num(self._eval(node.children[1], env))
        if d == "div":
            b = _num(self._eval(node.children[1], env))
            if b == 0: raise RillError("division by zero")
            a = _num(self._eval(node.children[0], env))
            return a // b if isinstance(a, int) and isinstance(b, int) else a / b
        if d == "mod":
            b = _num(self._eval(node.children[1], env))
            if b == 0: raise RillError("modulo by zero")
            return _num(self._eval(node.children[0], env)) % b

        if d == "or_expr":
            for c in node.children:
                if _truthy(self._eval(c, env)): return True
            return False
        if d == "and_expr":
            result = True
            for c in node.children:
                if not _truthy(self._eval(c, env)): return False
            return True

        if d == "cmp_expr":
            left = self._eval(node.children[0], env)
            op = node.children[1].data
            right = self._eval(node.children[2], env)
            return _compare(op, left, right)

        if d == "index":
            target = self._eval(node.children[0], env)
            idx = self._eval(node.children[1], env)
            return _index(target, idx)

        if d == "call":
            return self._eval_call(node, env)

        raise RillError(f"unknown expression: {d}")

    def _eval_call(self, node: Tree, env: _Env):
        callee_node = node.children[0]
        args_node = node.children[1]
        arg_nodes = args_node.children if args_node is not None else []
        args = [self._eval(a, env) for a in arg_nodes]

        # builtins are dispatched by name when the callee is a bare variable
        if isinstance(callee_node, Tree) and callee_node.data == "var":
            name = str(callee_node.children[0])
            if name in BUILTINS:
                return BUILTINS[name](self, args)
            value = env.get(name)
        else:
            value = self._eval(callee_node, env)

        if not isinstance(value, _Func):
            raise RillError("attempt to call a non-function")
        if len(args) != len(value.params):
            raise RillError(
                f"'{value.name}' expects {len(value.params)} arg(s), got {len(args)}"
            )
        call_env = _Env(value.closure)
        for p, a in zip(value.params, args):
            call_env.vars[p] = a
        try:
            self._exec_block(value.body, call_env)
        except _Give as g:
            return g.value
        return None


# ---- runtime helpers ----
def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RillError(f"expected a number, got {_type(v)}")
    return v

def _truthy(v):
    if v is None or v is False: return False
    if v is True: return True
    if isinstance(v, (int, float)): return v != 0
    if isinstance(v, str): return len(v) > 0
    if isinstance(v, list): return len(v) > 0
    return True

def _compare(op, a, b):
    if op in ("eq", "neq"):
        same = a == b and type(a) == type(b)
        return same if op == "eq" else not same
    # ordering only for numbers and strings of matching kind
    if isinstance(a, str) and isinstance(b, str):
        pass
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        pass
    else:
        raise RillError(f"cannot order {_type(a)} and {_type(b)}")
    return {"lt": a < b, "gt": a > b, "le": a <= b, "ge": a >= b}[op]

def _index(target, idx):
    if isinstance(target, (list, str)):
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise RillError("index must be an integer")
        if idx < 0 or idx >= len(target):
            raise RillError(f"index {idx} out of range (size {len(target)})")
        return target[idx]
    raise RillError(f"cannot index {_type(target)}")

def _type(v):
    if v is None: return "void"
    if isinstance(v, bool): return "flag"
    if isinstance(v, int): return "int"
    if isinstance(v, float): return "real"
    if isinstance(v, str): return "text"
    if isinstance(v, list): return "list"
    if isinstance(v, _Func): return "rite"
    return "unknown"

def _to_str(v):
    if v is None: return "void"
    if v is True: return "yes"
    if v is False: return "no"
    if isinstance(v, list): return "[" + ", ".join(_to_str(x) for x in v) + "]"
    return str(v)

def _unescape(s):
    return (s.replace("\\n", "\n").replace("\\t", "\t")
             .replace('\\"', '"').replace("\\\\", "\\"))


# ---- builtin library ----
def _b_count(it, args):
    (x,) = _arity(args, 1, "count")
    if isinstance(x, (list, str)): return len(x)
    raise RillError("count expects a list or text")

def _b_range(it, args):
    if len(args) == 1:
        n = args[0]; _need_int(n, "range")
        return list(range(n))
    if len(args) == 2:
        a, b = args; _need_int(a, "range"); _need_int(b, "range")
        return list(range(a, b))
    raise RillError("range expects 1 or 2 ints")

def _b_push(it, args):
    xs, v = _arity(args, 2, "push")
    if not isinstance(xs, list): raise RillError("push expects a list")
    return xs + [v]

def _b_head(it, args):
    (xs,) = _arity(args, 1, "head")
    if not isinstance(xs, list) or not xs: raise RillError("head of empty/non-list")
    return xs[0]

def _b_tail(it, args):
    (xs,) = _arity(args, 1, "tail")
    if not isinstance(xs, list): raise RillError("tail expects a list")
    return xs[1:]

def _b_slice(it, args):
    xs, a, b = _arity(args, 3, "slice")
    if not isinstance(xs, (list, str)): raise RillError("slice expects list/text")
    _need_int(a, "slice"); _need_int(b, "slice")
    return xs[a:b]

def _b_upper(it, args):
    (s,) = _arity(args, 1, "upper")
    if not isinstance(s, str): raise RillError("upper expects text")
    return s.upper()

def _b_lower(it, args):
    (s,) = _arity(args, 1, "lower")
    if not isinstance(s, str): raise RillError("lower expects text")
    return s.lower()

def _b_chars(it, args):
    (s,) = _arity(args, 1, "chars")
    if not isinstance(s, str): raise RillError("chars expects text")
    return list(s)

def _b_join(it, args):
    xs, sep = _arity(args, 2, "join")
    if not isinstance(xs, list) or not isinstance(sep, str):
        raise RillError("join expects (list, text)")
    return sep.join(_to_str(x) for x in xs)

def _b_abs(it, args):
    (x,) = _arity(args, 1, "abs"); return abs(_num(x))

def _b_max(it, args):
    a, b = _arity(args, 2, "max"); return a if _num(a) >= _num(b) else b

def _b_min(it, args):
    a, b = _arity(args, 2, "min"); return a if _num(a) <= _num(b) else b

def _b_int(it, args):
    (x,) = _arity(args, 1, "int")
    if isinstance(x, bool): raise RillError("int expects number/text")
    if isinstance(x, (int, float)): return int(x)
    if isinstance(x, str):
        try: return int(x)
        except ValueError: raise RillError(f"cannot read int from {x!r}")
    raise RillError("int expects number/text")

def _arity(args, n, name):
    if len(args) != n:
        raise RillError(f"{name} expects {n} arg(s), got {len(args)}")
    return args

def _need_int(v, name):
    if not isinstance(v, int) or isinstance(v, bool):
        raise RillError(f"{name} expects integer(s)")

BUILTINS = {
    "count": _b_count, "range": _b_range, "push": _b_push, "head": _b_head,
    "tail": _b_tail, "slice": _b_slice, "upper": _b_upper, "lower": _b_lower,
    "chars": _b_chars, "join": _b_join, "abs": _b_abs, "max": _b_max,
    "min": _b_min, "int": _b_int,
}


# ---- top-level entry point ----
def run_rill(src: str, *, max_steps: int = 100_000) -> Result:
    try:
        tree = _PARSER.parse(src)
    except Exception as e:
        return Result(ok=False, output="", error=f"parse:{_oneline(e)}", steps=0)

    interp = Interpreter(max_steps=max_steps)
    try:
        interp.run(tree)
    except _Give:
        pass  # `give` at top level just stops the program
    except (_Halt, _Skip):
        # A stray `halt`/`skip` with no enclosing loop is a runtime error, not a crash.
        return Result(ok=False, output="\n".join(interp.out),
                      error="runtime:halt/skip outside a loop", steps=interp.steps)
    except RillError as e:
        msg = str(e)
        cat = "budget" if "step budget" in msg else "runtime"
        return Result(ok=False, output="\n".join(interp.out),
                      error=f"{cat}:{msg}", steps=interp.steps)
    except RecursionError:
        return Result(ok=False, output="\n".join(interp.out),
                      error="runtime:call depth exceeded", steps=interp.steps)
    return Result(ok=True, output="\n".join(interp.out), error=None, steps=interp.steps)


def _oneline(e):
    return " ".join(str(e).split())[:200]


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python interp.py <file.rill>")
        sys.exit(2)
    res = run_rill(Path(sys.argv[1]).read_text())
    if res.output:
        print(res.output)
    if not res.ok:
        print(f"[{res.error}]", file=sys.stderr)
        sys.exit(1)
