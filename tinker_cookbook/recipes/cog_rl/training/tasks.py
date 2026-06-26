"""Task families for Cog RL — input-parameterized, graded on hidden inputs.

Each task asks the model to define a function ``forge solve(...)``. The prompt does not
contain concrete inputs; the grader (``grading.py``) calls ``solve`` on several hidden
inputs not shown to the model and checks each output. This is the Experiment 2 design:
it removes the Experiment 1 reward hack where the model just emitted the literal answer,
because a constant cannot satisfy multiple hidden inputs (see ``results/``).

Tasks are built by writing the reference ``solve`` in Cog and running it on each hidden
input to get the exact expected output, so targets are achievable and the checker is
exact. Train and eval use disjoint families to measure transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tinker_cookbook.recipes.cog_rl.agent_app.cog_lang import run_cog


@dataclass(frozen=True)
class CogTask:
    name: str
    family: str
    prompt: str
    # Hidden tests: each is (call-args literal, expected stdout of solve on those args).
    tests: tuple[tuple[str, str], ...]
    max_steps: int = 50_000
    metadata: dict = field(default_factory=dict)


def _lit(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    raise TypeError(type(v))


def _args(t: tuple) -> str:
    return ", ".join(_lit(a) for a in t)


# --------------------------------------------------------------------------- #
# Family definitions: (family, prompt, reference solve, hidden input tuples).
# --------------------------------------------------------------------------- #


def _sum_div(k: int):
    prompt = (
        f"Define `forge solve(n)` that returns the sum of every integer from 1 to n "
        f"inclusive that is divisible by {k}. It is tested on hidden values of n."
    )
    ref = f"""\
forge solve(n) {{
  0 -> s
  1 -> i
  sustain i <= n {{
    when i % {k} = 0 {{ s + i -> s }}
    i + 1 -> i
  }}
  give s
}}"""
    return prompt, ref, [(4,), (9,), (15,), (22,), (30,), (50,)]


def _count_div(k: int):
    prompt = (
        f"Define `forge solve(n)` that returns how many integers from 1 to n inclusive "
        f"are divisible by {k}. It is tested on hidden values of n."
    )
    ref = f"""\
forge solve(n) {{
  0 -> c
  1 -> i
  sustain i <= n {{
    when i % {k} = 0 {{ c + 1 -> c }}
    i + 1 -> i
  }}
  give c
}}"""
    return prompt, ref, [(5,), (10,), (18,), (27,), (40,)]


def _primes_below():
    prompt = (
        "Define `forge solve(n)` that returns how many prime numbers are strictly less "
        "than n. It is tested on hidden values of n."
    )
    ref = """\
forge is_prime(n) {
  when n < 2 { give no }
  2 -> d
  sustain d * d <= n {
    when n % d = 0 { give no }
    d + 1 -> d
  }
  give yes
}
forge solve(n) {
  0 -> c
  2 -> k
  sustain k < n {
    when is_prime(k) { c + 1 -> c }
    k + 1 -> k
  }
  give c
}"""
    return prompt, ref, [(3,), (6,), (12,), (20,), (30,), (50,)]


def _digit_sum():
    prompt = (
        "Define `forge solve(n)` that returns the sum of the decimal digits of n "
        "(n >= 0). It is tested on hidden values of n."
    )
    ref = """\
forge solve(n) {
  0 -> s
  sustain n > 0 {
    s + (n % 10) -> s
    n / 10 -> n
  }
  give s
}"""
    return prompt, ref, [(0,), (7,), (42,), (309,), (1234,), (90909,)]


def _digit_count():
    prompt = (
        "Define `forge solve(n)` that returns the number of decimal digits of n "
        "(n >= 0). It is tested on hidden values of n."
    )
    ref = """\
forge solve(n) {
  when n = 0 { give 1 }
  0 -> c
  sustain n > 0 {
    c + 1 -> c
    n / 10 -> n
  }
  give c
}"""
    return prompt, ref, [(0,), (5,), (42,), (1000,), (99999,)]


def _list_count_gt(t: int):
    prompt = (
        f"Define `forge solve(xs)`, where xs is a list of integers, that returns how "
        f"many elements are strictly greater than {t}. It is tested on hidden lists."
    )
    ref = f"""\
forge solve(xs) {{
  0 -> c
  walk v across xs {{ when v > {t} {{ c + 1 -> c }} }}
  give c
}}"""
    return (
        prompt,
        ref,
        [([1, 5, 8, 3],), ([10, 2, 7],), ([0, 0, 9, 9, 9],), ([5],), ([12, 3, 4, 15, 1],)],
    )


def _list_sum():
    prompt = (
        "Define `forge solve(xs)`, where xs is a list of integers, that returns their "
        "sum. It is tested on hidden lists."
    )
    ref = """\
forge solve(xs) {
  0 -> s
  walk v across xs { s + v -> s }
  give s
}"""
    return prompt, ref, [([1, 2, 3],), ([10, 20],), ([7],), ([4, 4, 4, 4],), ([0, 9, 1, 8],)]


def _list_reverse_join():
    prompt = (
        "Define `forge solve(xs)`, where xs is a list of integers, that returns the "
        "elements in reverse order joined by single spaces (as text). Tested on hidden lists."
    )
    ref = """\
forge solve(xs) {
  [] -> acc
  0 -> i
  sustain i < count(xs) {
    push(acc, xs @ (count(xs) - 1 - i)) -> acc
    i + 1 -> i
  }
  give join(acc, " ")
}"""
    return prompt, ref, [([1, 2, 3],), ([9, 8, 7, 6],), ([5],), ([2, 4, 6, 8],)]


def _vowel_count():
    prompt = (
        "Define `forge solve(w)`, where w is lowercase text, that returns the number of "
        "vowels (a, e, i, o, u) in w. It is tested on hidden words."
    )
    ref = """\
forge solve(w) {
  0 -> c
  walk ch across chars(w) {
    when ch = "a" either ch = "e" either ch = "i" either ch = "o" either ch = "u" {
      c + 1 -> c
    }
  }
  give c
}"""
    return prompt, ref, [("tokenizer",), ("rhythm",), ("aeiou",), ("gradient",), ("xyz",)]


# Extra families added in Experiment 4: transferable structure for the held-out families
# that lagged in Experiment 3 (list reduce -> list_max; accumulate/while-with-branch ->
# nth_fib/lcm; two-index text scan -> palindrome). None overlap the eval families.


def _list_min():
    prompt = (
        "Define `forge solve(xs)`, where xs is a non-empty list of integers, that returns "
        "the smallest element. It is tested on hidden lists."
    )
    ref = """\
forge solve(xs) {
  head(xs) -> m
  walk v across xs { when v < m { v -> m } }
  give m
}"""
    return prompt, ref, [([3, 1, 2],), ([9],), ([4, 4, 0, 2],), ([7, 8, 1],), ([5, 6, 2, 12, 3],)]


def _list_product():
    prompt = (
        "Define `forge solve(xs)`, where xs is a non-empty list of integers, that returns "
        "the product of all elements. It is tested on hidden lists."
    )
    ref = """\
forge solve(xs) {
  1 -> p
  walk v across xs { p * v -> p }
  give p
}"""
    return prompt, ref, [([1, 2, 3],), ([5],), ([2, 2, 2],), ([4, 0, 9],), ([3, 3, 3, 3],)]


def _pow2():
    prompt = (
        "Define `forge solve(n)` that returns 2 raised to the power n (n >= 0, "
        "solve(0) = 1). It is tested on hidden values of n."
    )
    ref = """\
forge solve(n) {
  1 -> p
  0 -> i
  sustain i < n {
    p * 2 -> p
    i + 1 -> i
  }
  give p
}"""
    return prompt, ref, [(0,), (1,), (4,), (8,), (10,), (13,)]


def _collatz_steps():
    prompt = (
        "Define `forge solve(n)` that returns how many steps it takes to reach 1 from "
        "n >= 1 under the Collatz rule (if even, halve it; if odd, 3n+1). solve(1) = 0. "
        "It is tested on hidden values of n."
    )
    ref = """\
forge solve(n) {
  0 -> c
  sustain n != 1 {
    when n % 2 = 0 { n / 2 -> n }
    otherwise { 3 * n + 1 -> n }
    c + 1 -> c
  }
  give c
}"""
    return prompt, ref, [(1,), (2,), (3,), (6,), (7,), (27,)]


def _lucas():
    prompt = (
        "Define `forge solve(n)` that returns the nth Lucas number, where L(0) = 2, "
        "L(1) = 1, and L(k) = L(k-1) + L(k-2). It is tested on hidden values of n."
    )
    ref = """\
forge solve(n) {
  2 -> a
  1 -> b
  0 -> i
  sustain i < n {
    a + b -> t
    b -> a
    t -> b
    i + 1 -> i
  }
  give a
}"""
    return prompt, ref, [(0,), (1,), (2,), (5,), (7,), (10,)]


def _adjacent_dup_count():
    prompt = (
        "Define `forge solve(w)`, where w is lowercase text, that returns how many adjacent "
        'character pairs are equal (e.g. "aabb" has 2). It is tested on hidden words.'
    )
    ref = """\
forge solve(w) {
  chars(w) -> cs
  0 -> c
  1 -> i
  sustain i < count(cs) {
    when cs @ (i - 1) = cs @ i { c + 1 -> c }
    i + 1 -> i
  }
  give c
}"""
    return prompt, ref, [("letter",), ("aabb",), ("abc",), ("bookkeeper",), ("mississippi",)]


# Experiment 6 additions: keyed-lookup / accumulate-by-rebuild shapes. Cog has no map/set
# type, so these force the parallel-list and seen-list idioms the handbook teaches. They are
# the shapes the model failed to generalize to before (e.g. "most frequent element").


def _most_frequent():
    prompt = (
        "Define `forge solve(xs)`, where xs is a non-empty list of integers, that returns the "
        "element that occurs most often. If several tie, return the one that first reaches that "
        "count (earliest in the list). It is tested on hidden lists."
    )
    ref = """\
forge index_of(xs, t) {
  0 -> i
  walk x across xs { when x = t { give i } i + 1 -> i }
  give -1
}
forge solve(xs) {
  [] -> keys
  [] -> counts
  walk it across xs {
    index_of(keys, it) -> pos
    when pos = -1 { push(keys, it) -> keys  push(counts, 1) -> counts }
    otherwise {
      [] -> nc
      0 -> j
      walk c across counts {
        when j = pos { push(nc, c + 1) -> nc } otherwise { push(nc, c) -> nc }
        j + 1 -> j
      }
      nc -> counts
    }
  }
  head(keys) -> best
  head(counts) -> bestc
  0 -> k
  walk c across counts {
    when c > bestc { c -> bestc  keys @ k -> best }
    k + 1 -> k
  }
  give best
}"""
    return prompt, ref, [([1, 2, 2, 3],), ([5, 5, 5, 1],), ([7],), ([4, 4, 5, 5, 5],), ([9, 1, 1],)]


def _count_distinct():
    prompt = (
        "Define `forge solve(xs)`, where xs is a list of integers, that returns how many "
        "distinct values it contains. It is tested on hidden lists."
    )
    ref = """\
forge seen(xs, t) {
  walk x across xs { when x = t { give yes } }
  give no
}
forge solve(xs) {
  [] -> u
  walk v across xs { when flip(seen(u, v)) { push(u, v) -> u } }
  give count(u)
}"""
    return prompt, ref, [([1, 2, 2, 3],), ([5, 5, 5],), ([1, 2, 3, 4],), ([7, 7, 1],), ([9],)]


def _dedupe_join():
    prompt = (
        "Define `forge solve(xs)`, where xs is a list of integers, that removes duplicates "
        "keeping the first occurrence of each value, then returns the kept values joined by "
        "single spaces (as text). It is tested on hidden lists."
    )
    ref = """\
forge seen(xs, t) {
  walk x across xs { when x = t { give yes } }
  give no
}
forge solve(xs) {
  [] -> u
  walk v across xs { when flip(seen(u, v)) { push(u, v) -> u } }
  give join(u, " ")
}"""
    return prompt, ref, [([1, 2, 2, 3],), ([5, 5, 5],), ([1, 2, 3],), ([3, 3, 1, 1, 2],)]


def _gcd_steps():
    prompt = (
        "Define `forge solve(a, b)` that returns how many division steps the Euclidean "
        "algorithm takes on non-negative integers a and b: repeatedly replace (a, b) with "
        "(b, a mod b), counting each step, until b is 0. It is tested on hidden pairs."
    )
    ref = """\
forge solve(a, b) {
  0 -> c
  sustain b != 0 {
    a % b -> r
    b -> a
    r -> b
    c + 1 -> c
  }
  give c
}"""
    return prompt, ref, [(48, 36), (109, 446), (100, 75), (17, 5), (60, 24)]


# ---- eval families (disjoint) ----


def _mode_count():
    prompt = (
        "Define `forge solve(xs)`, where xs is a non-empty list of integers, that returns how "
        "many times the most frequent element occurs (the count, not the element). It is tested "
        "on hidden lists."
    )
    ref = """\
forge index_of(xs, t) {
  0 -> i
  walk x across xs { when x = t { give i } i + 1 -> i }
  give -1
}
forge solve(xs) {
  [] -> keys
  [] -> counts
  walk it across xs {
    index_of(keys, it) -> pos
    when pos = -1 { push(keys, it) -> keys  push(counts, 1) -> counts }
    otherwise {
      [] -> nc
      0 -> j
      walk c across counts {
        when j = pos { push(nc, c + 1) -> nc } otherwise { push(nc, c) -> nc }
        j + 1 -> j
      }
      nc -> counts
    }
  }
  0 -> best
  walk c across counts { when c > best { c -> best } }
  give best
}"""
    return prompt, ref, [([1, 2, 2, 3],), ([5, 5, 5, 1],), ([7],), ([4, 4, 5, 5, 5],), ([9, 1, 1],)]


def _first_repeat():
    prompt = (
        "Define `forge solve(xs)`, where xs is a list of integers, that returns the first value "
        "that appears more than once (the first element equal to some earlier element). If every "
        "value is unique, return void. It is tested on hidden lists."
    )
    ref = """\
forge seen(xs, t) {
  walk x across xs { when x = t { give yes } }
  give no
}
forge solve(xs) {
  [] -> u
  walk v across xs {
    when seen(u, v) { give v }
    push(u, v) -> u
  }
  give void
}"""
    return prompt, ref, [([1, 2, 2, 3],), ([1, 2, 3],), ([5, 5],), ([1, 2, 3, 1],), ([9, 8, 7],)]


def _gcd():
    prompt = (
        "Define `forge solve(a, b)` that returns the greatest common divisor of "
        "non-negative integers a and b. It is tested on hidden pairs."
    )
    ref = """\
forge solve(a, b) {
  sustain b != 0 {
    a % b -> r
    b -> a
    r -> b
  }
  give a
}"""
    return prompt, ref, [(48, 36), (109, 446), (100, 75), (17, 5), (60, 24), (81, 27)]


def _nth_fib():
    prompt = (
        "Define `forge solve(n)` that returns the nth Fibonacci number, where "
        "fib(0) = 0 and fib(1) = 1. It is tested on hidden values of n."
    )
    ref = """\
forge solve(n) {
  0 -> a
  1 -> b
  0 -> i
  sustain i < n {
    a + b -> t
    b -> a
    t -> b
    i + 1 -> i
  }
  give a
}"""
    return prompt, ref, [(0,), (1,), (2,), (5,), (7,), (10,), (15,)]


def _factorial():
    prompt = (
        "Define `forge solve(n)` that returns n factorial (the product 1*2*...*n, with "
        "solve(0) = 1). It is tested on hidden values of n."
    )
    ref = """\
forge solve(n) {
  1 -> p
  1 -> i
  sustain i <= n {
    p * i -> p
    i + 1 -> i
  }
  give p
}"""
    return prompt, ref, [(0,), (1,), (3,), (5,), (7,), (9,)]


def _palindrome():
    prompt = (
        "Define `forge solve(w)`, where w is lowercase text, that returns yes if w is a "
        "palindrome (reads the same forwards and backwards), otherwise no. Tested on hidden words."
    )
    ref = """\
forge solve(w) {
  chars(w) -> cs
  [] -> rev
  0 -> i
  sustain i < count(cs) {
    push(rev, cs @ (count(cs) - 1 - i)) -> rev
    i + 1 -> i
  }
  when join(rev, "") = w { give yes } otherwise { give no }
}"""
    return prompt, ref, [("kayak",), ("hello",), ("racecar",), ("cog",), ("noon",), ("abc",)]


def _reverse_text():
    prompt = (
        "Define `forge solve(w)`, where w is lowercase text, that returns w reversed (as "
        "text). It is tested on hidden words."
    )
    ref = """\
forge solve(w) {
  chars(w) -> cs
  [] -> rev
  0 -> i
  sustain i < count(cs) {
    push(rev, cs @ (count(cs) - 1 - i)) -> rev
    i + 1 -> i
  }
  give join(rev, "")
}"""
    return prompt, ref, [("tinker",), ("abc",), ("level",), ("policy",), ("zz",), ("a",)]


def _lcm():
    prompt = (
        "Define `forge solve(a, b)` that returns the least common multiple of positive "
        "integers a and b. It is tested on hidden pairs."
    )
    ref = """\
forge gcd(a, b) {
  sustain b != 0 {
    a % b -> r
    b -> a
    r -> b
  }
  give a
}
forge solve(a, b) {
  gcd(a, b) -> g
  give a / g * b
}"""
    return prompt, ref, [(4, 6), (3, 5), (12, 8), (7, 7), (9, 6), (10, 4)]


def _list_max():
    prompt = (
        "Define `forge solve(xs)`, where xs is a non-empty list of integers, that returns "
        "the largest element. It is tested on hidden lists."
    )
    ref = """\
forge solve(xs) {
  head(xs) -> m
  walk v across xs { when v > m { v -> m } }
  give m
}"""
    return prompt, ref, [([3, 1, 2],), ([9],), ([4, 4, 9, 2],), ([0, 8, 1],), ([7, 5, 6, 12, 3],)]


def _is_sorted():
    prompt = (
        "Define `forge solve(xs)`, where xs is a list of integers, that returns yes if the "
        "list is sorted in non-decreasing order, otherwise no. Tested on hidden lists."
    )
    ref = """\
forge solve(xs) {
  1 -> i
  sustain i < count(xs) {
    when xs @ (i - 1) > xs @ i { give no }
    i + 1 -> i
  }
  give yes
}"""
    return prompt, ref, [([1, 2, 3],), ([3, 1, 2],), ([5, 5, 6],), ([9, 8],), ([1],), ([2, 2, 1],)]


def _count_char():
    prompt = (
        "Define `forge solve(w, ch)`, where w is lowercase text and ch is a single "
        "character, that returns how many times ch occurs in w. Tested on hidden inputs."
    )
    ref = """\
forge solve(w, ch) {
  0 -> c
  walk x across chars(w) { when x = ch { c + 1 -> c } }
  give c
}"""
    return (
        prompt,
        ref,
        [("banana", "a"), ("mississippi", "s"), ("cog", "l"), ("abc", "z"), ("aaa", "a")],
    )


def _power():
    prompt = (
        "Define `forge solve(b, e)` that returns b raised to the power e (e >= 0, "
        "solve(b, 0) = 1). It is tested on hidden pairs."
    )
    ref = """\
forge solve(b, e) {
  1 -> p
  0 -> i
  sustain i < e {
    p * b -> p
    i + 1 -> i
  }
  give p
}"""
    return prompt, ref, [(2, 0), (2, 5), (3, 3), (5, 2), (10, 4), (7, 1)]


# Each entry: (family_name, list of (prompt, reference, inputs) task specs).
def _train_specs():
    specs = []
    for k in range(1, 8):
        specs.append(("sum_div", _sum_div(k)))
    for k in (2, 3, 4):
        specs.append(("count_div", _count_div(k)))
    specs.append(("primes_below", _primes_below()))
    specs.append(("digit_sum", _digit_sum()))
    specs.append(("digit_count", _digit_count()))
    for t in (2, 4, 6, 8):
        specs.append(("list_count_gt", _list_count_gt(t)))
    specs.append(("list_sum", _list_sum()))
    specs.append(("list_reverse_join", _list_reverse_join()))
    specs.append(("vowel_count", _vowel_count()))
    # Experiment 4 additions (transferable structure for the lagging held-out families).
    specs.append(("list_min", _list_min()))
    specs.append(("list_product", _list_product()))
    specs.append(("pow2", _pow2()))
    specs.append(("collatz_steps", _collatz_steps()))
    specs.append(("adjacent_dup_count", _adjacent_dup_count()))
    # Experiment 5: a two-state recurrence (Lucas) — same structure as the held-out nth_fib.
    specs.append(("lucas", _lucas()))
    # Experiment 6: keyed-lookup / accumulate-by-rebuild shapes (the generalization gap). The
    # held-out mode_count/first_repeat reuse these exact idioms on different tasks.
    specs.append(("most_frequent", _most_frequent()))
    specs.append(("count_distinct", _count_distinct()))
    specs.append(("dedupe_join", _dedupe_join()))
    specs.append(("gcd_steps", _gcd_steps()))
    return specs


def _eval_specs():
    return [
        ("gcd", _gcd()),
        ("nth_fib", _nth_fib()),
        ("factorial", _factorial()),
        ("palindrome", _palindrome()),
        ("reverse_text", _reverse_text()),
        ("lcm", _lcm()),
        ("list_max", _list_max()),
        ("is_sorted", _is_sorted()),
        ("count_char", _count_char()),
        ("power", _power()),
        # Experiment 6 held-out keyed-lookup shapes: same idioms as the new train families
        # (parallel keys/counts, seen-list), different task, to measure shape transfer.
        ("mode_count", _mode_count()),
        ("first_repeat", _first_repeat()),
    ]


TRAIN_FAMILIES = sorted({f for f, _ in _train_specs()})
EVAL_FAMILIES = sorted({f for f, _ in _eval_specs()})


def _build(specs) -> list[CogTask]:
    tasks: list[CogTask] = []
    family_counts: dict[str, int] = {}
    for family, (prompt, ref, inputs) in specs:
        tests: list[tuple[str, str]] = []
        for args in inputs:
            arg_str = _args(args)
            res = run_cog(ref + f"\nemit solve({arg_str})")
            if not res.ok or not res.output:
                raise AssertionError(f"reference for {family} failed on {arg_str}: {res.error}")
            tests.append((arg_str, res.output))
        # A constant must not satisfy the task: require at least two distinct expected
        # outputs across the hidden inputs.
        if len({e for _, e in tests}) < 2:
            raise AssertionError(f"{family} hidden inputs are not discriminative (all equal)")
        idx = family_counts.get(family, 0)
        family_counts[family] = idx + 1
        tasks.append(
            CogTask(name=f"{family}_{idx}", family=family, prompt=prompt, tests=tuple(tests))
        )
    return tasks


def build_tasks(seed: int = 0) -> tuple[list[CogTask], list[CogTask]]:
    import random

    train = _build(_train_specs())
    eval_ = _build(_eval_specs())
    random.Random(seed).shuffle(train)
    return train, eval_


def get_tasks(source: str = "families", seed: int = 0) -> tuple[list[CogTask], list[CogTask]]:
    """Return (train, eval) tasks from the chosen source.

    - ``families``: the hand-authored, guaranteed-Cog-solvable families (default).
    - ``corpus``: tasks generated from the MBPP corpus, graded on the corpus's own I/O
      (no hand-written Cog; shape coverage comes from corpus diversity). Requires the
      ``datasets`` package and network access on first load.
    - ``both``: the union (corpus train + family train; eval kept separate per source).
    """
    if source == "families":
        return build_tasks(seed=seed)
    from tinker_cookbook.recipes.cog_rl.training.corpus_tasks import build_corpus_tasks

    if source == "corpus":
        return build_corpus_tasks(seed=seed)
    if source == "both":
        ftrain, feval = build_tasks(seed=seed)
        ctrain, ceval = build_corpus_tasks(seed=seed)
        return ftrain + ctrain, feval + ceval
    raise ValueError(f"unknown task source: {source!r} (want families|corpus|both)")


if __name__ == "__main__":
    train, eval_ = build_tasks()
    print(f"train: {len(train)} tasks across {len(TRAIN_FAMILIES)} families")
    print(f"eval:  {len(eval_)} tasks across {len(EVAL_FAMILIES)} families")
    t = eval_[0]
    print(f"\n[{t.family}] {t.name}\n  {t.prompt}\n  tests: {t.tests}")
