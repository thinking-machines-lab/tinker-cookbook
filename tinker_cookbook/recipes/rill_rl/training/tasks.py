"""Task-family generators for RILL RL.

Each task is built the cheap, reliable way recommended by the language pack: write
the reference solution *in RILL*, run it through ``run_rill``, and use its captured
output as the exact ``expect``. That guarantees the target is achievable and the
checker is exact.

To measure genuine generalization of the learned syntax (rather than memorization of
seen prompts), we hold out disjoint task *families* for eval — not just different
constants within a family. ``TRAIN_FAMILIES`` and ``EVAL_FAMILIES`` are disjoint.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from tinker_cookbook.recipes.rill_rl.agent_app.rill_lang import run_rill

from .grading import RillTask

# A family produces (prompt, reference_solution_src) for a given parameter value.
FamilyFn = Callable[[object], tuple[str, str]]


def _rill_list(xs: list[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


# --------------------------------------------------------------------------- #
# Family definitions. Each returns (natural_language_prompt, reference RILL src).
# --------------------------------------------------------------------------- #


def _fam_sum_to_n(n: int) -> tuple[str, str]:
    prompt = f"Emit the sum of the integers from 1 to {n} inclusive (a single number)."
    src = f"""\
0 -> s
walk k across range(1, {n + 1}) {{ s + k -> s }}
emit s"""
    return prompt, src


def _fam_fizzbuzz(n: int) -> tuple[str, str]:
    prompt = (
        f"For each integer from 1 to {n} inclusive, emit (one per line): "
        '"fizzbuzz" if it is divisible by both 3 and 5, otherwise "fizz" if '
        'divisible by 3, otherwise "buzz" if divisible by 5, otherwise the number.'
    )
    src = f"""\
walk x across range(1, {n + 1}) {{
  when x % 15 = 0 {{ emit "fizzbuzz" }}
  elsewhen x % 3 = 0 {{ emit "fizz" }}
  elsewhen x % 5 = 0 {{ emit "buzz" }}
  otherwise {{ emit x }}
}}"""
    return prompt, src


def _fam_primes_below(n: int) -> tuple[str, str]:
    prompt = f"Emit every prime number below {n}, one per line, in increasing order."
    src = f"""\
forge is_prime(n) {{
  when n < 2 {{ give no }}
  2 -> d
  sustain d * d <= n {{
    when n % d = 0 {{ give no }}
    d + 1 -> d
  }}
  give yes
}}
walk k across range(2, {n}) {{ when is_prime(k) {{ emit k }} }}"""
    return prompt, src


def _fam_list_reverse(xs: list[int]) -> tuple[str, str]:
    prompt = (
        f"Given the list {_rill_list(xs)}, emit its elements in reverse order on a "
        "single line, joined by single spaces."
    )
    src = f"""\
{_rill_list(xs)} -> xs
[] -> acc
0 -> i
sustain i < count(xs) {{
  push(acc, xs @ (count(xs) - 1 - i)) -> acc
  i + 1 -> i
}}
emit join(acc, " ")"""
    return prompt, src


def _fam_count_predicate(args: tuple[list[int], int]) -> tuple[str, str]:
    xs, threshold = args
    prompt = (
        f"Given the list {_rill_list(xs)}, emit how many of its elements are "
        f"strictly greater than {threshold} (a single number)."
    )
    src = f"""\
{_rill_list(xs)} -> xs
0 -> c
walk v across xs {{ when v > {threshold} {{ c + 1 -> c }} }}
emit c"""
    return prompt, src


def _fam_digit_sum(n: int) -> tuple[str, str]:
    prompt = f"Emit the sum of the decimal digits of {n} (a single number)."
    src = f"""\
{n} -> n
0 -> s
sustain n > 0 {{
  s + (n % 10) -> s
  n / 10 -> n
}}
emit s"""
    return prompt, src


def _fam_vowel_count(word: str) -> tuple[str, str]:
    prompt = (
        f'Emit how many vowels (a, e, i, o, u) appear in the lowercase word "{word}" '
        "(a single number)."
    )
    src = f"""\
"{word}" -> w
0 -> c
walk ch across chars(w) {{
  when ch = "a" either ch = "e" either ch = "i" either ch = "o" either ch = "u" {{
    c + 1 -> c
  }}
}}
emit c"""
    return prompt, src


def _fam_gcd(args: tuple[int, int]) -> tuple[str, str]:
    a, b = args
    prompt = f"Emit the greatest common divisor of {a} and {b} (a single number)."
    src = f"""\
forge gcd(a, b) {{
  sustain b != 0 {{
    a % b -> r
    b -> a
    r -> b
  }}
  give a
}}
emit gcd({a}, {b})"""
    return prompt, src


def _fam_nth_fib(n: int) -> tuple[str, str]:
    prompt = (
        f"Emit the {n}th Fibonacci number (a single number), where fib(0) = 0, "
        "fib(1) = 1, and fib(k) = fib(k-1) + fib(k-2)."
    )
    src = f"""\
forge fib(n) {{
  0 -> a
  1 -> b
  0 -> i
  sustain i < n {{
    a + b -> t
    b -> a
    t -> b
    i + 1 -> i
  }}
  give a
}}
emit fib({n})"""
    return prompt, src


def _fam_palindrome(word: str) -> tuple[str, str]:
    prompt = (
        f'Emit "yes" if the lowercase word "{word}" is a palindrome (reads the same '
        'forwards and backwards), otherwise emit "no".'
    )
    src = f"""\
"{word}" -> w
chars(w) -> cs
[] -> rev
0 -> i
sustain i < count(cs) {{
  push(rev, cs @ (count(cs) - 1 - i)) -> rev
  i + 1 -> i
}}
when join(rev, "") = w {{ emit yes }} otherwise {{ emit no }}"""
    return prompt, src


# --------------------------------------------------------------------------- #
# Parameter grids per family. Deterministic given the seed.
# --------------------------------------------------------------------------- #

_WORDS = [
    "level",
    "radar",
    "hello",
    "world",
    "banana",
    "racecar",
    "python",
    "noon",
    "stats",
    "kayak",
    "syntax",
    "rotor",
    "dataset",
    "civic",
    "tenet",
    "rill",
    "madam",
    "reward",
    "gradient",
    "policy",
    "rollout",
    "tokenizer",
    "deifit",
    "refer",
    "system",
    "encoder",
    "sagas",
    "minimal",
    "wow",
    "deed",
]


def _params_for(family: str, rng: random.Random) -> tuple[FamilyFn, list[object]]:
    if family == "sum_to_n":
        return _fam_sum_to_n, list(range(5, 65))
    if family == "fizzbuzz":
        return _fam_fizzbuzz, list(range(10, 50))
    if family == "primes_below":
        return _fam_primes_below, list(range(10, 80))
    if family == "list_reverse":
        return _fam_list_reverse, [
            [rng.randint(0, 99) for _ in range(rng.randint(3, 8))] for _ in range(60)
        ]
    if family == "count_predicate":
        return _fam_count_predicate, [
            ([rng.randint(0, 20) for _ in range(rng.randint(4, 9))], rng.randint(3, 15))
            for _ in range(60)
        ]
    if family == "digit_sum":
        return _fam_digit_sum, [rng.randint(0, 999_999) for _ in range(60)]
    if family == "vowel_count":
        return _fam_vowel_count, list(_WORDS)
    if family == "gcd":
        return _fam_gcd, [(rng.randint(2, 600), rng.randint(2, 600)) for _ in range(60)]
    if family == "nth_fib":
        return _fam_nth_fib, list(range(0, 30))
    if family == "palindrome":
        return _fam_palindrome, list(_WORDS)
    raise ValueError(f"unknown family: {family}")


# Disjoint family split. Eval families are never seen during training, so eval
# measures transfer of the learned syntax, not memorized prompts.
TRAIN_FAMILIES = [
    "sum_to_n",
    "fizzbuzz",
    "primes_below",
    "list_reverse",
    "count_predicate",
    "digit_sum",
    "vowel_count",
]
EVAL_FAMILIES = ["gcd", "nth_fib", "palindrome"]


def _build_family(family: str, rng: random.Random) -> list[RillTask]:
    fn, params = _params_for(family, rng)
    tasks: list[RillTask] = []
    for i, p in enumerate(params):
        prompt, src = fn(p)
        res = run_rill(src)
        if not res.ok or not res.output:
            raise AssertionError(
                f"reference solution for {family}[{i}] did not produce output: "
                f"error={res.error!r}\n--- src ---\n{src}"
            )
        tasks.append(
            RillTask(
                name=f"{family}_{i}",
                family=family,
                prompt=prompt,
                expect=res.output,
                metadata={"param": p},
            )
        )
    return tasks


def build_tasks(seed: int = 0) -> tuple[list[RillTask], list[RillTask]]:
    """Build (train_tasks, eval_tasks) with disjoint families.

    Every task's ``expect`` is the verified output of its reference solution, so the
    set contains no unsolvable items.
    """
    rng = random.Random(seed)
    train = [t for fam in TRAIN_FAMILIES for t in _build_family(fam, rng)]
    eval_ = [t for fam in EVAL_FAMILIES for t in _build_family(fam, rng)]
    rng.shuffle(train)
    return train, eval_


if __name__ == "__main__":
    train, eval_ = build_tasks()
    print(f"train tasks: {len(train)} across {len(TRAIN_FAMILIES)} families")
    print(f"eval tasks:  {len(eval_)} across {len(EVAL_FAMILIES)} families")
    for t in train[:2] + eval_[:2]:
        print(f"\n[{t.family}] {t.name}\n  prompt: {t.prompt}\n  expect: {t.expect!r}")
