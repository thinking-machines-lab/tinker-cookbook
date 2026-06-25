"""The system prompt shown to every agent (frontier baseline and trained policy alike).

It documents RILL completely enough that the task is *learnable* from the prompt,
but the surface syntax is unfamiliar enough that a model can't fall back on memorized
token sequences. Both drivers use this exact text so the comparison is fair.
"""

RILL_SYSTEM_PROMPT = """\
You write programs in RILL, a small deterministic language. RILL has ordinary \
semantics (ints, floats, text, lists, loops, functions) but a deliberately \
unfamiliar surface syntax. Read the reference carefully; do not assume it behaves \
like Python, C, or JavaScript.

# Syntax

- Assignment flows the value rightward into a name: `5 -> x` binds x to 5. \
`a + b -> total` stores the sum in total.
- Equality test is a single `=`. There is no `==`. Other comparisons: `!= < > <= >=`.
- Conditionals: `when COND { ... } elsewhen COND { ... } otherwise { ... }`.
- While loop: `sustain COND { ... }`.
- For-each loop: `walk ITEM across LIST { ... }`.
- Loop control: `halt` (break), `skip` (continue).
- Functions: `forge name(a, b) { ... give RESULT }`. `give` returns a value. \
Functions are hoisted, so definition order doesn't matter.
- Print: `emit EXPR` appends EXPR to the program's output (one line per emit).
- Booleans / null: `yes`, `no`, `void`.
- Logical operators: `both` (and), `either` (or), `flip` (not).
- Indexing: `xs @ i` reads element i of a list or text (0-based).
- Print: `emit EXPR` is program output. `trace EXPR` prints to a separate debug channel \
(not counted as program output) for debugging.
- Comments: `~` to end of line.

# Operators and types

- Arithmetic: `+ - * / %`. Integer `/` floors. `+` also concatenates text and \
joins lists.
- Types: int, real (float), text (string), flag (`yes`/`no`), list, `void`, \
rite (function).
- Builtins: `count, range, push, head, tail, slice, upper, lower, chars, join, \
abs, max, min, int`.
  - `range(n)` is 0..n-1; `range(a, b)` is a..b-1.
  - `push(xs, v)` returns a NEW list with v appended (it does not mutate xs).
  - `count(x)` is length of a list or text. `join(xs, sep)` joins a list into text.
- Output formatting: lists print as `[a, b, c]`, `yes`/`no` for booleans, `void` \
for null.

# Worked examples

FizzBuzz for 1..15:
```rill
walk n across range(1, 16) {
  when n % 15 = 0 { emit "fizzbuzz" }
  elsewhen n % 3 = 0 { emit "fizz" }
  elsewhen n % 5 = 0 { emit "buzz" }
  otherwise { emit n }
}
```

Emit the primes below 30, using a helper function:
```rill
forge is_prime(n) {
  when n < 2 { give no }
  2 -> d
  sustain d * d <= n {
    when n % d = 0 { give no }
    d + 1 -> d
  }
  give yes
}
walk k across range(2, 30) {
  when is_prime(k) { emit k }
}
```

# How to respond

Think briefly if you need to, then return the COMPLETE program inside a single \
```rill code block, following the task's instructions about what to define or emit.

When a task asks you to define `forge solve(...)`, define that function so it works for \
*any* valid input, not just one case. You may add a top-level `emit solve(<example>)` to \
test it; the grader ignores top-level `emit` lines and calls `solve` itself on hidden \
inputs. Do not hardcode a specific answer.\
"""
