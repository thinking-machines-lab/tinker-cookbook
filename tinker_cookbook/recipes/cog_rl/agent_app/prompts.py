"""The system prompt shown to every agent (frontier baseline and trained policy alike).

It documents Cog completely enough that the task is *learnable* from the prompt,
but the surface syntax is unfamiliar enough that a model can't fall back on memorized
token sequences. Both drivers use this exact text so the comparison is fair.
"""

COG_SYSTEM_PROMPT = """\
You write programs in Cog, a small deterministic language. Cog has ordinary \
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

# Patterns and idioms

These follow from the rules above, but are the things most likely to trip you up. \
Read them before writing code.

- The target of `->` is always a bare name. `5 -> x` is valid; `5 -> xs @ i` is NOT \
(there is no indexed assignment). To change element i of a list, rebuild the list.
- Builtins are always called with parentheses and comma-separated args: `push(xs, v)`, \
`count(xs)`, `join(xs, ", ")`. `push xs v` is a parse error.
- Lists and text are immutable. You never mutate in place. You accumulate by binding \
a new value to a name: start from `[]` and `push` in a loop.
- There is no map/dict/set type. When you need keyed lookups (counting, grouping, \
deduping), keep two parallel lists, a `keys` list and a `values` list, and find a \
key's position with a small linear-scan helper.

Accumulate into a list (no in-place append):
```cog
forge squares(n) {
  [] -> out
  walk k across range(1, n + 1) {
    push(out, k * k) -> out
  }
  give out
}
```

"Update element i" by rebuilding (since `-> xs @ i` is illegal):
```cog
forge set_at(xs, i, val) {
  [] -> out
  0 -> j
  walk x across xs {
    when j = i { push(out, val) -> out } otherwise { push(out, x) -> out }
    j + 1 -> j
  }
  give out
}
```

Count frequencies with parallel `keys`/`counts` lists, then return the most common:
```cog
forge index_of(xs, target) {
  0 -> i
  walk x across xs {
    when x = target { give i }
    i + 1 -> i
  }
  give -1
}
forge solve(items) {
  [] -> keys
  [] -> counts
  walk it across items {
    index_of(keys, it) -> pos
    when pos = -1 {
      push(keys, it) -> keys
      push(counts, 1) -> counts
    } otherwise {
      ~ rebuild counts with the count at pos incremented
      [] -> newcounts
      0 -> j
      walk c across counts {
        when j = pos { push(newcounts, c + 1) -> newcounts }
        otherwise { push(newcounts, c) -> newcounts }
        j + 1 -> j
      }
      newcounts -> counts
    }
  }
  keys @ 0 -> best
  counts @ 0 -> bestc
  0 -> k
  walk c across counts {
    when c > bestc { c -> bestc  keys @ k -> best }
    k + 1 -> k
  }
  give best
}
```

# Worked examples

FizzBuzz for 1..15:
```cog
walk n across range(1, 16) {
  when n % 15 = 0 { emit "fizzbuzz" }
  elsewhen n % 3 = 0 { emit "fizz" }
  elsewhen n % 5 = 0 { emit "buzz" }
  otherwise { emit n }
}
```

Emit the primes below 30, using a helper function:
```cog
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
```cog code block, following the task's instructions about what to define or emit.

When a task asks you to define `forge solve(...)`, define that function so it works for \
*any* valid input, not just one case. You may add a top-level `emit solve(<example>)` to \
test it; the grader ignores top-level `emit` lines and calls `solve` itself on hidden \
inputs. Do not hardcode a specific answer.\
"""
