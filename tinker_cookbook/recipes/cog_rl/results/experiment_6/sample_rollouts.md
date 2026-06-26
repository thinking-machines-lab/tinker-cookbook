# Experiment 6 — example rollouts (handbook + keyed-lookup transfer)

Base `Qwen/Qwen3.5-4B` (with the handbook prompt) vs after 80 GRPO steps. The story is
transfer: training on keyed-lookup *train* families taught the parallel-list / seen-list
idiom, and it carried to the *held-out* keyed-lookup families `mode_count` and
`first_repeat`.


## `mode_count` (held-out) — count of the most frequent element

**Prompt:** Define `forge solve(xs)`, where xs is a non-empty list of integers, that returns how many times the most frequent element occurs (the count, not the element).


### Before / base+handbook (intermittent fail, pass@1 0.60)
The handbook teaches the parallel `keys`/`counts` pattern, so the base model reaches for it,
but it often **drops the `index_of` helper it calls**, so `solve` dies with an
undefined-function runtime error on the hidden inputs:
```cog
forge solve(xs) {
  [] -> keys
  [] -> counts
  walk it across xs {
    index_of(keys, it) -> pos    ~ <- index_of is never defined
    when pos = -1 { ... } otherwise { ... }
  }
  ...
}
```

### After / Exp 6 (pass@1 1.00)
Defines the helper and the rebuild loop, on a task it never trained on:
```cog
forge index_of(xs, target) {
  0 -> i
  walk x across xs {
    when x = target { give i }
    i + 1 -> i
  }
  give -1
}
forge solve(xs) {
  [] -> keys
  [] -> counts
  walk it across xs {
    index_of(keys, it) -> pos
    when pos = -1 {
      push(keys, it) -> keys
      push(counts, 1) -> counts
    } otherwise {
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
  counts @ 0 -> maxc
  walk c across counts { when c > maxc { c -> maxc } }
  give maxc
}
```


## `first_repeat` (held-out) — first value that appears twice

After training the model didn't copy a train solution, it derived a different valid
approach: a `slice`-based membership scan over the prefix seen so far.
```cog
forge solve(xs) {
  walk idx across range(count(xs)) {
    xs @ idx -> current
    walk item across slice(xs, 0, idx) {
      when item = current { give current }
    }
  }
  give void
}
```


## `gcd` — Euclidean algorithm (lifted by the `gcd_steps` train family)

`gcd` was 0.40 at base+handbook and **1.00** after training. The `gcd_steps` train family
(same `a % b -> r; b -> a; r -> b` swap loop, different return) put that exact structure in
front of the policy:
```cog
forge solve(a, b) {
  sustain b > 0 {
    a % b -> r
    a -> b
    r -> a
  }
  give a
}
```
