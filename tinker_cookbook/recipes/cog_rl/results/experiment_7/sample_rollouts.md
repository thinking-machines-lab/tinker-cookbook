# Experiment 7 — example rollouts (corpus-trained, held-out MBPP)

After 120 GRPO steps on 561 MBPP-derived tasks (no hand-written Cog), the model solves
diverse *held-out* problems — distinct problems, not family variants. A few of the 36/99
it got exactly right:


## `mbpp_135` — nth hexagonal number
Recognized the closed form `n * (2n - 1)`:
```cog
forge solve(n) {
  2 * n - 1 -> multiplier
  n * multiplier -> result
  give result
}
```


## `mbpp_168` — frequency of a value in an array
```cog
forge solve(a, x) {
  0 -> count
  walk item across a {
    when item = x { count + 1 -> count }
  }
  give count
}
```


## `mbpp_22` — first duplicate element
A nested membership scan, with `skip` to avoid comparing an element to itself:
```cog
forge solve(nums) {
  walk i across range(0, count(nums)) {
    nums @ i -> current
    walk j across range(0, count(nums)) {
      when j = i { skip }
      nums @ j -> other
      when current = other { give current }
    }
  }
  give -1
}
```


## `mbpp_221` — first even number
```cog
forge solve(nums) {
  walk k across nums {
    when k % 2 = 0 { give k }
  }
  give 0
}
```

These are exactly the kinds of shapes (frequency, duplicate detection, conditional search)
that previously required a hand-authored family each. Here they came for free from the
corpus, and the policy generalized to them on held-out problems.
