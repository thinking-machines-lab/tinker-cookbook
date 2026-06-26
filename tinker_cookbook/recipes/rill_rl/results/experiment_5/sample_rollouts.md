# Experiment 5 — example rollouts (Lucas family + 16-sample eval)

Base `Qwen/Qwen3.5-4B` vs after 60 GRPO steps. `nth_fib` was the long-standing weak family; the new `lucas` train family (same two-state recurrence) moved it.


## `nth_fib`

**Prompt:** Define `forge solve(n)` that returns the nth Fibonacci number, where fib(0) = 0 and fib(1) = 1. It is tested on hidden values of n.


### Before / base (fail, reward 0.00)
```rill
forge solve(n) {
  when n = 0 { give 0 }
  when n = 1 { give 1 }
  sustain n > 2 {
    a -> 0
    b -> 1
    a + b -> next
    b -> next
    a -> 0
    b -> 1
    n - 1 -> n
    a + b -> next
    b -> next
    a -> 0
    b -> 1
  }
  give b
}

forge solve2(n) {
  a -> 0
  b -> 1
  when n = 0 { give a }
  when n = 1 { give b }
  sustain n >= 2 {
    a + b -> sum
    b -> sum
    a -> 0
    b -> 1
    n - 1 -> n
  }
  give b
}

forge solve3(n) {
  a -> 0
  b -> 1
  n = n - 1
  while a < 1 and n >= 0 {
    a + b -> sum
    b -> sum
    a -> 0
  }
  give b
}
```


### After Exp 5 (PASS, reward 1.00)
```rill
forge solve(n) {
  when n = 0 { give 0 }
  when n = 1 { give 1 }
  otherwise {
    0 -> a
    1 -> b
    sustain n > 1 {
      a + b -> c
      b -> a
      c -> b
      n - 1 -> n
    }
    give b
  }
}
emit solve(10)
```


## `gcd`

**Prompt:** Define `forge solve(a, b)` that returns the greatest common divisor of non-negative integers a and b. It is tested on hidden pairs.


### Before / base (fail, reward 0.40)
```rill
forge solve(a, b) {
  when b = 0 { give a } 
  a % b -> r
  solve(b, r)
}
emit solve(48, 18)
```


### After Exp 5 (PASS, reward 1.00)
```rill
forge solve(a, b) {
  when b = 0 { give a }
  sustain b != 0 {
    a % b -> temp
    b -> a
    temp -> b
  }
  give a
}

emit solve(12, 18)
```
