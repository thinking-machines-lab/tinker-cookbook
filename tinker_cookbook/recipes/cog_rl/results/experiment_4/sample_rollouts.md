# Experiment 4 — example rollouts (enriched train mix)

Base `Qwen/Qwen3.5-4B` vs after 60 GRPO steps with the enriched train mix. The first two families were lifted by structure-matched training data; `nth_fib` stayed hard.


## `list_max`

**Prompt:** Define `forge solve(xs)`, where xs is a non-empty list of integers, that returns the largest element. It is tested on hidden lists.


### Before / base (fail, reward 0.15)
```cog
forge solve(xs) {
          head xs -> max
          walk n across xs {
            when n > max {
              n -> max
            }
          }
          give max
        }
```


### After Exp 4 (PASS, reward 1.00)
```cog
forge solve(xs) {
  xs @ 0 -> max_val
  walk elem across xs {
    when elem > max_val {
      elem -> max_val
    }
  }
  give max_val
}

emit solve([3, 7, 2, 9, 1])
```


## `palindrome`

**Prompt:** Define `forge solve(w)`, where w is lowercase text, that returns yes if w is a palindrome (reads the same forwards and backwards), otherwise no. Tested on hidden words.


### Before / base (PASS, reward 1.00)
```cog
forge solve(w) {
  count(w) -> len
  0 -> i
  sustain i < (len / 2) {
    w @ i -> char1
    w @ (len - 1 - i) -> char2
    when char1 = char2 {
      (i + 1) -> i
    }
    otherwise {
      give no
      halt
    }
  }
  give yes
}

emit solve("radar")
```


### After Exp 4 (PASS, reward 1.00)
```cog
forge solve(w) {
  when count(w) <= 1 { give yes }
  0 -> left
  count(w) - 1 -> right
  sustain left < right {
    when w @ left != w @ right { give no }
    left + 1 -> left
    right - 1 -> right
  }
  give yes
}
emit solve("hello")
```


## `nth_fib`

**Prompt:** Define `forge solve(n)` that returns the nth Fibonacci number, where fib(0) = 0 and fib(1) = 1. It is tested on hidden values of n.


### Before / base (fail, reward 0.00)
```cog
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


### After Exp 4 (PASS, reward 1.00)
```cog
forge solve(n) {
  when n = 0 { give 0 }
  when n = 1 { give 1 }
  0 -> a
  1 -> b
  2 -> k
  sustain k <= n {
    a + b -> c
    b -> a
    c -> b
    k + 1 -> k
  }
  give b
}

emit solve(10)
```
