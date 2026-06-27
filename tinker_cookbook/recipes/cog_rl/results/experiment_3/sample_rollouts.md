# Experiment 3 — example rollouts (async loop, bigger held-out)

Base `Qwen/Qwen3.5-4B` vs after 50 GRPO steps, on the expanded 10-family held-out set. All trained completions are real `solve(...)` functions.


## `reverse_text`

**Prompt:** Define `forge solve(w)`, where w is lowercase text, that returns w reversed (as text). It is tested on hidden words.


### Before (fail, reward 0.15)
```cog
forge solve(w) {
  chars w -> cs
  count(cs) -> n
  walk i across range(n, 0) {
    cs @ (n - 1 - i) -> c
    c -> cs
  }
  join(cs, "") -> result
  give result
}
```


### After 50 steps (PASS, reward 1.00)
```cog
forge solve(w) {
  chars(w) -> chars
  "" -> result
  0 -> i
  walk i across range(0, count(chars)) {
    count(chars) - 1 - i -> idx
    chars @ idx -> c
    result + c -> result
  }
  give result
}
emit solve("hello")
```


## `gcd`

**Prompt:** Define `forge solve(a, b)` that returns the greatest common divisor of non-negative integers a and b. It is tested on hidden pairs.


### Before (fail, reward 0.40)
```cog
forge solve(a, b) {
  when b = 0 { give a } 
  a % b -> r
  solve(b, r)
}
emit solve(48, 18)
```


### After 50 steps (PASS, reward 1.00)
```cog
forge solve(a, b) {
  sustain b > 0 {
    a % b -> temp
    b -> a
    temp -> b
  }
  give a
}
emit solve(48, 18)
```


## `is_sorted`

**Prompt:** Define `forge solve(xs)`, where xs is a list of integers, that returns yes if the list is sorted in non-decreasing order, otherwise no. Tested on hidden lists.


### Before (fail, reward 0.70)
```cog
forge solve(xs) {
  when count(xs) <= 1 { give yes }
  walk i across range(0, count(xs) - 1) {
    when xs @ i > xs @ i+1 { give no }
  }
  give yes
}
emit solve(range(5, 2))
emit solve([1, 2, 3])
emit solve([3, 2, 1])
emit solve([])
emit solve([1])
emit solve([1, 1, 1])
emit solve([1, 2, 1, 2, 3])
```


### After 50 steps (PASS, reward 1.00)
```cog
forge solve(xs) {
  when count(xs) < 2 { give yes }
  0 -> i
  sustain i + 1 < count(xs) {
    when xs @ i > xs @ (i + 1) { give no }
    i + 1 -> i
  }
  give yes
}
emit solve([1, 2, 3, 4, 5])
```


## `lcm`

**Prompt:** Define `forge solve(a, b)` that returns the least common multiple of positive integers a and b. It is tested on hidden pairs.


### Before (fail, reward 0.15)
```cog
forge gcd(a, b) {
  sustain b = 0 {
    a % b -> temp
    b -> a
    temp -> b
  }
  give a
}
```


### After 50 steps (fail, reward 0.15)
```cog
forge gcd(a, b) {
  when b = 0 { give a }
  a % b -> r
  b -> a
  r -> b
}

forge solve(a, b) {
  a * b -> prod
  gcd(a, b) -> g
  prod / g -> result
  give result
}

emit solve(3, 5)
```
