# Experiment 2 — example rollouts (hidden-input grading)

Base `Qwen/Qwen3.5-4B` vs the same model after 30 GRPO steps under the fixed reward (grade `solve(...)` on hidden inputs). Unlike Experiment 1, after training the model writes real Cog functions, not constant emits.


## `factorial`

**Prompt:** Define `forge solve(n)` that returns n factorial (the product 1*2*...*n, with solve(0) = 1). It is tested on hidden values of n.


### Before training (PASS, reward 1.00)
```cog
forge solve(n) {
  when n <= 1 { give 1 }
  1 -> x
  sustain n >= 2 {
    x * n -> x
    n - 1 -> n
  }
  give x
}
```


### After 30 GRPO steps (PASS, reward 1.00)
```cog
forge solve(n) {
  1 -> result
  2 -> d
  sustain d <= n {
    result * d -> result
    d + 1 -> d
  }
  give result
}
```


## `gcd`

**Prompt:** Define `forge solve(a, b)` that returns the greatest common divisor of non-negative integers a and b. It is tested on hidden pairs.


### Before training (PASS, reward 1.00)
```cog
forge solve(a, b) {
  when b = 0 { give a }
  sustain b > 0 {
    a % b -> r
    b -> a
    r -> b
  }
  give a
}
emit solve(48, 18)
```


### After 30 GRPO steps (fail, reward 0.15)
```cog
forge solve(a, b) {
  sustain b > 0 {
    a % b -> r
    a -> b
    b -> r
  }
  give a
}
```


## `palindrome`

**Prompt:** Define `forge solve(w)`, where w is lowercase text, that returns yes if w is a palindrome (reads the same forwards and backwards), otherwise no. Tested on hidden words.


### Before training (fail, reward 0.15)
```cog
forge solve(w) {
  when w = "" { give yes }
  
  c = chars(w)
  n = count(c)
  i = 0
  sustain i < n / 2 {
    when c @ i != c @ (n - 1 - i) {
      give no
    }
    i + 1 -> i
  }
  give yes
}
```


### After 30 GRPO steps (PASS, reward 1.00)
```cog
forge solve(w) {
  when count(w) = 0 { give yes }
  0 -> i
  sustain i < count(w) / 2 {
    when w @ i != w @ (count(w) - 1 - i) { give no }
    i + 1 -> i
  }
  give yes
}
emit solve("abcba")
```


## `reverse_text`

**Prompt:** Define `forge solve(w)`, where w is lowercase text, that returns w reversed (as text). It is tested on hidden words.


### Before training (fail, reward 0.00)
```cog
forge solve(w) {
  chars(w) -> w_chars
  count(w_chars) -> w_len
  empty_list -> r_list
  walk i across range(0, w_len) {
    push r_list, w_chars @ (w_len - 1 - i) -> r_list
  }
  give join r_list
}
```


### After 30 GRPO steps (fail, reward 0.15)
```cog
forge solve(w) {
  0 -> s
  0 -> i
  sustain i < count(w) {
    w @ (count(w) - 1 - i) -> c
    push(s, c) -> s
    i + 1 -> i
  }
  give s
}
emit solve("hello")
```
