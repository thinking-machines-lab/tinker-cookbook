# Experiment 7 — scalable corpus task source (MBPP-derived)

**Date:** 2026-06-26
**Model:** `Qwen/Qwen3.5-4B` (LoRA rank 32)
**Tinker session:** `2b817cd3-f4db-5d7c-8c8e-d50c68b85b89` (project `8a4e2d8a-…`, label `exp7`)
**Checkpoint:** `tinker://2b817cd3-f4db-5d7c-8c8e-d50c68b85b89:train:0/sampler_weights/final`

## Why this experiment

Experiments 1–6 worked, but the way we added coverage didn't scale: every new shape
(`most_frequent`, then `top_k`, then …) needed a hand-written Cog reference. After Exp 6 a
user asked the trained model for `top_k` and it produced a frequency counter with no
selection-without-replacement step — a shape we'd never put in training. Authoring a family
per shape is O(shapes) of human effort and never converges.

The fix is to **decouple the verifier from the DSL**. The grader is already
language-agnostic on the expected side: it runs the model's Cog `solve` on hidden inputs and
compares the printed output to a stored string. We were producing that string by writing the
reference *in Cog*. Nothing requires that. Instead:

> Grade Cog against I/O pairs computed from an existing corpus's own **Python** reference and
> tests. Any problem that ships a reference + tests becomes a Cog task with zero hand-written
> Cog. Shape coverage then comes from corpus diversity, not from us enumerating families.

## What was built

- `cog_format.py` — the one fixed adapter (written once, not per task): `cog_repr` formats a
  Python value exactly as Cog's `emit` prints it (verified byte-identical to the
  interpreter's `_to_str`), plus escaped source literals and the int/text/flag/list type
  filter.
- `corpus_tasks.py` — parses MBPP `assert f(args) == expected` tests into `(args, expected)`
  pairs, runs the Python reference to validate (dropping broken refs), keeps only
  Cog-representable and discriminative tasks (≥2 distinct outputs, same anti-constant
  guard as the hidden-input families), and emits a `CogTask`.
- `tasks.get_tasks(source)` dispatches `families | corpus | both`; `train.py` gained a
  `task_source` config and `eval.py` a `--task-source` flag.

**Yield: 561 train + 99 held-out tasks from MBPP, zero hand-written Cog.** Train/eval are
split by problem, so eval problems are entirely unseen — and they are 99 *distinct* problems,
not family variants, so this measures generalization across arbitrary shapes.

Config: handbook prompt (shared), `group_size=8`, `groups_per_batch=16`, 120 batches,
`learning_rate=4e-5`, `max_turns=3`, `max_tokens=2048`. On-policy.

## Result (held-out, 99 distinct MBPP problems, 1 sample each)

| | pass@1 | mean reward |
|---|---|---|
| base + handbook (before) | 0.283 | 0.467 |
| **Exp 7 (after)** | **0.364** | **0.665** |

The pass@1 lift is real but smaller than the hand-authored families (Exp 6 was 0.60 → 0.93),
and the absolute numbers are lower — both expected. This is the *real* distribution: harder,
far more diverse, and a meaningful fraction of MBPP problems aren't expressible in Cog at all
(no regex, no dicts/floats internally even when I/O types pass the filter), which caps the
achievable ceiling. The **+0.20 mean-reward** jump shows the model moving much closer on
partial credit broadly, not just flipping a few tasks.

The trained model solves diverse held-out shapes it never saw a sibling of: nth hexagonal
number (closed form), frequency-of-a-value, first-duplicate (nested scan), first-even, etc.
See `trained_final_programs.jsonl` and `sample_rollouts.md`.

## Verdict

- **The scalable approach works.** Training on 561 auto-generated, diverse tasks lifts
  held-out performance on 99 unseen problems with zero hand-authored Cog. Adding `top_k` is
  no longer a code change — it's whatever the corpus already contains, and the corpus can
  grow (HumanEval, APPS, model-proposed-and-validated tasks) without touching the recipe.
- **Honest ceiling.** Per-problem pass@1 on this set is capped by Cog-infeasible problems; a
  cleaner number would pre-filter to Cog-expressible problems, or grade by metamorphic
  properties where exact I/O is the wrong contract.
- The hand-authored families remain the *clean, guaranteed-solvable* benchmark (Exp 6); the
  corpus is the *scalable, realistic* one. `task_source=both` trains on the union.

## Takeaways → next

- Pre-filter the corpus to Cog-expressible problems (drop regex/dict-internal ones) to raise
  the achievable ceiling and de-noise the pass@1.
- Run `task_source=both` so the guaranteed-solvable families anchor the easy floor while the
  corpus supplies open-ended shape coverage.
- Longer training / larger token budget: train pass@1 plateaued near 0.45 (mean reward ~0.68),
  so there is headroom on the feasible subset.
