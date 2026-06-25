# Results from one real run

Artifacts from a single `Qwen/Qwen3.5-4B` GRPO run (30 steps, `group_size=8`,
`groups_per_batch=8`, `lr=4e-5`, `lora_rank=32`, `max_turns=2`), trained through the
production app via the sampling proxy and evaluated on the held-out families.

- `training_metrics.jsonl` — per-batch `reward/mean`, `reward/pass@1`, datum counts. The
  training metric climbs from pass@1 ≈ 0.41 to ≈ 0.98.
- `trained_final_programs.jsonl` — the trained model's final program for each of the 120
  held-out tasks, with correctness and reward. This is the data behind the headline
  finding: 98% are hardcoded constant emits.
- `sample_rollouts.md` — curated before/after transcripts (one per family) plus the
  reward-hacking writeup.

**Read `sample_rollouts.md` first.** Held-out pass@1 went 0.10 → 0.97, but the policy
reward-hacked: it emits the literal answer rather than writing a general RILL program,
because each task has fixed inputs in the prompt and the reward only checks output. The
recipe README's results caveat explains the fix (grade against hidden inputs).
