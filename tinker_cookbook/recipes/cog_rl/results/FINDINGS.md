# Findings: scaling, distillation, and where the gains actually come from

The logbook (experiments 1-8) is the tutorial arc: build the proxy-trained recipe, catch a
reward hack, scale the task source, and bootstrap training data with no hand-written Cog.
After that we kept pushing on one question: how far can a small open model get in this app,
and which post-training methods actually move it? This file consolidates that phase
(originally experiments 9-11) into the results and the transferable lessons.

All numbers are held-out corpus pass@1 at 5 samples per task (n=495, SE about ±0.022)
unless marked otherwise. Student models are LoRA rank 32.

## Where each rung came from

| model / stage | pass@1 | note |
|---|---|---|
| 4B base + handbook prompt | 0.283 | |
| 4B best (self-distilled SFT, then GRPO; 3 expert-iteration rounds) | 0.513 | converges ~0.51 regardless of further rounds |
| 9B base + handbook prompt | 0.347 | |
| 9B SFT (same 408 self-distilled solutions) | 0.519 | matches the 4B ceiling before any 9B RL |
| 9B SFT + GRPO | 0.562 | RL has headroom again at 9B (+0.043; it was flat at 4B) |
| 9B on-policy distillation from a trained 397B teacher | 0.598-0.600 | saturates when the forward KL flattens (~0.11) |
| **9B OPD, then GRPO at 8e-5** | **0.618** | best; RL adds the same +0.02-0.04 on top of imitation |
| Kimi K2.6, untrained (reference) | 0.616 | |
| Qwen3.5-397B, untrained (reference) | 0.620 | |
| Qwen3.5-397B trained with this recipe (the teacher) | 0.673 | |
| gpt-5.5 (frontier reference) | 0.751 | |

The one-line version: a trained 9B lands level with untrained 400B-class open models on
this task, at a fraction of the serving cost.

Test-time technique stacks on top of any of these. Showing one example I/O in the prompt
and retrying until the program passes it (best-of-4, self-verified against the visible
example only, graded strictly on the remaining hidden tests) lifts the trained 9B from
0.626 to **0.768** on that protocol. It lifts gpt-5.5 equally (0.768 to 0.889), so it's a
product improvement rather than a gap-closer, but it means the deployed 9B agent scores
what plain gpt-5.5 scores. These are now `--show-example` / `--best-of` flags on
[`training/eval.py`](../training/eval.py).

## Lessons

1. **Scale the student before tuning the method.** Every technique we tried at 4B
   converged to ~0.51. The same recipe on the 9B started higher and kept responding.
   The failure analysis backs this up: on tasks the 9B misses but the bigger model solves,
   the 9B writes valid, running Cog with the wrong reasoning (zero parse errors anywhere),
   commits to a wrong answer in ~1 of its 3 turns, and never recovers. That is a capacity
   signature, not a syntax or data problem.

2. **Per-method learning rates differ by an order of magnitude.**
   `hyperparam_utils.get_lr` (~4.7e-4 for these models) is right for SFT and for on-policy
   distillation, but GRPO collapsed outright at that LR (on-policy pass@1 to 0.000 by batch
   30) and degraded at 1.5e-4; it wants ~8e-5 here. Conversely 4e-5 silently under-trains
   OPD, because OPD's per-token advantages (teacher logprob minus student logprob) average
   only ~0.1-0.2, several times smaller than GRPO's group-centered advantages. Normalizing
   the deltas by their per-batch std decouples step size from how close the student already
   is, and made OPD converge further and faster.

3. **Imitation first, then verifier RL, and neither repeats.** On-policy distillation
   covers most of the distance quickly, then saturates when the student's forward KL to the
   teacher flattens; more OPD batches or a better init don't move held-out accuracy. One
   round of GRPO on top adds a consistent +0.02-0.04 (it optimizes correctness directly,
   which imitation can't), and further RL is flat. The chain order matters because RL from
   a model that already writes valid Cog spends its steps on correctness rather than syntax
   discovery.

4. **Off-policy imitation of a stronger teacher's text hurts.** SFT on the teacher's
   solutions scored below SFT on the student's own verified solutions, and SFT on the
   teacher's full reasoning traces was worse still (0.414). The style mismatch costs more
   than the content is worth; the working alternatives are rejection-sampled *self*-
   distillation ([`training/curate.py`](../training/curate.py) + [`training/sft.py`](../training/sft.py))
   and *on-policy* distillation, where the teacher only scores the student's own rollouts
   (see `tinker_cookbook/distillation/`).

5. **The absolute ceiling is partly the tasks.** About a third of the held-out problems
   are missed by every model including the frontier, mostly ambiguous prompt semantics
   ("number of solutions", "parity") and problems that don't map to Cog's type system.
   Prompt disambiguation and feasibility filtering raise everyone's score; they don't
   change the model comparison.

## Reproducing this phase

- Teacher: run [`training/train.py`](../training/train.py) with
  `model_name=Qwen/Qwen3.5-397B-A17B` (GRPO from base; the big model already writes valid
  Cog, so it needs no SFT warm-start).
- Student SFT: harvest verified solutions with `curate.py`, train with `sft.py`.
- On-policy distillation: `tinker_cookbook/distillation/` has the library implementation
  (teacher `compute_logprobs` on the student's own samples). Student and teacher must share
  a tokenizer, which holds within the Qwen3.5 family.
- Chaining: every stage saves a trainable state; pass it as `init_state_path=` to the next
  stage (`train.py` supports warm-starting GRPO from any prior state).
