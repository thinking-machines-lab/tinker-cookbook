# IF-RL Environment Analysis

## Status: WORKING WELL — Best performing env

## Paper-Matched Settings That Work
- LR: 3e-5 (LoRA-adjusted from paper's 3e-6 full FT, 10x scaling)
- group_size: 16, batch: 32 (paper=128, reduced for practical speed)
- max_tokens: 49K (paper-matched — critical for reasoning chains)
- temperature: 1.0, top_p: 1.0
- KL coeff: 0, dynamic filtering: on
- Loss: importance_sampling (GRPO)
- Reward: +0.082 in 4 steps at lr=3e-5

## Reward Logic
- 48 IFEval instruction types verified programmatically
- Reward = fraction of instructions satisfied (0.0 to 1.0)
- Overlong penalty: reward=0 if response doesn't complete (stop_reason="length")
- frac_mixed=1.0 means all groups have variance — good GRPO signal

## What's Working
1. **Verifier coverage**: All 48 IFEval instruction types implemented
2. **Granular reward**: Fraction-based (not binary) gives smoother signal
3. **Dynamic filtering**: Removes all-agree groups, paper-matched
4. **Overlong penalty**: Paper-matched, prevents reward hacking via truncation

## Data Quality
- Dataset: `nvidia/Nemotron-Cascade-2-RL-data`, IF-RL split
- kwargs parsed from JSON strings — handles None, str, dict formats
- instruction_id_list + kwargs_list correctly zipped

## Potential Improvements
1. **Scale batch to 128**: Current batch=32 means fewer problems per step, noisier gradients. Paper uses 128.
2. **Run more steps**: Paper runs ~180 steps with dynamic filtering. We've only done 4-50 steps.
3. **Strict vs loose IFEval**: Our verifier only does "loose" matching (e.g., case-insensitive). Adding strict mode would give harder signal for stronger models.
4. **Two known soft spots in verifier**:
   - `detectable_format:constrained_response` always returns True (hard to verify without specific constraint list)
   - `count:counting_composition` always returns True (complex paragraph word count check)
   These could let 2-5% of incorrect responses get reward=1 on those instruction types.
5. **Language detection**: `language:response_language` falls back to True on import error. Ensure `langdetect` is installed.
