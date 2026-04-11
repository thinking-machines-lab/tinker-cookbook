# golf_forecasting autoresearch

This recipe is designed for autonomous experimentation by a coding agent such as Cursor or Claude Code.

## Mission

Build a more accurate golf forecasting system.

Almost everything is flexible:

- model choice
- training approach
- prompt design
- output format
- reward function
- data pipeline
- feature engineering
- retrieval or tool use
- priors
- train, validation, and held-out definitions
- benchmark design
- recipe structure
- overall system architecture

The core objective is fixed: improve golf forecasting.

## In-Scope Files

Read these files for full context before starting:

- `tinker_cookbook/recipes/golf_forecasting/README.md`
- `tinker_cookbook/recipes/golf_forecasting/data.py`
- `tinker_cookbook/recipes/golf_forecasting/build_dataset.py`
- `tinker_cookbook/recipes/golf_forecasting/env.py`
- `tinker_cookbook/recipes/golf_forecasting/train.py`
- `tinker_cookbook/recipes/golf_forecasting/eval.py`

## Edit Scope

You may edit anything inside `tinker_cookbook/recipes/golf_forecasting/`.

You may also edit nearby shared cookbook infrastructure if it is genuinely necessary to support a better golf forecasting system, but prefer keeping changes local to this recipe.

## Research Freedom

You are explicitly allowed to:

- search the web
- call public HTTP APIs
- fetch public webpages
- discover new public data sources
- redesign the dataset schema
- change the forecasting output format
- switch models
- replace RL with SFT, DPO, distillation, prompt optimization, or hybrids
- add retrieval, tools, or external priors
- redefine the research benchmark

You must:

- use real public data rather than hard-coded fake golf examples
- cache fetched raw data under recipe-local artifacts
- record source URLs and fetch timestamps
- prefer reproducible public sources over brittle ad hoc scraping
- log what changed and why
- log whether each change affected data, model, prompt, training, evaluation, or overall system design

## Two Evaluation Tracks

You must maintain two evaluation tracks.

### 1. Anchor Eval

Create one frozen anchor benchmark as soon as you have a minimally viable dataset.

Rules:

- once created, the anchor eval manifest must never change
- use it as the stable apples-to-apples benchmark for long-term progress
- always record anchor metrics for every experiment

### 2. Research Eval

You may redesign the research eval at any time.

Use it to explore:

- richer data
- new task formulations
- alternative output formats
- better priors
- better metrics
- larger or more realistic datasets

You may replace or expand the research eval whenever doing so helps build a stronger forecaster.

## Setup

1. Create a fresh branch for the run, for example `autoresearch/golf-overnight`.
2. Discover public golf data sources and write a source manifest at, for example, `/tmp/golf_sources.json`.
3. Build an initial dataset:

```bash
python -m tinker_cookbook.recipes.golf_forecasting.build_dataset \
  source_manifest_path=/tmp/golf_sources.json \
  output_dir=tinker_cookbook/example_data/golf_forecasting \
  fetch_online=true
```

1. Freeze a clean anchor eval manifest as soon as you have a minimally viable dataset.
2. Define an initial research eval setup.
3. Create an untracked `results.tsv` with header:

```tsv
commit	anchor_log_loss	anchor_brier	research_score	status	change_type	description
```

1. Establish a baseline on both anchor eval and research eval before changing anything.

## Baseline

Start with a short, bounded baseline run using your current best judgment for:

- model
- dataset
- training approach
- output format
- evaluation setup

If the current recipe implementation is not the right shape, change it.

## Experiment Loop

LOOP FOREVER:

1. Inspect the current best system and experiment history.
2. Form one concrete hypothesis.
3. Implement the change.
4. If data logic changed, rebuild or enrich the dataset.
5. If benchmark logic changed, update only the research eval, never the frozen anchor eval.
6. Train, fine-tune, prompt-optimize, distill, or otherwise update the system.
7. Run both anchor eval and research eval.
8. Record the metrics and what changed.
9. Keep the change if it improves the overall system by your judgment.
10. Always preserve the anchor metric history, even if the research eval changes.
11. Revert changes that are not worth keeping.
12. Continue immediately with the next hypothesis.

## Good Hypotheses

Prefer hypotheses with a clear reason they might help:

- better public data sources
- more complete leaderboard history
- stronger player priors
- improved normalization or entity matching
- better prompt structure
- more robust output parsing
- alternative output formats
- improved reward shaping
- better model selection
- a different training objective
- retrieval or tool-augmented forecasting
- ensemble strategies

## Acceptance Rule

Use judgment, but always record both scoreboards:

- anchor eval: stable, frozen, comparable over time
- research eval: flexible, evolving, exploratory

If a change improves research performance but hurts the anchor benchmark, think carefully before keeping it. If a change meaningfully improves the anchor benchmark, it is usually worth strong consideration even if the research eval has changed.

## Failure Handling

- If a source is flaky, replace it.
- If parsing is brittle, tighten the format or redesign the output.
- If a training run crashes, fix it quickly and continue.
- If a benchmark redesign helps research, keep it in the research eval only.
- If you need a cleaner anchor eval, create it once and then freeze it permanently.

## Never Stop

Do not stop to ask whether you should continue once the loop has started. Keep researching, gathering better public data, redesigning the system, and iterating until you are manually interrupted.