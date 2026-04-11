# Golf Forecasting

This recipe trains an LLM to read a golf leaderboard snapshot and output a calibrated probability distribution over likely winners.

The design follows the same pattern as the other Tinker Cookbook recipes:

- `build_dataset.py` fetches and normalizes public data into versioned JSONL artifacts.
- `data.py` defines the forecasting schema and RL dataset builder.
- `env.py` implements a single-turn forecasting environment with dense proper-scoring rewards.
- `train.py` launches RL training with `tinker_cookbook.rl.train`.
- `eval.py` runs fixed offline evaluation on a held-out manifest.
- `program.md` is a Karpathy-style instruction file for a coding agent running the recipe in a loop.

## Data Format

Each JSONL example stores one leaderboard snapshot with its resolved winner:

```json
{
  "example_id": "masters-2025-r4-h10",
  "tournament_id": "masters-2025",
  "tournament_name": "The Masters",
  "course_name": "Augusta National",
  "round_number": 4,
  "event_day": "Sunday",
  "snapshot_timestamp": "2025-04-13T20:05:00Z",
  "players": [
    {
      "name": "Scottie Scheffler",
      "position": "1",
      "score_to_par": -11,
      "strokes_behind": 0,
      "holes_completed": 10,
      "current_hole": 11,
      "holes_remaining": 8,
      "prior_win_prob": 0.34
    }
  ],
  "target_winner": "Scottie Scheffler",
  "other_field_prior": 0.18,
  "system_context": {
    "weather_summary": "Light breeze, scoring expected to stay low"
  },
  "source_urls": ["https://example.com/leaderboard"]
}
```

The model must return strict JSON:

```json
{
  "winner_probs": {
    "Scottie Scheffler": 0.41,
    "Rory McIlroy": 0.23,
    "other": 0.36
  }
}
```

## No Bundled Data

This recipe intentionally does not ship with a hard-coded golf dataset. The expectation is that your coding agent will discover public sources, fetch raw leaderboard history and priors, and build the dataset itself.

The only bundled artifact is a source-manifest template:

- `tinker_cookbook/example_data/golf_forecasting/source_manifest.template.json`

## Building A Dataset

`build_dataset.py` reads a JSON manifest that points to public data sources and optional prior files. It caches raw responses under `raw/`, normalizes them, and writes:

- `train.jsonl`
- `val.jsonl`
- `heldout.jsonl`
- `dataset_manifest.json`

Typical flow:

```bash
# 1. Copy and edit the source-manifest template after discovering real public sources
cp tinker_cookbook/example_data/golf_forecasting/source_manifest.template.json /tmp/golf_sources.json

# 2. Build the normalized dataset
python -m tinker_cookbook.recipes.golf_forecasting.build_dataset \
  source_manifest_path=/tmp/golf_sources.json \
  output_dir=tinker_cookbook/example_data/golf_forecasting \
  fetch_online=true
```

Important rules:

- Keep the held-out split frozen once `dataset_manifest.json` exists.
- Cache raw sources so experiments are reproducible.
- Record source URLs and timestamps for every fetch.

## Training

Example command:

```bash
python -m tinker_cookbook.recipes.golf_forecasting.train \
  model_name="meta-llama/Llama-3.2-1B-Instruct" \
  dataset_manifest_path=tinker_cookbook/example_data/golf_forecasting/dataset_manifest.json \
  group_size=8 \
  groups_per_batch=32 \
  learning_rate=4e-5 \
  max_tokens=256
```

The RL environment rewards forecasts using a normalized multiclass Brier score. Validation rollouts are run automatically through the standard `RLTestSetEvaluator` path in `tinker_cookbook.rl.train`.

## Offline Evaluation

Run the frozen held-out benchmark with:

```bash
python -m tinker_cookbook.recipes.golf_forecasting.eval \
  model_name="meta-llama/Llama-3.2-1B-Instruct" \
  dataset_manifest_path=tinker_cookbook/example_data/golf_forecasting/dataset_manifest.json
```

This writes:

- `metrics.json`
- `predictions.jsonl`

under `tinker_cookbook/recipes/golf_forecasting/results/<timestamp>/`.

Primary metric:

- `eval/log_loss`

Secondary metrics:

- `eval/brier`
- `eval/top1_accuracy`
- `eval/top3_recall`
- `eval/format_valid_rate`

## Autoresearch

Use `program.md` as the top-level instruction file for Cursor or Claude Code. The agent is allowed to:

- edit the golf forecasting recipe files
- discover public data sources
- gather public web data and priors
- rebuild the training dataset
- choose different models, output formats, and training strategies
- maintain a frozen anchor eval plus a flexible research eval

The agent must preserve one rule:

- once it creates the anchor eval manifest, that anchor benchmark stays frozen even if the research eval evolves

## Overnight tmux Launch

If you want to kick off autoresearch on a remote host and disconnect, you can launch Claude Code inside a detached `tmux` session.

Before using this:

- make sure `claude` already works interactively on the machine
- make sure `TINKER_API_KEY` is set
- make sure Claude Code is configured to avoid blocking on approval prompts

Create a launch script:

```bash
cat > ~/run_golf_autoresearch.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-golf-autoresearch}"
REPO="${REPO:-/work/dylan/Git/tinker-cookbook-dylan}"
LOG_DIR="${LOG_DIR:-$HOME/golf-autoresearch-logs}"
PROMPT_FILE="${PROMPT_FILE:-$LOG_DIR/golf_prompt.txt}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-8}"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed"
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI is not installed or not on PATH"
  exit 1
fi

if [[ ! -d "$REPO" ]]; then
  echo "Repo not found: $REPO"
  exit 1
fi

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set"
  exit 1
fi

cat > "$PROMPT_FILE" <<'PROMPT'
Read `tinker_cookbook/recipes/golf_forecasting/program.md` and follow it exactly.

Start by discovering public golf data sources, building the first dataset, freezing a clean anchor eval as soon as you have a minimally viable benchmark, defining an initial research eval, establishing a baseline on both, and then entering the endless autoresearch loop.

You have broad freedom to change models, prompts, output formats, training methods, dataset structure, and evaluation design, but once the anchor eval is created it must never change.

Additional operating instructions:
- Work autonomously until manually interrupted.
- Use real public golf data; do not hard-code fake golf examples into the recipe.
- Cache fetched raw data and record source URLs and fetch timestamps.
- Maintain both scoreboards:
  - Anchor eval: frozen forever once created
  - Research eval: may evolve freely
- Keep an experiment log with what changed and why.
- If useful, redesign the current recipe structure to better support the research loop.
- Prefer small, testable hypotheses, but do not hesitate to rethink the full system if needed.
PROMPT

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

RUN_LOG="$LOG_DIR/${SESSION_NAME}-$(date +%Y%m%d-%H%M%S).log"

tmux new-session -d -s "$SESSION_NAME" "bash -lc '
  set -euo pipefail
  cd \"$REPO\"
  mkdir -p \"$LOG_DIR\"
  echo \"[$(date -Is)] starting claude in $REPO\" >> \"$RUN_LOG\"
  uv sync --extra dev
  claude
'"

tmux pipe-pane -o -t "$SESSION_NAME:0.0" "cat >> \"$RUN_LOG\""

sleep "$STARTUP_WAIT_SECONDS"

tmux load-buffer "$PROMPT_FILE"
tmux paste-buffer -t "$SESSION_NAME:0.0"
tmux send-keys -t "$SESSION_NAME:0.0" Enter

echo "Started tmux session: $SESSION_NAME"
echo "Log file: $RUN_LOG"
echo
echo "Useful commands:"
echo "  Attach:        tmux attach -t $SESSION_NAME"
echo "  Detach:        Ctrl-b then d"
echo "  Tail log:      tail -f \"$RUN_LOG\""
echo "  Stop session:  tmux kill-session -t $SESSION_NAME"
EOF

chmod +x ~/run_golf_autoresearch.sh
```

Then launch it:

```bash
export TINKER_API_KEY=your_key_here
~/run_golf_autoresearch.sh
```

Useful follow-ups:

```bash
tmux attach -t golf-autoresearch
tail -f ~/golf-autoresearch-logs/golf-autoresearch-*.log
tmux kill-session -t golf-autoresearch
```

Notes:

- if `claude` shows a first-run login or approval prompt, the session may block
- doing one manual dry run first is a good idea
- the script assumes the repo lives at `/work/dylan/Git/tinker-cookbook-dylan`, but you can override `REPO`

