#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"

SESSION_NAME="${SESSION_NAME:-golf-autoresearch}"
REPO="${REPO:-$DEFAULT_REPO}"
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
