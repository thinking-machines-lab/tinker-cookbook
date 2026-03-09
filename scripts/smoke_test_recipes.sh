#!/bin/bash
#
# Smoke test for all modified recipe train scripts.
#
# Launches all recipes in parallel. Each recipe runs until step 1 is detected
# in its output (meaning step 0 completed successfully), then gets killed.
# Falls back to a timeout if step 1 is not detected.
#
# Usage:
#   bash scripts/smoke_test_recipes.sh                  # run all recipes in parallel
#   bash scripts/smoke_test_recipes.sh chat_sl          # run a single recipe
#   TIMEOUT=600 bash scripts/smoke_test_recipes.sh      # custom timeout (default: 300s)
#
# Skipped recipes (require extra setup):
#   - prompt_distillation: needs a local JSONL data file
#   - harbor_rl: needs Modal + downloaded Harbor tasks

set -uo pipefail

TIMEOUT="${TIMEOUT:-300}"
LOG_DIR=$(mktemp -d)

# Each entry: "name|module|arg1|arg2|..."
RECIPES=()

add_recipe() {
    local name="$1"
    shift
    # Join remaining args with |
    local entry="$name"
    for arg in "$@"; do
        entry="$entry|$arg"
    done
    RECIPES+=("$entry")
}

# Allow filtering to a single recipe
FILTER="${1:-}"

should_add() {
    [ -z "$FILTER" ] || [ "$FILTER" = "$1" ]
}

# --- Define recipes ---

should_add chat_sl && add_recipe "chat_sl" \
    tinker_cookbook.recipes.chat_sl.train \
    behavior_if_log_dir_exists=delete

should_add dpo && add_recipe "dpo" \
    tinker_cookbook.recipes.preference.dpo.train \
    behavior_if_log_dir_exists=delete

should_add vlm_classifier && add_recipe "vlm_classifier" \
    tinker_cookbook.recipes.vlm_classifier.train \
    experiment_dir=/tmp/tinker-smoke-test/vlm_classifier \
    behavior_if_log_dir_exists=delete

should_add off_policy_reasoning && add_recipe "off_policy_reasoning" \
    tinker_cookbook.recipes.distillation.off_policy_reasoning \
    batch_size=128 \
    behavior_if_log_dir_exists=delete

should_add shorter && add_recipe "shorter" \
    tinker_cookbook.recipes.preference.shorter.train \
    behavior_if_log_dir_exists=delete

should_add on_policy_distillation && add_recipe "on_policy_distillation" \
    tinker_cookbook.recipes.distillation.on_policy_distillation \
    behavior_if_log_dir_exists=delete

should_add on_policy_multi_teacher && add_recipe "on_policy_multi_teacher" \
    tinker_cookbook.recipes.distillation.on_policy_multi_teacher \
    behavior_if_log_dir_exists=delete

should_add rlhf_pipeline && add_recipe "rlhf_pipeline" \
    tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline \
    short_name=smoke-test

should_add guess_number && add_recipe "guess_number" \
    tinker_cookbook.recipes.multiplayer_rl.guess_number.train \
    behavior_if_log_dir_exists=delete

should_add text_arena && add_recipe "text_arena" \
    tinker_cookbook.recipes.multiplayer_rl.text_arena.train \
    behavior_if_log_dir_exists=delete

should_add twenty_questions && add_recipe "twenty_questions" \
    tinker_cookbook.recipes.multiplayer_rl.twenty_questions.train \
    behavior_if_log_dir_exists=delete

if [ ${#RECIPES[@]} -eq 0 ]; then
    echo "No recipes to run (filter: '$FILTER')"
    exit 0
fi

# --- Run a single recipe (called as a subprocess) ---

run_one() {
    local entry="$1"
    IFS='|' read -ra parts <<< "$entry"
    local name="${parts[0]}"
    local args=("${parts[@]:1}")
    local logfile="$LOG_DIR/$name.log"
    local statusfile="$LOG_DIR/$name.status"

    echo "[${name}] Starting..." >&2

    # Launch recipe, capture all output
    uv run python -m "${args[@]}" > "$logfile" 2>&1 &
    local pid=$!

    # Monitor for step 1
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
        if grep -qE "Step 1|Sampling batch 1|batch_idx=1" "$logfile" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            echo "PASSED" > "$statusfile"
            echo "[${name}] PASSED (step 1 reached after ${elapsed}s)" >&2
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$TIMEOUT" ]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            echo "TIMEOUT" > "$statusfile"
            echo "[${name}] FAILED (timeout after ${TIMEOUT}s)" >&2
            return 1
        fi
    done

    # Process exited on its own — check if step 0 appeared at all
    wait "$pid" 2>/dev/null
    local exit_code=$?
    if grep -qE "Step 0|Sampling batch 0|batch_idx=0|step.*=.*0" "$logfile" 2>/dev/null; then
        echo "PASSED" > "$statusfile"
        echo "[${name}] PASSED (completed, exit=$exit_code)" >&2
        return 0
    else
        echo "FAILED" > "$statusfile"
        echo "[${name}] FAILED (exit=$exit_code, no training steps seen)" >&2
        echo "[${name}] Last 20 lines of output:" >&2
        tail -20 "$logfile" >&2
        return 1
    fi
}

# --- Launch all recipes in parallel ---

echo "Launching ${#RECIPES[@]} recipes in parallel (timeout: ${TIMEOUT}s each)"
echo "Logs: $LOG_DIR"
echo ""

PIDS=()
NAMES=()
for entry in "${RECIPES[@]}"; do
    IFS='|' read -ra parts <<< "$entry"
    local_name="${parts[0]}"
    run_one "$entry" &
    PIDS+=($!)
    NAMES+=("$local_name")
done

# Wait for all to finish
PASSED=0
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null || true
    local_name="${NAMES[$i]}"
    statusfile="$LOG_DIR/$local_name.status"
    if [ -f "$statusfile" ] && [ "$(cat "$statusfile")" = "PASSED" ]; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# --- Summary ---

echo ""
echo "=========================================="
echo "RESULTS: $PASSED passed, $FAILED failed out of ${#RECIPES[@]}"
echo "=========================================="
echo "Logs available at: $LOG_DIR"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed recipes:"
    for name in "${NAMES[@]}"; do
        statusfile="$LOG_DIR/$name.status"
        if [ ! -f "$statusfile" ] || [ "$(cat "$statusfile")" != "PASSED" ]; then
            echo "  - $name (see $LOG_DIR/$name.log)"
        fi
    done
    exit 1
fi
