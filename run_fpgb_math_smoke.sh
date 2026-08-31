#!/bin/bash
set -o pipefail

mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="fpgb_qwen3_8b_math_lora8_beta0.1_tok2048_smoke_${TIMESTAMP}"
LOG="logs/${RUN_NAME}.log"

echo "Run name:   $RUN_NAME"
echo "Logging to: $LOG"
echo "FPGB smoke test: 2 training steps"

python -m tinker_cookbook.recipes.fpgb.train_math \
  model_name=Qwen/Qwen3-8B \
  env=math \
  lora_rank=8 \
  fpgb_beta=0.1 \
  group_size=4 \
  groups_per_batch=8 \
  learning_rate=1e-5 \
  num_substeps=1 \
  max_tokens=2048 \
  max_steps=5 \
  remove_constant_reward_groups=True \
  eval_every=999999 \
  save_every=999999 \
  wandb_project=tinker-fpgb-math-rl \
  wandb_name="$RUN_NAME" \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -eq 0 ]; then
  echo "FPGB smoke test finished successfully."
else
  echo "FPGB smoke test FAILED with exit code $STATUS."
fi

echo "Log saved to: $LOG"
exit "$STATUS"
