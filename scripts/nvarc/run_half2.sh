#!/usr/bin/env bash
# Local stand-in for NVARC+ on 1x3090: two half-cost TTT passes, then pool.
#
# Kaggle hidden = 240 tasks / 4xL4 / 12h, so each pass must be ~half the stock
# NVARC work (8 train augs, 8 decode views). Here we run the same recipe on the
# official 120 public eval with no per-task cap, which on one 3090 costs about
# the same wall time as the existing full-cost eval120 (~13.5h for both passes).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}

export PYTHONHASHSEED=0
export ARC_N_TRAIN_AUG=8
export ARC_N_EVAL_AUG=1

echo "=== half-A (stock seeds, 8 aug / 8 views) ==="
export ARC_LORA_SEED=42
export ARC_TRAIN_AUG_SEED=1
export ARC_EVAL_AUG_SEED=2
export ARC_SCORE_SEED_OFFSET=0
bash "$HERE/run_local.sh" eval120_half_a 0 --skip-done

echo "=== half-B (NVARC+ seeds, same half-cost) ==="
export ARC_LORA_SEED=137
export ARC_TRAIN_AUG_SEED=17
export ARC_EVAL_AUG_SEED=29
export ARC_SCORE_SEED_OFFSET=7
bash "$HERE/run_local.sh" eval120_half_b 0 --skip-done

echo "=== pool half-A + half-B (pure kgmon top-2, no keep-primary) ==="
POOL=$WORK/eval120_half_pool
mkdir -p "$POOL"
"$VENV/bin/python" "$HERE/finalize.py" \
  --data "$DATA" --solutions "$SOL" \
  --outputs "$WORK/eval120_half_a/outputs" \
  --outputs-extra "$WORK/eval120_half_b/outputs" \
  --submission "$POOL/submission.json" \
  --report "$POOL/report.json"
touch "$POOL/done"
echo "half2 done"
date
