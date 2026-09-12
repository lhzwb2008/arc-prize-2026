#!/usr/bin/env bash
# Three 1/3-cost TTT passes on the official 120 eval set, then pure kgmon pool.
#
# Full NVARC (Kaggle 12h / 240 tasks): 16 train augs × 16 decode views.
# Each pass here: 5 train augs × 5 decode views, so 3 passes ≈ 15/16 of one full run.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}

export PYTHONHASHSEED=0
export ARC_N_TRAIN_AUG=5
export ARC_N_EVAL_AUG=1
export ARC_N_EVAL_GEOS=5

echo "=== third-A (5 train augs / 5 decode views) ==="
export ARC_LORA_SEED=42
export ARC_TRAIN_AUG_SEED=1
export ARC_EVAL_AUG_SEED=2
export ARC_SCORE_SEED_OFFSET=0
bash "$HERE/run_local.sh" eval120_t3_a 0 --skip-done

echo "=== third-B ==="
export ARC_LORA_SEED=137
export ARC_TRAIN_AUG_SEED=17
export ARC_EVAL_AUG_SEED=29
export ARC_SCORE_SEED_OFFSET=7
bash "$HERE/run_local.sh" eval120_t3_b 0 --skip-done

echo "=== third-C ==="
export ARC_LORA_SEED=271
export ARC_TRAIN_AUG_SEED=31
export ARC_EVAL_AUG_SEED=41
export ARC_SCORE_SEED_OFFSET=13
bash "$HERE/run_local.sh" eval120_t3_c 0 --skip-done

echo "=== pool A+B+C (pure kgmon top-2) ==="
POOL=$WORK/eval120_t3_pool
mkdir -p "$POOL"
"$VENV/bin/python" "$HERE/finalize.py" \
  --data "$DATA" --solutions "$SOL" \
  --outputs "$WORK/eval120_t3_a/outputs" \
  --outputs-extra "$WORK/eval120_t3_b/outputs" \
  --outputs-extra "$WORK/eval120_t3_c/outputs" \
  --submission "$POOL/submission.json" \
  --report "$POOL/report.json"
touch "$POOL/done"
echo "third3 done"
date
