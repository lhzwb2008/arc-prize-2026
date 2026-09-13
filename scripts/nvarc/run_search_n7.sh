#!/usr/bin/env bash
# After n=5×8 finishes, run n_train=7 / geos=8 (same seeds as half-A / n6).
# Two-pass pooling is left to Kaggle; this box keeps searching single-pass n_train.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}

log() { echo "[$(date '+%F %T')] $*"; }

wait_n5() {
  log "waiting for n5_g8"
  while true; do
    if [ -f "$WORK/eval120_search/n5_g8/done" ] && [ -f "$WORK/eval120_search/n5_g8/report.json" ]; then
      log "n5_g8 done"; break
    fi
    if [ ! -f "$WORK/eval120_search/n5_g8/done" ] \
       && ! pgrep -f "run_local.sh eval120_search/n5_g8|starter.py .*eval120_search/n5_g8" >/dev/null; then
      n=$(ls "$WORK/eval120_search/n5_g8/outputs" 2>/dev/null | wc -l | tr -d ' ')
      log "n5 process gone, pickles=$n"
      if [ "${n:-0}" -gt 100 ] && [ ! -f "$WORK/eval120_search/n5_g8/report.json" ]; then
        "$VENV/bin/python" "$HERE/finalize.py" \
          --data "$DATA" --solutions "$SOL" \
          --outputs "$WORK/eval120_search/n5_g8/outputs" \
          --submission "$WORK/eval120_search/n5_g8/submission.json" \
          --report "$WORK/eval120_search/n5_g8/report.json"
        touch "$WORK/eval120_search/n5_g8/done"
        break
      fi
    fi
    sleep 60
  done
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${used:-0}" -lt 400 ]; then
      log "GPU free memory_used=${used}MiB"; return
    fi
    sleep 20
  done
  log "GPU did not drain"; exit 1
}

wait_n5

if [ -f "$WORK/eval120_search/n7_g8/done" ]; then
  log "skip n7_g8 (done)"
else
  log "start n7_g8"
  export PYTHONHASHSEED=0
  export ARC_LORA_SEED=42
  export ARC_TRAIN_AUG_SEED=1
  export ARC_EVAL_AUG_SEED=2
  export ARC_SCORE_SEED_OFFSET=0
  export ARC_N_TRAIN_AUG=7
  export ARC_N_EVAL_AUG=1
  export ARC_N_EVAL_GEOS=8
  bash "$HERE/run_local.sh" eval120_search/n7_g8 0 --skip-done
fi

log "n7_g8 finished"
date
