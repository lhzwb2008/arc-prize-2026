#!/usr/bin/env bash
# After v11 6×6 two-pass: find the cheapest (n_train × n_geos) whose candidate
# pool (oracle) still matches 8×8 / 16×16.
#
# Views 2/4/5/6/8 are CPU subsets of existing 8-view pickles (same TTT adapter).
# Train=7 is the only new GPU job: resume eval120_search/n7_g8 (62/120 done).
# Does not start the old gen_queue (lr / epochs / lora_r).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
LOGDIR=$WORK/eval120_search_logs
SEARCH=$WORK/eval120_search
V11_DONE=$WORK/eval120_n6x6_v11_pool/done
OUT=$SEARCH/min_pool_views.json

log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$LOGDIR" "$SEARCH"

cpu_table() {
  log "CPU oracle × view table"
  "$VENV/bin/python" "$HERE/oracle_view_table.py" \
    --geos 2,4,5,6,8 \
    --out "$OUT" \
    --run "16x16=$WORK/eval120/outputs" \
    --run "8x8A=$WORK/eval120_half_a/outputs" \
    --run "8x8B=$WORK/eval120_half_b/outputs" \
    --run "6x8=$WORK/eval120_search/n6_g8/outputs" \
    --run "5x8=$WORK/eval120_search/n5_g8/outputs" \
    --run "7x8=$WORK/eval120_search/n7_g8/outputs" \
    || true
}

wait_gpu() {
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${used:-0}" -lt 400 ]; then
      log "GPU free memory_used=${used}MiB"
      return
    fi
    sleep 20
  done
  log "GPU did not drain below 400MiB"
  exit 1
}

wait_v11() {
  log "waiting for v11 two-pass ($V11_DONE)"
  while true; do
    if [ -f "$V11_DONE" ]; then
      log "v11 pool done"
      break
    fi
    if ! pgrep -af "run_n6x6_two.sh|run_local.sh eval120_n6x6_v11|starter.py .*eval120_n6x6_v11" \
          | grep -v "run_min_pool_search" >/dev/null; then
      log "v11 process gone without pool/done — continue after GPU drain"
      break
    fi
    sleep 60
  done
  wait_gpu
}

# Views are independent of the GPU job; run this first so we have 8×5 / 8×4
# numbers tonight even if v11 B still has hours left.
cpu_table

wait_v11

export PYTHONHASHSEED=0
export ARC_LORA_SEED=42
export ARC_TRAIN_AUG_SEED=1
export ARC_EVAL_AUG_SEED=2
export ARC_SCORE_SEED_OFFSET=0
export ARC_N_EVAL_AUG=1
export ARC_N_TRAIN_AUG=7
export ARC_N_EVAL_GEOS=8

if [ -f "$SEARCH/n7_g8/done" ]; then
  log "skip n7_g8 (done)"
else
  log "resume n7_g8 (skip-done; was 62/120)"
  bash "$HERE/run_local.sh" "eval120_search/n7_g8" 0 --skip-done
fi

cpu_table

log "min-pool search done"
date
touch "$SEARCH/min_pool.done"
