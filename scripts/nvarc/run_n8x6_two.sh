#!/usr/bin/env bash
# Native 8×6 TTT: pass A, then leftover-B expensive-first with keep-primary merge.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
A_NAME=${NVARC_N8X6_A:-eval120_n8x6_a}
SUM=$WORK/eval120_n8x6_two
LOGDIR=$WORK/eval120_n8x6_two_logs

log() { echo "[$(date '+%F %T')] $*"; }

stock() {
  unset ARC_LR ARC_EPOCHS ARC_LORA_R ARC_LORA_ALPHA
  export PYTHONHASHSEED=0
  export ARC_N_TRAIN_AUG=8
  export ARC_N_EVAL_AUG=1
  export ARC_N_EVAL_GEOS=6
}

mkdir -p "$SUM" "$LOGDIR"
stock

if [ ! -f "$SUM/t0.epoch" ]; then
  date +%s > "$SUM/t0.epoch"
fi
export NVARC_T0=$(cat "$SUM/t0.epoch")
export NVARC_FINISH_ALL=${NVARC_FINISH_ALL:-1}
export NVARC_PYTHON="$VENV/bin/python"

if [ ! -f "$WORK/$A_NAME/done" ]; then
  log "=== n8x6-A seed42 8 train / 6 views  lr=5e-5 ep=1 ==="
  export ARC_LORA_SEED=42
  export ARC_TRAIN_AUG_SEED=1
  export ARC_EVAL_AUG_SEED=2
  export ARC_SCORE_SEED_OFFSET=0
  bash "$HERE/run_local.sh" "$A_NAME" 0 --skip-done
else
  log "pass A already done ($WORK/$A_NAME/done)"
fi

log "=== n8x6-B expensive-first + v15 keep-primary pool ==="
exec "$VENV/bin/python" -u "$HERE/schedule_local_v14.py"
