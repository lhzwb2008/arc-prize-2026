#!/usr/bin/env bash
# TTT hparam queue on the Kaggle unit (8×6, seed A). Not 6×6: n_train=6
# is below the full-pool floor, so lr/epoch wins there would not transfer.
#
# Kept (unknown, could move 8×6):
#   lr=2e-5, lr=1e-4, epochs=2
# Dropped:
#   n_eval_aug=2 — 16×16 already +~1; 8×6×2-perm blows the 12h two-pass budget
#   lora_r=128/512 — author 256; r=512 OOM risk; low chance of changing recipe
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
LOGDIR=$WORK/eval120_search_logs
SEARCH=$WORK/eval120_search
BASELINE=$WORK/eval120_half_a/outputs

log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$LOGDIR" "$SEARCH"

reset_recipe() {
  unset ARC_LR ARC_EPOCHS ARC_LORA_R ARC_LORA_ALPHA
  export PYTHONHASHSEED=0
  export ARC_N_TRAIN_AUG=8
  export ARC_N_EVAL_GEOS=6
  export ARC_N_EVAL_AUG=1
  export ARC_LORA_SEED=42
  export ARC_TRAIN_AUG_SEED=1
  export ARC_EVAL_AUG_SEED=2
  export ARC_SCORE_SEED_OFFSET=0
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

wait_idle() {
  log "waiting for GPU"
  wait_gpu
}

compare() {
  local tag=$1 outputs=$2
  "$VENV/bin/python" "$HERE/compare_gen.py" --tag "$tag" --outputs "$outputs" \
    --baseline "$BASELINE" --baseline-geos 6 || true
}

run_job() {
  local name=$1
  shift
  local dir=$WORK/$name
  if [ -f "$dir/done" ]; then
    log "skip $name (done)"
    compare "$name" "$dir/outputs"
    return 0
  fi
  if [ -f "$dir/failed" ]; then
    log "skip $name (previous fail)"
    return 0
  fi
  reset_recipe
  local kv
  for kv in "$@"; do
    export "$kv"
  done
  log "start $name  n_train=$ARC_N_TRAIN_AUG geos=$ARC_N_EVAL_GEOS aug=$ARC_N_EVAL_AUG lr=${ARC_LR:-5e-5} ep=${ARC_EPOCHS:-1} r=${ARC_LORA_R:-256}"
  if bash "$HERE/run_local.sh" "$name" 0 --skip-done; then
    compare "$name" "$dir/outputs"
  else
    log "FAILED $name — mark and continue"
    mkdir -p "$dir"
    date > "$dir/failed"
  fi
}

wait_idle
log "=== 8x6 hparam queue start ==="

run_job eval120_search/n8x6_lr2e5 ARC_LR=2e-5
run_job eval120_search/n8x6_lr1e4 ARC_LR=1e-4
run_job eval120_search/n8x6_ep2 ARC_EPOCHS=2

log "=== 8x6 hparam queue done ==="
if [ -f "$SEARCH/gen_queue_summary.json" ]; then
  "$VENV/bin/python" - << 'PY'
import json
from pathlib import Path
d=json.loads(Path("/opt/work/nvarc/eval120_search/gen_queue_summary.json").read_text())
base=d.get("baseline_8x6A") or {}
print(f"{'tag':<28} {'mean_q':>7} {'kgmon':>7} {'oracle':>7}  d_mq  d_ora")
if base:
    print(f"{'8x6A baseline':<28} {base['mean_quality']:7.3f} {base['kgmon']:7.3f} {base['oracle']:7.3f}")
for r in d.get("runs", []):
    dm=r["mean_quality"]-base.get("mean_quality", 0) if base else 0
    do=r["oracle"]-base.get("oracle", 0) if base else 0
    print(f"{r['tag']:<28} {r['mean_quality']:7.3f} {r['kgmon']:7.3f} {r['oracle']:7.3f}  {dm:+5.2f} {do:+5.2f}")
PY
fi
date
touch "$SEARCH/gen_queue.done"
