#!/usr/bin/env bash
# Generation-axis GPU queue. Starts after v11 6×6 two-pass finishes.
#
# Order:
#   1. 6×6 n_eval_aug=2 (decode color perm) — same A seeds as 6×6 A
#      then CPU-ablate back to aug=1 on the same TTT pickle
#   2. TTT hparams, one factor at a time, still 6×6 / seed A / aug=1:
#        lr=2e-5, lr=1e-4, epochs=2, lora_r=128, lora_r=512
#
# Each job is skip-if-done. A failed job (e.g. r=512 OOM) is marked and
# the queue continues. Does not touch the running v11 process.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
LOGDIR=$WORK/eval120_search_logs
SEARCH=$WORK/eval120_search
V11_DONE=$WORK/eval120_n6x6_v11_pool/done

log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$LOGDIR" "$SEARCH"

reset_recipe() {
  unset ARC_LR ARC_EPOCHS ARC_LORA_R ARC_LORA_ALPHA
  export PYTHONHASHSEED=0
  export ARC_N_TRAIN_AUG=6
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

wait_v11() {
  log "waiting for v11 two-pass ($V11_DONE)"
  while true; do
    if [ -f "$V11_DONE" ]; then
      log "v11 pool done"
      break
    fi
    if ! pgrep -f "run_n6x6_two.sh|run_local.sh eval120_n6x6_v11" >/dev/null; then
      log "v11 process gone without pool/done — continue after GPU drain"
      break
    fi
    sleep 60
  done
  wait_gpu
}

compare() {
  local tag=$1 outputs=$2
  "$VENV/bin/python" "$HERE/compare_gen.py" --tag "$tag" --outputs "$outputs" || true
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

wait_v11
log "=== gen queue start ==="

run_job eval120_search/n6x6_aug2 ARC_N_EVAL_AUG=2

if [ -d "$SEARCH/n6x6_aug2/outputs" ] && [ ! -f "$SEARCH/n6x6_aug2_g6a1/report.json" ]; then
  log "CPU ablate n6x6_aug2 → geos=6 aug=1 (same TTT)"
  mkdir -p "$SEARCH/n6x6_aug2_g6a1"
  "$VENV/bin/python" "$HERE/score_view_subset.py" \
    --outputs "$SEARCH/n6x6_aug2/outputs" \
    --geos 6 --n-eval-aug 1 \
    --data "$DATA" --solutions "$SOL" \
    --report "$SEARCH/n6x6_aug2_g6a1/report.json" \
    --submission "$SEARCH/n6x6_aug2_g6a1/submission.json" || true
fi

run_job eval120_search/n6x6_lr2e5 ARC_LR=2e-5
run_job eval120_search/n6x6_lr1e4 ARC_LR=1e-4
run_job eval120_search/n6x6_ep2 ARC_EPOCHS=2
run_job eval120_search/n6x6_r128 ARC_LORA_R=128
run_job eval120_search/n6x6_r512 ARC_LORA_R=512

log "=== gen queue done ==="
if [ -f "$SEARCH/gen_queue_summary.json" ]; then
  "$VENV/bin/python" - << 'PY'
import json
from pathlib import Path
d=json.loads(Path("/opt/work/nvarc/eval120_search/gen_queue_summary.json").read_text())
base=d.get("baseline_6x6A") or {}
print(f"{'tag':<28} {'mean_q':>7} {'kgmon':>7} {'oracle':>7}  d_mq  d_ora")
if base:
    print(f"{'6x6A baseline':<28} {base['mean_quality']:7.3f} {base['kgmon']:7.3f} {base['oracle']:7.3f}")
for r in d.get("runs", []):
    dm=r["mean_quality"]-base.get("mean_quality", 0) if base else 0
    do=r["oracle"]-base.get("oracle", 0) if base else 0
    print(f"{r['tag']:<28} {r['mean_quality']:7.3f} {r['kgmon']:7.3f} {r['oracle']:7.3f}  {dm:+5.2f} {do:+5.2f}")
PY
fi
date
touch "$SEARCH/gen_queue.done"
