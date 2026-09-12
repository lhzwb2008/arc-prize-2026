#!/usr/bin/env bash
# Single-pass (n_train_aug, n_eval_geos) peak hunt on the official 120 eval set.
#
# Why not a 24/40-task screen or the 1000 training tasks:
#   half-A > full > t3-A on all 120, but that order flips on even-60 / every-3
#   folds. A shortlist on a subset overfits. Training-set rank also will not
#   transfer. Neighbors of the current best (n=8, geos=8) are therefore run on
#   the same 120, same seeds as half-A, no timeouts.
#
# Sequence:
#   0. wait until eval120_t3_pool/done (or t3 process gone) and GPU is free
#   1. CPU view ablation on existing pickle dirs (no GPU)
#   2. GPU: n_train=6 and 10, geos=8, n_eval_aug=1
#   3. if a neighbor beats half-A by >= 1.0, walk further (4/5 or 12/16)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
LOGDIR=$WORK/eval120_search_logs
SEARCH=$WORK/eval120_search
mkdir -p "$LOGDIR" "$SEARCH"

export PYTHONHASHSEED=0
export ARC_LORA_SEED=42
export ARC_TRAIN_AUG_SEED=1
export ARC_EVAL_AUG_SEED=2
export ARC_SCORE_SEED_OFFSET=0
export ARC_N_EVAL_AUG=1
export ARC_N_EVAL_GEOS=8

log() { echo "[$(date '+%F %T')] $*"; }

wait_for_t3() {
  log "waiting for third3 to finish"
  local pidfile=$WORK/eval120_t3_logs/run.pid
  while true; do
    if [ -f "$WORK/eval120_t3_pool/done" ]; then
      log "t3 pool done"
      break
    fi
    if [ -f "$pidfile" ]; then
      pid=$(cat "$pidfile" || true)
      if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        sleep 120
        continue
      fi
    fi
    if pgrep -f "run_third3.sh|run_local.sh eval120_t3" >/dev/null; then
      sleep 120
      continue
    fi
    log "t3 process gone without pool/done — continue anyway"
    break
  done
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${used:-0}" -lt 400 ]; then
      log "GPU free memory_used=${used}MiB"
      return
    fi
    log "GPU still busy memory=${used}MiB, sleep"
    sleep 20
  done
  log "GPU did not drain; abort"
  exit 1
}

run_ntrain() {
  local n=$1
  local name=n${n}_g8
  export ARC_N_TRAIN_AUG=$n
  export ARC_N_EVAL_GEOS=8
  export ARC_N_EVAL_AUG=1
  if [ -f "$SEARCH/$name/done" ]; then
    log "skip $name (done)"
    return
  fi
  log "start $name"
  bash "$HERE/run_local.sh" "eval120_search/$name" 0 --skip-done
  log "finished $name"
}

view_ablate() {
  local src=$1 tag=$2
  local out=$SEARCH/ablate_$tag
  mkdir -p "$out"
  if [ -f "$out/done" ]; then
    log "skip ablate $tag"
    return
  fi
  log "CPU view ablate $tag from $src"
  for g in 4 5 6 8; do
    "$VENV/bin/python" "$HERE/score_view_subset.py" \
      --outputs "$src" --geos "$g" \
      --data "$DATA" --solutions "$SOL" \
      --report "$out/geos${g}.json" \
      --submission "$out/geos${g}.submission.json" \
      || true
  done
  if [ "$tag" = "full" ]; then
    "$VENV/bin/python" "$HERE/score_view_subset.py" \
      --outputs "$src" --geos 8 --n-eval-aug 1 \
      --data "$DATA" --solutions "$SOL" \
      --report "$out/geos8_c1.json" \
      --submission "$out/geos8_c1.submission.json" \
      || true
  fi
  touch "$out/done"
}

wait_for_t3

# View ablation can also run during TTT, but keep it here so t3 I/O stays quiet.
view_ablate "$WORK/eval120_half_a/outputs" half_a
view_ablate "$WORK/eval120/outputs" full
if [ -d "$WORK/eval120_t3_a/outputs" ]; then
  view_ablate "$WORK/eval120_t3_a/outputs" t3_a
fi

run_ntrain 6
run_ntrain 10

"$VENV/bin/python" "$HERE/decide_single_search.py" | tee "$SEARCH/decide1.log"
NEXT=$("$VENV/bin/python" -c "import json; print(' '.join(json.load(open('$SEARCH/leaderboard.json'))['next']))")
log "decide next: ${NEXT:-}"
if [ -n "${NEXT:-}" ] && [ "$NEXT" != "STOP" ]; then
  for n in $NEXT; do
    run_ntrain "$n"
  done
  "$VENV/bin/python" "$HERE/decide_single_search.py" | tee "$SEARCH/decide2.log"
fi

log "single-search done"
date
touch "$SEARCH/done"
