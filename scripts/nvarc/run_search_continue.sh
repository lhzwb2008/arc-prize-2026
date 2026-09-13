#!/usr/bin/env bash
# Single-pass search continuation after n=5×8.
#
# Findings so far (do not two-pass on this box):
#   - n_train 6×8 36.83 > 8×8 36.00 > 16×16 33.83 > 5×5 31.83
#   - geos 6 == geos 8 on both n=6 and n=8 TTT; extra 2 views buy 0–1 tasks
#   - full-solve overlap is large (~30) but not nested: 5×5 uniquely hits
#     80a900e0/c4d067a0; 8×8 vs 6×8 trade 2 vs 5 tasks
#   - same recipe, different seed (halfA vs halfB) also trades ~5 full solves
#
# After n5: run 7×8 (fill 6–8 gap), CPU-ablate views, then n=9×8 only if 7
# clearly beats 6. Two-pass / extra seeds go to Kaggle.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}

log() { echo "[$(date '+%F %T')] $*"; }

stock_seeds() {
  export PYTHONHASHSEED=0
  export ARC_LORA_SEED=42
  export ARC_TRAIN_AUG_SEED=1
  export ARC_EVAL_AUG_SEED=2
  export ARC_SCORE_SEED_OFFSET=0
  export ARC_N_EVAL_AUG=1
}

wait_gpu() {
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${used:-0}" -lt 400 ]; then
      log "GPU free memory_used=${used}MiB"; return
    fi
    sleep 20
  done
  log "GPU did not drain"; exit 1
}

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
  wait_gpu
}

run_ntrain() {
  local n=$1
  stock_seeds
  export ARC_N_TRAIN_AUG=$n
  export ARC_N_EVAL_GEOS=8
  if [ -f "$WORK/eval120_search/n${n}_g8/done" ]; then
    log "skip n${n}_g8 (done)"; return
  fi
  log "start n${n}_g8"
  bash "$HERE/run_local.sh" "eval120_search/n${n}_g8" 0 --skip-done
}

ablate() {
  local src=$1 tag=$2
  local out=$WORK/eval120_search/ablate_$tag
  mkdir -p "$out"
  log "CPU view ablate $tag"
  for g in 4 5 6 8; do
    "$VENV/bin/python" "$HERE/score_view_subset.py" \
      --outputs "$src" --geos "$g" \
      --data "$DATA" --solutions "$SOL" \
      --report "$out/geos${g}.json" \
      --submission "$out/geos${g}.submission.json" || true
  done
}

score_of() {
  local f=$1
  "$VENV/bin/python" -c "import json; print(json.load(open('$f'))['score'])"
}

wait_n5
log "overlap after n5"
"$VENV/bin/python" "$HERE/analyze_overlap.py" || true

run_ntrain 7
ablate "$WORK/eval120_search/n7_g8/outputs" n7
log "overlap after n7"
"$VENV/bin/python" "$HERE/analyze_overlap.py" || true

s6=$(score_of "$WORK/eval120_search/n6_g8/report.json")
s7=$(score_of "$WORK/eval120_search/n7_g8/report.json")
s8=$(score_of "$WORK/eval120_half_a/report.json")
log "scores n6=$s6 n7=$s7 halfA=$s8"
NEED9=$("$VENV/bin/python" -c "print('yes' if float('$s7') > max(float('$s6'), float('$s8')) + 0.99 else 'no')")
if [ "$NEED9" = "yes" ]; then
  log "n7 beats n6/halfA by >=1 — run n9_g8"
  run_ntrain 9
  ablate "$WORK/eval120_search/n9_g8/outputs" n9
  "$VENV/bin/python" "$HERE/analyze_overlap.py" || true
else
  log "n7 does not beat n6/halfA by 1pt — stop n_train sweep; next is Kaggle multi-seed or n=6 geos=6"
fi

log "single-pass continuation done"
date
