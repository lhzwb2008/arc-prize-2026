#!/usr/bin/env bash
# After n=5×8 and n=6×8 single passes: take the better recipe, run a second
# independent TTT pass (half-B seeds), pool with pure kgmon (AB mixed rank).
#
# Pass A is reused (n5_g8 or n6_g8). Target: beat 33% on the official 120.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
LOGDIR=$WORK/eval120_pair_logs
PAIR=$WORK/eval120_pair
mkdir -p "$LOGDIR" "$PAIR"

log() { echo "[$(date '+%F %T')] $*"; }

wait_gpu_free() {
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${used:-0}" -lt 400 ]; then
      log "GPU free memory_used=${used}MiB"
      return
    fi
    sleep 20
  done
  log "GPU did not drain"; exit 1
}

wait_n5() {
  log "waiting for n5_g8"
  while true; do
    if [ -f "$WORK/eval120_search/n5_g8/done" ] && [ -f "$WORK/eval120_search/n5_g8/report.json" ]; then
      log "n5_g8 done"
      break
    fi
    if [ ! -f "$WORK/eval120_search/n5_g8/done" ] \
       && ! pgrep -f "run_local.sh eval120_search/n5_g8|starter.py .*eval120_search/n5_g8" >/dev/null; then
      n=$(ls "$WORK/eval120_search/n5_g8/outputs" 2>/dev/null | wc -l | tr -d ' ')
      log "n5 process gone, pickles=$n — finalize if needed"
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
  wait_gpu_free
}

wait_n5

N=$("$VENV/bin/python" - << 'PY'
import json
from pathlib import Path
w = Path("/opt/work/nvarc/eval120_search")
s5 = json.loads((w / "n5_g8/report.json").read_text())["score"]
s6 = json.loads((w / "n6_g8/report.json").read_text())["score"]
# tie -> n=6 (pass A already known good)
n = 6 if s6 >= s5 else 5
print(f"{n} {s5:.6f} {s6:.6f}")
open("/opt/work/nvarc/eval120_pair/choice.txt", "w").write(
    f"winner_ntrain={n}\nn5={s5}\nn6={s6}\n"
)
PY
)
WIN=${N%% *}
log "winner n_train=$N"

PASS_A=$WORK/eval120_search/n${WIN}_g8
PASS_B_NAME=eval120_pair/n${WIN}_b
POOL=$PAIR/n${WIN}_pool
mkdir -p "$POOL"

export PYTHONHASHSEED=0
export ARC_N_TRAIN_AUG=$WIN
export ARC_N_EVAL_AUG=1
export ARC_N_EVAL_GEOS=8

if [ -f "$WORK/$PASS_B_NAME/done" ]; then
  log "skip pass-B (done)"
else
  log "start pass-B n_train=$WIN geos=8 (half-B seeds)"
  export ARC_LORA_SEED=137
  export ARC_TRAIN_AUG_SEED=17
  export ARC_EVAL_AUG_SEED=29
  export ARC_SCORE_SEED_OFFSET=7
  bash "$HERE/run_local.sh" "$PASS_B_NAME" 0 --skip-done
fi

log "pool A=$PASS_A + B=$WORK/$PASS_B_NAME (kgmon mixed top-2)"
"$VENV/bin/python" "$HERE/finalize.py" \
  --data "$DATA" --solutions "$SOL" \
  --outputs "$PASS_A/outputs" \
  --outputs-extra "$WORK/$PASS_B_NAME/outputs" \
  --submission "$POOL/submission.json" \
  --report "$POOL/report.json"

"$VENV/bin/python" - << PY
import json
from pathlib import Path
pool = json.loads(Path("$POOL/report.json").read_text())
half = json.loads(Path("$WORK/eval120_half_pool/report.json").read_text())
s, n = pool["score"], pool["n_tasks"]
pct = 100.0 * s / n
print(f"pair n={$WIN} pool {s:.3f}/{n} = {pct:.2f}%")
print(f"half2 pool {half['score']:.3f}/{half['n_tasks']} = {100*half['score']/half['n_tasks']:.2f}%")
print(f"beat_33pct={pct > 33.0}  beat_half2={s > half['score']}")
Path("$POOL/summary.json").write_text(json.dumps({
    "n_train": $WIN,
    "pair_score": s,
    "pair_pct": pct,
    "half2_score": half["score"],
    "beat_33pct": pct > 33.0,
    "beat_half2": s > half["score"],
}, indent=2) + "\n")
PY

touch "$POOL/done" "$PAIR/done"
log "best-pair done"
date
