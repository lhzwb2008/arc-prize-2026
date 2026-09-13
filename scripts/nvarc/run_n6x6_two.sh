#!/usr/bin/env bash
# Two 6×6 TTT passes on the official 120 eval, then pool — local stand-in for
# the Kaggle NVARC+ notebook (SEED_A / SEED_B, cheap-first omitted locally
# because starter.py has no --order; both passes still run all 120, no caps).
#
# Recipe: ARC_N_TRAIN_AUG=6, ARC_N_EVAL_GEOS=6, ARC_N_EVAL_AUG=1.
# Pool: pure kgmon top-2 (same as eval120_half_pool) plus keep-primary (Kaggle checkpoint).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
A_NAME=${NVARC_N6X6_A:-eval120_n6x6_a}
B_NAME=${NVARC_N6X6_B:-eval120_n6x6_b}
POOL=$WORK/eval120_n6x6_pool
META=$POOL/timing.json

log() { echo "[$(date '+%F %T')] $*"; }

stock() {
  export PYTHONHASHSEED=0
  export ARC_N_TRAIN_AUG=6
  export ARC_N_EVAL_AUG=1
  export ARC_N_EVAL_GEOS=6
}

mkdir -p "$POOL"
stock

log "=== n6x6-A (stock seeds, 6 train augs / 6 decode views) ==="
export ARC_LORA_SEED=42
export ARC_TRAIN_AUG_SEED=1
export ARC_EVAL_AUG_SEED=2
export ARC_SCORE_SEED_OFFSET=0
t0=$(date +%s)
bash "$HERE/run_local.sh" "$A_NAME" 0 --skip-done
t1=$(date +%s)
A_SEC=$((t1 - t0))
log "pass A wall ${A_SEC}s = $(awk -v s="$A_SEC" 'BEGIN{printf "%.2fh", s/3600}')"

log "=== n6x6-B (NVARC+ seeds, same 6×6) ==="
export ARC_LORA_SEED=137
export ARC_TRAIN_AUG_SEED=17
export ARC_EVAL_AUG_SEED=29
export ARC_SCORE_SEED_OFFSET=7
t2=$(date +%s)
bash "$HERE/run_local.sh" "$B_NAME" 0 --skip-done
t3=$(date +%s)
B_SEC=$((t3 - t2))
log "pass B wall ${B_SEC}s = $(awk -v s="$B_SEC" 'BEGIN{printf "%.2fh", s/3600}')"

log "=== pool A+B pure kgmon (match half2) ==="
"$VENV/bin/python" "$HERE/finalize.py" \
  --data "$DATA" --solutions "$SOL" \
  --outputs "$WORK/$A_NAME/outputs" \
  --outputs-extra "$WORK/$B_NAME/outputs" \
  --submission "$POOL/submission.json" \
  --report "$POOL/report.json"

log "=== pool A+B keep-primary (Kaggle checkpoint) ==="
"$VENV/bin/python" "$HERE/finalize.py" \
  --data "$DATA" --solutions "$SOL" \
  --outputs "$WORK/$A_NAME/outputs" \
  --outputs-extra "$WORK/$B_NAME/outputs" \
  --keep-primary \
  --submission "$POOL/submission_keep_primary.json" \
  --report "$POOL/report_keep_primary.json"

"$VENV/bin/python" - << PY
import json, time
from pathlib import Path
work = Path("$WORK")
pool = json.loads((work / "eval120_n6x6_pool/report.json").read_text())
kp = json.loads((work / "eval120_n6x6_pool/report_keep_primary.json").read_text())
a = json.loads((work / "$A_NAME/report.json").read_text())
b = json.loads((work / "$B_NAME/report.json").read_text())
half_p = work / "eval120_half_pool/report.json"
half = json.loads(half_p.read_text()) if half_p.exists() else None
n = pool["n_tasks"]
def row(tag, d):
    s = d["score"]
    print(f"{tag:22s} {s:7.3f}/{n} = {100*s/n:5.2f}%")
    return s
print("--- n6x6 two-pass ---")
sa = row("pass A 6x6", a)
sb = row("pass B 6x6", b)
sp = row("pool kgmon", pool)
sk = row("pool keep-primary", kp)
if half:
    row("half2 8x8 pool", half)
print(f"A_sec=$A_SEC B_sec=$B_SEC total_sec={int('$A_SEC')+int('$B_SEC')}")
meta = {
    "recipe": {"n_train_aug": 6, "n_eval_geos": 6, "n_eval_aug": 1},
    "pass_a_sec": int("$A_SEC"),
    "pass_b_sec": int("$B_SEC"),
    "total_sec": int("$A_SEC") + int("$B_SEC"),
    "pass_a_score": sa,
    "pass_b_score": sb,
    "pool_kgmon": sp,
    "pool_keep_primary": sk,
    "half2_pool": None if not half else half["score"],
    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
Path("$META").write_text(json.dumps(meta, indent=2) + "\n")
PY

touch "$POOL/done"
log "n6x6 two-pass done"
date
