#!/usr/bin/env bash
# Third independent 6×6 TTT pass (SEED_C from the Kaggle notebook), then
# pool AC / BC / ABC against the existing A/B pair.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
C_NAME=${NVARC_N6X6_C:-eval120_n6x6_c}
A_NAME=${NVARC_N6X6_A:-eval120_n6x6_a}
B_NAME=${NVARC_N6X6_B:-eval120_n6x6_b}
OUT=$WORK/eval120_n6x6_c_pool

log() { echo "[$(date '+%F %T')] $*"; }

export PYTHONHASHSEED=0
export ARC_N_TRAIN_AUG=6
export ARC_N_EVAL_AUG=1
export ARC_N_EVAL_GEOS=6
export ARC_LORA_SEED=271
export ARC_TRAIN_AUG_SEED=31
export ARC_EVAL_AUG_SEED=41
export ARC_SCORE_SEED_OFFSET=13

mkdir -p "$OUT"
log "=== n6x6-C (SEED_C, 6 train augs / 6 decode views) ==="
t0=$(date +%s)
bash "$HERE/run_local.sh" "$C_NAME" 0 --skip-done
t1=$(date +%s)
C_SEC=$((t1 - t0))
log "pass C wall ${C_SEC}s = $(awk -v s="$C_SEC" 'BEGIN{printf "%.2fh", s/3600}')"

pool() {
  local tag=$1; shift
  mkdir -p "$OUT/$tag"
  "$VENV/bin/python" "$HERE/finalize.py" \
    --data "$DATA" --solutions "$SOL" \
    --submission "$OUT/$tag/submission.json" \
    --report "$OUT/$tag/report.json" \
    "$@"
}

log "=== pool AC ==="
pool ac --outputs "$WORK/$A_NAME/outputs" --outputs-extra "$WORK/$C_NAME/outputs"
log "=== pool BC ==="
pool bc --outputs "$WORK/$B_NAME/outputs" --outputs-extra "$WORK/$C_NAME/outputs"
log "=== pool ABC ==="
pool abc --outputs "$WORK/$A_NAME/outputs" \
  --outputs-extra "$WORK/$B_NAME/outputs" \
  --outputs-extra "$WORK/$C_NAME/outputs"

"$VENV/bin/python" - << PY
import json, time
from pathlib import Path
work = Path("$WORK")
out = work / "eval120_n6x6_c_pool"
def sc(p):
    d = json.loads(Path(p).read_text())
    return d["score"], d["n_tasks"]
rows = []
for tag, path in [
    ("A", work / "$A_NAME/report.json"),
    ("B", work / "$B_NAME/report.json"),
    ("C", work / "$C_NAME/report.json"),
    ("AB", work / "eval120_n6x6_pool/report.json"),
    ("AC", out / "ac/report.json"),
    ("BC", out / "bc/report.json"),
    ("ABC", out / "abc/report.json"),
]:
    if path.exists():
        s, n = sc(path)
        print(f"{tag:4s} {s:7.3f}/{n} = {100*s/n:5.2f}%")
        rows.append((tag, s, n))
meta = {
    "pass_c_sec": int("$C_SEC"),
    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "scores": {t: s for t, s, _ in rows},
}
(out / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
PY
touch "$OUT/done"
log "n6x6-C done"
date
