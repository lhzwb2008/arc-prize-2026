#!/usr/bin/env bash
# Two independent native 8×6 TTT passes (default lr/epochs). No pool —
# pickles stay in eval120_n8x6_{a,b}/outputs for later merging.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
A_NAME=${NVARC_N8X6_A:-eval120_n8x6_a}
B_NAME=${NVARC_N8X6_B:-eval120_n8x6_b}
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

score_one() {
  local tag=$1 outputs=$2
  "$VENV/bin/python" "$HERE/compare_gen.py" --tag "$tag" --outputs "$outputs" \
    --baseline "$WORK/eval120_half_a/outputs" --baseline-geos 6 \
    --summary "$SUM/compare.json" || true
}

mkdir -p "$SUM" "$LOGDIR"
stock

log "=== n8x6-A seed42 8 train / 6 views  lr=5e-5 ep=1 ==="
export ARC_LORA_SEED=42
export ARC_TRAIN_AUG_SEED=1
export ARC_EVAL_AUG_SEED=2
export ARC_SCORE_SEED_OFFSET=0
t0=$(date +%s)
bash "$HERE/run_local.sh" "$A_NAME" 0 --skip-done
t1=$(date +%s)
A_SEC=$((t1 - t0))
log "pass A wall ${A_SEC}s = $(awk -v s="$A_SEC" 'BEGIN{printf "%.2fh", s/3600}')"
score_one "$A_NAME" "$WORK/$A_NAME/outputs"

log "=== n8x6-B seed137 8 train / 6 views  lr=5e-5 ep=1 ==="
export ARC_LORA_SEED=137
export ARC_TRAIN_AUG_SEED=17
export ARC_EVAL_AUG_SEED=29
export ARC_SCORE_SEED_OFFSET=7
t2=$(date +%s)
bash "$HERE/run_local.sh" "$B_NAME" 0 --skip-done
t3=$(date +%s)
B_SEC=$((t3 - t2))
log "pass B wall ${B_SEC}s = $(awk -v s="$B_SEC" 'BEGIN{printf "%.2fh", s/3600}')"
score_one "$B_NAME" "$WORK/$B_NAME/outputs"

"$VENV/bin/python" - << PY
import json, re, statistics, time
from pathlib import Path

work = Path("$WORK")
sum_dir = Path("$SUM")
a_rep = json.loads((work / "$A_NAME" / "report.json").read_text())
b_rep = json.loads((work / "$B_NAME" / "report.json").read_text())
n = a_rep["n_tasks"]

def pct(score):
    return 100.0 * score / n

def times_from_log(text, marker):
    if marker not in text:
        return {}
    rest = text.split(marker, 1)[1]
    nxt = rest.find("=== n8x6-")
    if nxt > 0:
        rest = rest[:nxt]
    out = {}
    for k, s in re.findall(r"\[Rank \d+\] finished (\S+) in ([0-9.]+)s", rest):
        out[k] = float(s)
    return out

log = Path("$LOGDIR/console.log").read_text(errors="replace") if Path("$LOGDIR/console.log").exists() else ""
ta = times_from_log(log, "=== n8x6-A")
tb = times_from_log(log, "=== n8x6-B")

def tstat(d):
    if not d:
        return None
    xs = list(d.values())
    xs.sort()
    return {
        "n": len(xs),
        "mean_s": sum(xs) / len(xs),
        "p50_s": statistics.median(xs),
        "p90_s": xs[int(0.9 * len(xs)) - 1],
        "max_s": xs[-1],
        "sum_h": sum(xs) / 3600.0,
    }

cmp_path = sum_dir / "compare.json"
cmp = json.loads(cmp_path.read_text()) if cmp_path.exists() else {}
runs = {r["tag"]: r for r in cmp.get("runs", [])}

def extra(tag):
    r = runs.get(tag, {})
    return {
        "oracle": r.get("oracle"),
        "mean_quality": r.get("mean_quality"),
        "kgmon": r.get("kgmon"),
        "gold_in_pool": r.get("gold_in_pool"),
        "n_out": r.get("n_out"),
    }

sa, sb = a_rep["score"], b_rep["score"]
print("--- native 8x6 single-pass (no pool) ---")
print(f"{'pass A seed42':22s} {sa:7.3f}/{n} = {pct(sa):5.2f}%  wall={int('$A_SEC')/3600:.2f}h")
print(f"{'pass B seed137':22s} {sb:7.3f}/{n} = {pct(sb):5.2f}%  wall={int('$B_SEC')/3600:.2f}h")
print(f"delta B-A  {pct(sb)-pct(sa):+.2f} pp")
for tag, ts in (("A", ta), ("B", tb)):
    st = tstat(ts)
    if st:
        print(f"  {tag} per-task n={st['n']} mean={st['mean_s']:.0f}s p50={st['p50_s']:.0f} p90={st['p90_s']:.0f} max={st['max_s']:.0f} sum={st['sum_h']:.2f}h")

doc = {
    "recipe": {
        "n_train_aug": 8,
        "n_eval_geos": 6,
        "n_eval_aug": 1,
        "lr": 5e-5,
        "epochs": 1,
        "lora_r": 256,
        "pooled": False,
    },
    "pass_a": {
        "name": "$A_NAME",
        "wall_sec": int("$A_SEC"),
        "report_score": sa,
        "report_pct": pct(sa),
        "decoded": len(a_rep.get("decoded_tasks") or []),
        "per_task_times": tstat(ta),
        **extra("$A_NAME"),
    },
    "pass_b": {
        "name": "$B_NAME",
        "wall_sec": int("$B_SEC"),
        "report_score": sb,
        "report_pct": pct(sb),
        "decoded": len(b_rep.get("decoded_tasks") or []),
        "per_task_times": tstat(tb),
        **extra("$B_NAME"),
    },
    "delta_pct_b_minus_a": pct(sb) - pct(sa),
    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
(sum_dir / "timing.json").write_text(json.dumps(doc, indent=2) + "\n")
print("wrote", sum_dir / "timing.json")
PY

touch "$SUM/done"
log "n8x6 two single-pass done (not pooled)"
date
