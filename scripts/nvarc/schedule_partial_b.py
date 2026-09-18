#!/usr/bin/env python3
"""After pass A, leftover-B until the Kaggle-12h-equivalent wall.

Default cap is max(first 16x16 pickle span, GPU 6x6 two-pass spans).
6x6 two-pass is known COMPLETE on hidden Kaggle 12h. --kaggle12 uses a
literal 12h-20min clock instead.

B queue order comes from b_priority_queue.py: pass-A top-1 vote share per
unit estimated cost (generic rule from A's own outputs, no ids). Falls back
to cheap-first. Pooling is mixed mean_quality. Does not push a kernel and
does not submit.

  python schedule_partial_b.py --dry-run
  python schedule_partial_b.py --dry-run --kaggle12
  python schedule_partial_b.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.getenv("NVARC_WORK", "/opt/work/nvarc"))
DATA = os.getenv("NVARC_DATA", "/opt/data/kaggle/arc-agi_evaluation_challenges.json")
SOL = os.getenv("NVARC_SOL", "/opt/data/kaggle/arc-agi_evaluation_solutions.json")
PYTHON = os.getenv("NVARC_PYTHON", sys.executable)

A_NAME = os.getenv("NVARC_N8X6_A", "eval120_n8x6_a")
B_NAME = os.getenv("NVARC_PARTIAL_B", "eval120_n8x6_b_cheap")
SUM = WORK / os.getenv("NVARC_PARTIAL_SUM", "eval120_n8x6_partial")
PRIMARY = str(WORK / A_NAME / "outputs")
B_OUT = str(WORK / B_NAME / "outputs")
SUB = str(SUM / "submission.json")
T16_H = float(os.getenv("NVARC_T16_HOURS", "13.58"))
KAGGLE_H = float(os.getenv("NVARC_KAGGLE_HOURS", "12.0"))
FULL_16 = str(WORK / os.getenv("NVARC_FULL_16X16", "eval120") / "outputs")
N6A = str(WORK / os.getenv("NVARC_N6X6_A", "eval120_n6x6_v11_a") / "outputs")
N6B = str(WORK / os.getenv("NVARC_N6X6_B", "eval120_n6x6_v11_b") / "outputs")
N6OLD_A = str(WORK / os.getenv("NVARC_N6X6_OLD_A", "eval120_n6x6_a") / "outputs")
N6OLD_B = str(WORK / os.getenv("NVARC_N6X6_OLD_B", "eval120_n6x6_b") / "outputs")
N6_TIMING = str(WORK / os.getenv("NVARC_N6X6_TIMING", "eval120_n6x6_pool/timing.json"))


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def pickle_span_h(out_dir: str) -> float:
    p = Path(out_dir)
    if not p.is_dir():
        return 0.0
    ts = [f.stat().st_mtime for f in p.iterdir() if f.is_file()]
    if len(ts) < 2:
        return 0.0
    return (max(ts) - min(ts)) / 3600.0


def two_pass_hours(a_dir: str, b_dir: str) -> float:
    a = pickle_span_h(a_dir)
    b = pickle_span_h(b_dir)
    if a <= 0 or b <= 0:
        return 0.0
    return a + b


def kaggle_equiv_cap() -> tuple[float, str, dict[str, float]]:
    forced = os.getenv("NVARC_CAP_HOURS")
    t16 = pickle_span_h(FULL_16) or T16_H
    walls = {"16x16": t16}
    n6 = two_pass_hours(N6A, N6B)
    if n6 > 0:
        walls["n6x6_v11_A+B"] = n6
    n6old = two_pass_hours(N6OLD_A, N6OLD_B)
    if n6old > 0:
        walls["n6x6_old_A+B"] = n6old
    n6t = 0.0
    tp = Path(N6_TIMING)
    if tp.is_file():
        try:
            d = json.loads(tp.read_text())
            if isinstance(d, dict) and d.get("total_sec") is not None:
                n6t = float(d["total_sec"]) / 3600.0
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            n6t = 0.0
    if n6t > 0:
        walls["n6x6_old_timing"] = n6t
    cap_name = max(walls, key=walls.get)
    cap_h = walls[cap_name]
    if forced:
        return float(forced), "NVARC_CAP_HOURS", walls
    return cap_h, cap_name, walls


def main() -> int:
    dry = "--dry-run" in sys.argv
    cap, cap_name, walls = kaggle_equiv_cap()
    if "--kaggle12" in sys.argv:
        cap = KAGGLE_H - 20.0 / 60.0
        cap_name = "literal-12h-20min"
    a_h = pickle_span_h(PRIMARY)
    b_budget = max(0.15, cap - a_h)
    wall_s = " ".join(f"{k}={v:.2f}h" for k, v in walls.items())
    log(
        f"walls {wall_s}  A {a_h:.2f}h  cap {cap:.2f}h ({cap_name})  "
        f"leftover-B {b_budget:.2f}h  order=A-uncertainty/cost (fallback cheap)  pool=mixed mean_quality"
    )
    if dry:
        log("dry-run: not launching starter")
        return 0

    SUM.mkdir(parents=True, exist_ok=True)
    Path(B_OUT).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "UNSLOTH_DISABLE_STATISTICS": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "ARC_N_EVAL_AUG": "1",
        "ARC_N_TRAIN_AUG": "8",
        "ARC_N_EVAL_GEOS": "6",
        "ARC_LORA_SEED": "137",
        "ARC_TRAIN_AUG_SEED": "17",
        "ARC_EVAL_AUG_SEED": "29",
        "ARC_SCORE_SEED_OFFSET": "7",
        "NVARC_TASK_LIMIT": "0",
        "NVARC_DFS_LIMIT": "0",
        "NVARC_POOL_MODE": "mixed",
        "NVARC_CHECKPOINT_PRIMARY": PRIMARY,
        "NVARC_CHECKPOINT_EXTRAS": B_OUT,
        "NVARC_CHECKPOINT_SUB": SUB,
        "NVARC_CHECKPOINT_EVERY": "90",
        "NVARC_CHECKPOINT_MIN_GAP": "300",
        "NVARC_DATA": DATA,
        "NVARC_OUT": B_OUT,
    })
    if Path("/usr/local/cuda/bin/ptxas").exists():
        env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

    # B queue: pass-A uncertainty per unit cost (b_priority_queue.py); no ids, no labels.
    keys_file = str(SUM / "b_priority_keys.json")
    qcmd = [
        PYTHON, str(HERE / "b_priority_queue.py"),
        "--data", DATA, "--outputs", PRIMARY,
        "--skip-done", B_OUT, "--keys-out", keys_file,
    ]
    log(" ".join(qcmd))
    rcq = subprocess.call(qcmd, env=env)
    order_args = ["--order", "file", "--keys-file", keys_file] if rcq == 0 else ["--order", "cheap"]
    if rcq != 0:
        log(f"b_priority_queue rc={rcq}; falling back to --order cheap")

    end_time = time.time() + b_budget * 3600
    cmd = [
        PYTHON, str(HERE / "starter.py"),
        "--data", DATA,
        "--out", B_OUT,
        "--end-time", str(end_time),
        *order_args,
        "--skip-done",
    ]
    log(" ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    log(f"starter rc={rc}")

    fin = [
        PYTHON, str(HERE / "finalize.py"),
        "--data", DATA, "--solutions", SOL,
        "--outputs", PRIMARY, "--outputs-extra", B_OUT,
        "--pool-mode", "mixed",
        "--submission", SUB,
        "--report", str(SUM / "report.json"),
    ]
    log(" ".join(fin))
    rc2 = subprocess.call(fin, env=env)
    log(f"finalize rc={rc2} mixed mean_quality")
    return rc2 or rc


if __name__ == "__main__":
    raise SystemExit(main())
