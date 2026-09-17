#!/usr/bin/env python3
"""After pass A, leftover-B cheap-first until the 16x16 (or Kaggle 12h) wall.

Generic policy, no per-task key list: starter --order cheap + --end-time.
Pooling is mixed mean_quality. Does not push a kernel and does not submit.

  python schedule_partial_b.py --dry-run
  python schedule_partial_b.py --dry-run --kaggle12
  python schedule_partial_b.py
"""
from __future__ import annotations

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


def main() -> int:
    dry = "--dry-run" in sys.argv
    cap = T16_H
    if "--kaggle12" in sys.argv:
        cap = KAGGLE_H - 20.0 / 60.0
    a_h = pickle_span_h(PRIMARY)
    b_budget = max(0.15, cap - a_h)
    log(
        f"A wall {a_h:.2f}h  cap {cap:.2f}h  leftover-B {b_budget:.2f}h  "
        f"order=cheap  pool=mixed mean_quality"
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
        "NVARC_DATA": DATA,
        "NVARC_OUT": B_OUT,
    })
    if Path("/usr/local/cuda/bin/ptxas").exists():
        env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

    end_time = time.time() + b_budget * 3600
    cmd = [
        PYTHON, str(HERE / "starter.py"),
        "--data", DATA,
        "--out", B_OUT,
        "--end-time", str(end_time),
        "--order", "cheap",
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
