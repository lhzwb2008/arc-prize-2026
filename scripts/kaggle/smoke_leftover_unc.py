#!/usr/bin/env python3
"""Smoke the leftover-B 8×6 Kaggle submit path. Does not submit."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL = os.environ.get(
    "NVARC_KERNEL", "wenbozhang2026/nvarc-qwen3-4b-ttt-2026-ceiling"
)
COMP = "arc-prize-2026-arc-agi-2"
NB = ROOT / "notebooks/nvarc_2026/nvarc_qwen3_4b_ttt_2026.ipynb"
KERNEL_INFO = Path(os.environ.get(
    "NVARC_LEFTOVER_KERNEL", "/opt/work/nvarc/leftover_unc_kernel.json"
))
STATE = Path(os.environ.get(
    "NVARC_LEFTOVER_STATE", "/opt/work/nvarc/leftover_unc_submit_state.json"
))
WINDOW = "2026-09-19T00:00:00+00:00"


def find_kaggle(explicit: str = "") -> str:
    for c in (
        explicit,
        os.environ.get("NVARC_KAGGLE", ""),
        "/opt/venv_kaggle/bin/kaggle",
    ):
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise SystemExit("FAIL: kaggle CLI not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", default="")
    ap.add_argument(
        "--allow-missing-kernel-info",
        action="store_true",
        help="pre-push smoke: notebook/creds may pass before kernels_push",
    )
    args = ap.parse_args()
    fails = []

    src = "".join(
        "".join(c.get("source") or [])
        for c in json.loads(NB.read_text())["cells"]
    )
    if 'SCHEDULE = "leftover"' not in src:
        fails.append(f"notebook {NB} is not SCHEDULE=leftover")
    else:
        print("OK notebook SCHEDULE=leftover")
    if 'SCHEDULE = "single"' in src:
        fails.append("leftover notebook still contains SCHEDULE=single")
    if 'COMMON["ARC_N_TRAIN_AUG"] = "8"' not in src:
        fails.append("notebook is not n_train=8")
    else:
        print("OK notebook n_train=8")
    if 'COMMON["ARC_N_EVAL_GEOS"] = "6"' not in src:
        fails.append("notebook is not geos=6")
    else:
        print("OK notebook geos=6")
    if "Save Version smoke: skip leftover-B" not in src:
        fails.append("notebook missing Save Version skip leftover-B")
    else:
        print("OK Save Version skips leftover-B")
    if "b_priority_queue.py" not in src:
        fails.append("notebook missing b_priority_queue.py")
    else:
        print("OK b_priority_queue.py writefile")
    if "_hard_merge_watchdog" not in src or "hard-Tminus5" not in src:
        fails.append("notebook missing 12h-5min hard merge")
    else:
        print("OK 12h-5min hard merge")
    if 'live_checkpoint("tick"' in src:
        fails.append("notebook still has live checkpoint ticks")
    else:
        print("OK no live ticks")
    if 'NVARC_POOL_MODE": "keep-primary"' not in src:
        fails.append("notebook pool mode is not keep-primary")
    else:
        print("OK keep-primary")

    kaggle = find_kaggle(args.kaggle)
    print(f"OK kaggle CLI {kaggle}")

    cred = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    if not (cred / "credentials.json").exists() and not (cred / "kaggle.json").exists():
        fails.append(f"missing kaggle credentials in {cred}")
    else:
        print(f"OK credentials dir {cred}")

    p = subprocess.run([kaggle, "kernels", "status", KERNEL], capture_output=True, text=True)
    status = ((p.stdout or "") + (p.stderr or "")).strip()
    print("kernel status:", status or f"rc={p.returncode}")
    up = status.upper()
    if p.returncode != 0:
        fails.append("kernels status failed")
    elif "ERROR" in up and "COMPLETE" not in up:
        fails.append(f"kernel status error: {status}")

    info = {}
    if KERNEL_INFO.exists():
        info = json.loads(KERNEL_INFO.read_text())
        print(f"OK kernel info {KERNEL_INFO} version={info.get('version_number')} pushed={info.get('pushed_utc')}")
        if info.get("schedule") != "leftover":
            fails.append(f"kernel info schedule={info.get('schedule')} want leftover")
    else:
        print(f"WARN no {KERNEL_INFO} yet (push first)")
        if not args.allow_missing_kernel_info:
            fails.append(f"missing {KERNEL_INFO}; run push_leftover_unc.py")

    p = subprocess.run(
        [kaggle, "competitions", "submissions", "-c", COMP, "--csv"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        fails.append("submissions list failed: " + (p.stderr or p.stdout)[-300:])
        rows = []
    else:
        rows = list(csv.DictReader(io.StringIO(p.stdout)))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    last = (rows[0].get("date") or "")[:10] if rows else None
    print(f"OK last submit utc date={last} today={today} n={len(rows)}")
    if last == today:
        print("WARN today's UTC quota is already used; 08:00 CST 19 Sep is the next window")
    window = datetime.fromisoformat(WINDOW)
    now = datetime.now(timezone.utc)
    print(f"submit window {window.isoformat()}  now {now.isoformat()}  "
          f"{'OPEN' if now >= window else 'still closed'}")

    if STATE.exists():
        st = json.loads(STATE.read_text())
        print(f"submit state exists ref={st.get('ref')} submitted={st.get('submitted_utc')}")
    else:
        print("OK no submit state yet")

    cron = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    cron_txt = cron.stdout or ""
    for mark in ("nvarc leftover_unc auto-submit", "nvarc leftover_unc submission watch"):
        if mark in cron_txt:
            print(f"OK cron: {mark}")
        else:
            print(f"WARN cron missing: {mark}")

    if fails:
        print("SMOKE FAIL:")
        for f in fails:
            print(" -", f)
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
