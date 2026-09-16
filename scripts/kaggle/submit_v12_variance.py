#!/usr/bin/env python3
"""Competition-submit a Kaggle kernel version after a UTC-day window.

Cron `--once` every 10 minutes. No-ops outside the window or after success.
Does not push a Save Version; `kaggle competitions submit` starts the 12h hidden rerun.

Lock files: cron `flock -n` must NOT be the same path as this script's fcntl lock.
Same-path flock + fcntl deadlocks on Linux and the poll never submits.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMP = "arc-prize-2026-arc-agi-2"
KERNEL = os.environ.get(
    "NVARC_KERNEL", "wenbozhang2026/nvarc-qwen3-4b-ttt-2026-ceiling"
)
VERSION = int(os.environ.get("NVARC_KERNEL_VERSION", "13"))
STATE = Path(os.environ.get("NVARC_V13_STATE", "/opt/work/nvarc/v13_submit_state.json"))
LOG = Path(os.environ.get("NVARC_V13_LOG", "/opt/work/nvarc/v13_submit.log"))
# Must NOT be the same path as the cron `flock -n` file
# (`/opt/work/nvarc/v13_submit.cron.lock`). Linux flock is per file-description:
# inheriting the cron lock then opening a second fd on that inode makes LOCK_NB
# fail forever ("another poll holds the lock") and never submit.
LOCK = Path(os.environ.get("NVARC_V13_LOCK", "/tmp/nvarc_v13_submit.lock"))

RUNS = [
    {
        "id": "v13",
        "not_before_utc": "2026-09-17T00:02:00+00:00",
        "message": (
            "NVARC 8x6 v13: leftover-B, end-of-pass mean_quality pool, "
            "no keep-primary, catchup then B, queue.get"
        ),
    },
]


def log(msg: str) -> None:
    # Print only. Cron jobs append stdout to LOG (`>> log 2>&1`). Do not also
    # write LOG here: that duplicated every line and looked like two crons.
    print(
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}",
        flush=True,
    )


def find_kaggle(explicit: str = "") -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("NVARC_KAGGLE")
    if env:
        candidates.append(env)
    candidates.extend(
        [
            "/opt/venv_kaggle/bin/kaggle",
            str(ROOT / ".venv/bin/kaggle"),
        ]
    )
    which = shutil.which("kaggle")
    if which:
        candidates.append(which)
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise SystemExit("FAIL: kaggle CLI not found")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"runs": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2) + "\n")


def kernel_ready(kaggle: str) -> str:
    p = subprocess.run([kaggle, "kernels", "status", KERNEL], capture_output=True, text=True)
    out = ((p.stdout or "") + (p.stderr or "")).upper()
    if "RUNNING" in out:
        return "RUNNING"
    if "CANCEL" in out:
        return "ERROR"
    if "COMPLETE" in out:
        return "COMPLETE"
    if "ERROR" in out:
        return "ERROR"
    return "UNKNOWN"


def last_submit_utc_date(kaggle: str) -> str | None:
    p = subprocess.run(
        [kaggle, "competitions", "submissions", "-c", COMP, "--csv"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        log("submissions list failed: " + (p.stderr or p.stdout)[-400:])
        return None
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    if not rows:
        return None
    date = (rows[0].get("date") or "").strip()
    return date[:10] if date else None


def acquire_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = LOCK.open("a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def submit_run(kaggle: str, run: dict) -> int:
    stt = kernel_ready(kaggle)
    log(f"kernel {KERNEL} v{VERSION} status={stt}")
    if stt != "COMPLETE":
        log(f"run {run['id']}: kernel not COMPLETE; skip")
        return 0
    d = last_submit_utc_date(kaggle)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log(f"run {run['id']}: last_submit_date={d} today_utc={today}")
    if d is None:
        log("could not read submissions; retry next cron")
        return 0
    if d == today:
        log("this UTC day already has a submission; wait for next window")
        return 0
    msg = run["message"]
    log(f"submitting {KERNEL} -v {VERSION}  ({run['id']})")
    p = subprocess.run(
        [
            kaggle, "competitions", "submit", "-c", COMP,
            "-k", KERNEL, "-v", str(VERSION),
            "-f", "submission.json",
            "-m", msg,
        ],
        capture_output=True, text=True,
    )
    log("stdout: " + (p.stdout or "")[-1500:])
    log("stderr: " + (p.stderr or "")[-1500:])
    if p.returncode != 0:
        log(f"FAIL rc={p.returncode}")
        return p.returncode
    st = load_state()
    st.setdefault("runs", {})[run["id"]] = {
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "message": msg,
        "version": VERSION,
        "last_submit_date_before": d,
    }
    save_state(st)
    log(f"OK submitted run {run['id']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--kaggle", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lock = acquire_lock()
    if lock is None:
        log("another submit poll holds the lock; skip")
        return 0
    try:
        kaggle = find_kaggle(args.kaggle)
        st = load_state()
        now = datetime.now(timezone.utc)
        pending = []
        for run in RUNS:
            if st.get("runs", {}).get(run["id"], {}).get("submitted_utc"):
                continue
            start = datetime.fromisoformat(run["not_before_utc"])
            if now < start:
                log(f"run {run['id']}: too early; window {start.isoformat()}")
                continue
            pending.append(run)
        if not pending:
            if all(st.get("runs", {}).get(r["id"], {}).get("submitted_utc") for r in RUNS):
                log("scheduled submit already done; nothing to do")
            return 0
        run = pending[0]
        if args.dry_run:
            log(f"dry-run would submit run {run['id']} now")
            return 0
        return submit_run(kaggle, run)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
