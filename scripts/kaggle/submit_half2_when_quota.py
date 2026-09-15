#!/usr/bin/env python3
"""Submit a pending Kaggle kernel version after the UTC daily quota refreshes.

Designed to run on the always-on GPU box via cron (`--once` every 10 minutes).
Does not long-sleep: if it is too early, the quota is still used, or the kernel
is still RUNNING, it exits 0 and waits for the next cron tick.

State file: notebooks/nvarc_2026/.pending_submit.json
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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = Path(os.environ.get(
    "NVARC_SUBMIT_STATE",
    str(ROOT / "notebooks/nvarc_2026/.pending_submit.json"),
))
LOG = Path(os.environ.get("NVARC_SUBMIT_LOG", str(STATE.with_suffix(".log"))))
DONE = Path(os.environ.get("NVARC_SUBMIT_DONE", str(STATE.with_suffix(".done.json"))))
LOCK = Path(os.environ.get("NVARC_SUBMIT_LOCK", "/tmp/nvarc_kaggle_submit.lock"))
COMP = "arc-prize-2026-arc-agi-2"
KERNEL = "wenbozhang2026/nvarc-qwen3-4b-ttt-2026-ceiling"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def find_kaggle(explicit: str = "") -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("NVARC_KAGGLE")
    if env:
        candidates.append(env)
    candidates.extend(
        [
            str(ROOT / ".venv/bin/kaggle"),
            "/opt/venv_kaggle/bin/kaggle",
            "/opt/venv_nvarc/bin/kaggle",
        ]
    )
    which = shutil.which("kaggle")
    if which:
        candidates.append(which)
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise SystemExit("FAIL: kaggle CLI not found (install /opt/venv_kaggle)")


def kernel_ready(kaggle: str, kernel: str) -> str:
    p = subprocess.run([kaggle, "kernels", "status", kernel], capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    up = out.upper()
    if "RUNNING" in up:
        return "RUNNING"
    if "CANCEL" in up:
        return "ERROR"
    if "COMPLETE" in up:
        return "COMPLETE"
    if "ERROR" in up:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="single poll for cron; exit instead of sleeping")
    ap.add_argument("--kaggle", default="")
    args = ap.parse_args()
    once = args.once

    lock = acquire_lock()
    if lock is None:
        log("another submit poll holds the lock; skip")
        return 0

    try:
        return _run(once, args.kaggle)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _run(once: bool, kaggle_bin: str) -> int:
    kaggle = find_kaggle(kaggle_bin)
    if DONE.exists():
        log(f"already submitted ({DONE.name}); nothing to do")
        return 0
    if not STATE.exists():
        log(f"FAIL: missing {STATE}")
        return 1

    st = json.loads(STATE.read_text())
    version = int(st["version"])
    not_before = datetime.fromisoformat(st["not_before_utc"].replace("Z", "+00:00"))
    used = st.get("used_utc_date") or ""
    kernel = st.get("kernel") or KERNEL
    msg = st.get("message") or "NVARC v12 auto-submit"

    now = datetime.now(timezone.utc)
    if now < not_before:
        if once:
            log(f"too early for v{version}; next window {not_before.isoformat()}")
            return 0
        wait = (not_before - now).total_seconds()
        log(f"sleeping {wait/3600:.2f}h until {not_before.isoformat()} then submit kernel v{version}")
        time.sleep(max(0, wait) + 30)

    def quota_ok() -> tuple[str | None, str]:
        d = last_submit_utc_date(kaggle)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log(f"last_submit_date={d} today_utc={today} used={used}")
        return d, today

    for _ in range(1 if once else 36):
        d, today = quota_ok()
        if d is None:
            if once:
                log("could not read submissions; retry next cron")
                return 0
            log("could not read submissions; retry in 5 min")
            time.sleep(300)
            continue
        # Daily quota is one submit per UTC calendar day. After midnight the
        # last row is still yesterday (used) — that means the slot is free.
        if d == today:
            log("this UTC day already has a submission; marking done")
            DONE.write_text(json.dumps({**st, "submitted_utc": datetime.now(timezone.utc).isoformat(),
                                        "note": "detected existing same-day submission"}, indent=2) + "\n")
            return 0
        break
    else:
        log("FAIL: timed out waiting for a new UTC day")
        return 1

    for _ in range(1 if once else 192):
        stt = kernel_ready(kaggle, kernel)
        log(f"kernel {kernel} v{version} status={stt}")
        if stt == "COMPLETE":
            break
        if stt == "ERROR":
            log("FAIL: kernel did not complete; not submitting")
            return 1
        if once:
            log("kernel not COMPLETE; retry next cron")
            return 0
        time.sleep(300)
    else:
        log("FAIL: timed out waiting for kernel COMPLETE")
        return 1

    log(f"submitting {kernel} -v {version}")
    p = subprocess.run(
        [
            kaggle, "competitions", "submit", "-c", COMP,
            "-k", kernel, "-v", str(version),
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
    log("OK submitted")
    DONE.write_text(json.dumps({**st, "submitted_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
