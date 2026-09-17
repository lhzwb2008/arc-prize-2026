#!/usr/bin/env python3
"""Competition-submit the single-pass 8×6 kernel after 08:00 CST.

Cron `--once` every 10 minutes. No-ops before the window, if the kernel Save
Version is still RUNNING, or after a successful submit. Hidden rerun starts
here; duration is observed by watch_submission.py.

Lock files: cron `flock -n` must NOT be the same path as this script's fcntl lock.
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

COMP = "arc-prize-2026-arc-agi-2"
KERNEL = os.environ.get(
    "NVARC_KERNEL", "wenbozhang2026/nvarc-qwen3-4b-ttt-2026-ceiling"
)
KERNEL_INFO = Path(os.environ.get(
    "NVARC_SINGLE_KERNEL", "/opt/work/nvarc/single8x6_kernel.json"
))
STATE = Path(os.environ.get(
    "NVARC_SINGLE_STATE", "/opt/work/nvarc/single8x6_submit_state.json"
))
LOCK = Path(os.environ.get("NVARC_SINGLE_LOCK", "/tmp/nvarc_single8x6_submit.lock"))
RUN_ID = "single8x6"
NOT_BEFORE = os.environ.get("NVARC_SINGLE_NOT_BEFORE", "2026-09-18T00:00:00+00:00")
MESSAGE = os.environ.get(
    "NVARC_SINGLE_MESSAGE",
    "NVARC 8x6 SINGLE pass A-only timing probe (no leftover-B)",
)


def log(msg: str) -> None:
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
    candidates.extend(["/opt/venv_kaggle/bin/kaggle"])
    which = shutil.which("kaggle")
    if which:
        candidates.append(which)
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise SystemExit("FAIL: kaggle CLI not found")


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def kernel_version() -> int:
    env = os.environ.get("NVARC_KERNEL_VERSION", "").strip()
    if env:
        return int(env)
    info = load_json(KERNEL_INFO, {})
    v = info.get("version_number") or info.get("version")
    if v is None:
        raise SystemExit(f"FAIL: no kernel version in {KERNEL_INFO} or NVARC_KERNEL_VERSION")
    return int(v)


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


def list_rows(kaggle: str) -> list[dict]:
    p = subprocess.run(
        [kaggle, "competitions", "submissions", "-c", COMP, "--csv"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        log("submissions list failed: " + (p.stderr or p.stdout)[-400:])
        return []
    return list(csv.DictReader(io.StringIO(p.stdout)))


def last_submit_utc_date(rows: list[dict]) -> str | None:
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


def submit_now(kaggle: str, version: int, dry_run: bool) -> int:
    stt = kernel_ready(kaggle)
    log(f"kernel {KERNEL} v{version} status={stt}")
    if stt != "COMPLETE":
        log("kernel not COMPLETE; skip")
        return 0
    rows = list_rows(kaggle)
    d = last_submit_utc_date(rows)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log(f"last_submit_date={d} today_utc={today}")
    if d is None and rows:
        log("could not parse last submit date; retry next cron")
        return 0
    if d == today:
        log("this UTC day already has a submission; stop")
        return 0
    if dry_run:
        log(f"dry-run would submit {KERNEL} -v {version}")
        return 0
    log(f"submitting {KERNEL} -v {version}")
    p = subprocess.run(
        [
            kaggle, "competitions", "submit", "-c", COMP,
            "-k", KERNEL, "-v", str(version),
            "-f", "submission.json",
            "-m", MESSAGE,
        ],
        capture_output=True, text=True,
    )
    log("stdout: " + (p.stdout or "")[-1500:])
    log("stderr: " + (p.stderr or "")[-1500:])
    if p.returncode != 0:
        log(f"FAIL rc={p.returncode}")
        return p.returncode
    rows2 = list_rows(kaggle)
    ref = str((rows2[0] or {}).get("ref") or "").strip() if rows2 else ""
    rec = {
        "id": RUN_ID,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "message": MESSAGE,
        "version": version,
        "ref": ref,
        "last_submit_date_before": d,
        "kernel": KERNEL,
    }
    st = load_json(STATE, {"runs": {}})
    st.setdefault("runs", {})[RUN_ID] = rec
    st["ref"] = ref
    st["version"] = version
    st["submitted_utc"] = rec["submitted_utc"]
    save_json(STATE, st)
    log(f"OK submitted ref={ref} version={version}")
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
        st = load_json(STATE, {"runs": {}})
        if (st.get("runs") or {}).get(RUN_ID, {}).get("submitted_utc") or st.get("submitted_utc"):
            log("scheduled single8x6 submit already done; nothing to do")
            return 0
        now = datetime.now(timezone.utc)
        start = datetime.fromisoformat(NOT_BEFORE)
        if now < start:
            log(f"too early; window {start.isoformat()} (08:00 CST)")
            return 0
        kaggle = find_kaggle(args.kaggle)
        version = kernel_version()
        return submit_now(kaggle, version, args.dry_run)
    except SystemExit as e:
        log(str(e))
        return 1
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
