#!/usr/bin/env python3
"""Poll one Kaggle competition submission until it leaves PENDING.

Records wall hours from the submission `date` to first COMPLETE/ERROR sighting.
Logs are not available on hidden reruns; this is the duration proxy.

Does not submit. Cron `--once` every 30 minutes. No-ops until the matching
submit script has written a ref, and after a terminal state.
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

COMP = os.environ.get("NVARC_COMP", "arc-prize-2026-arc-agi-2")
STATE = Path(os.environ.get("NVARC_WATCH_STATE", "/opt/work/nvarc/single8x6_watch.json"))
SUBMIT_STATE = Path(os.environ.get(
    "NVARC_SINGLE_STATE", "/opt/work/nvarc/single8x6_submit_state.json"
))
LOCK = Path(os.environ.get("NVARC_WATCH_LOCK", "/tmp/nvarc_single8x6_watch.lock"))


def log(msg: str) -> None:
    print(
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}",
        flush=True,
    )


def find_kaggle(explicit: str = "") -> str:
    candidates = [explicit, os.environ.get("NVARC_KAGGLE", ""),
                  "/opt/venv_kaggle/bin/kaggle"]
    which = shutil.which("kaggle")
    if which:
        candidates.append(which)
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise SystemExit("FAIL: kaggle CLI not found")


def parse_dt(s: str) -> datetime | None:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def list_rows(kaggle: str) -> list[dict]:
    p = subprocess.run(
        [kaggle, "competitions", "submissions", "-c", COMP, "--csv"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        log("submissions list failed: " + (p.stderr or p.stdout)[-400:])
        return []
    return list(csv.DictReader(io.StringIO(p.stdout)))


def status_name(raw: str) -> str:
    s = (raw or "").upper()
    if "PENDING" in s:
        return "PENDING"
    if "COMPLETE" in s:
        return "COMPLETE"
    if "ERROR" in s or "FAIL" in s:
        return "ERROR"
    return (raw or "UNKNOWN").strip() or "UNKNOWN"


def wanted_ref(cli_ref: str) -> str:
    if cli_ref:
        return str(cli_ref).strip()
    env = os.environ.get("NVARC_WATCH_REF", "").strip()
    if env:
        return env
    if SUBMIT_STATE.exists():
        try:
            st = json.loads(SUBMIT_STATE.read_text())
        except Exception:
            st = {}
        for key in ("ref", "submission_ref"):
            v = str(st.get(key) or "").strip()
            if v:
                return v
        runs = st.get("runs") or {}
        if isinstance(runs, dict):
            for rec in runs.values():
                v = str((rec or {}).get("ref") or "").strip()
                if v:
                    return v
    return ""


def pick_row(rows: list[dict], ref: str) -> dict | None:
    if ref:
        for r in rows:
            if str(r.get("ref") or "").strip() == str(ref):
                return r
        return None
    for r in rows:
        if status_name(r.get("status") or "") == "PENDING":
            return r
    return None


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=2) + "\n")
    tmp.replace(STATE)


def acquire_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = LOCK.open("a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def snapshot(row: dict, now: datetime) -> dict:
    submitted = parse_dt(row.get("date") or "")
    elapsed_h = None
    if submitted:
        elapsed_h = round((now - submitted).total_seconds() / 3600.0, 3)
    return {
        "ref": str(row.get("ref") or "").strip(),
        "status": status_name(row.get("status") or ""),
        "status_raw": row.get("status") or "",
        "publicScore": (row.get("publicScore") or "").strip(),
        "privateScore": (row.get("privateScore") or "").strip(),
        "description": (row.get("description") or "").strip(),
        "submitted_utc": row.get("date") or "",
        "observed_utc": now.isoformat(),
        "elapsed_h": elapsed_h,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--kaggle", default="")
    ap.add_argument("--ref", default="")
    args = ap.parse_args()

    lock = acquire_lock()
    if lock is None:
        log("another watch poll holds the lock; skip")
        return 0
    try:
        st = load_state()
        if st.get("terminal"):
            log(
                f"already terminal ref={st.get('ref')} status={st.get('status')} "
                f"elapsed_h={st.get('elapsed_h')} publicScore={st.get('publicScore')}"
            )
            return 0
        ref = wanted_ref(args.ref)
        if not ref:
            log("waiting for single8x6 submit ref; skip")
            return 0
        kaggle = find_kaggle(args.kaggle)
        rows = list_rows(kaggle)
        row = pick_row(rows, ref)
        if not row:
            log(f"submission ref={ref} not in list yet")
            return 0
        now = datetime.now(timezone.utc)
        snap = snapshot(row, now)
        if not st.get("first_seen_utc"):
            snap["first_seen_utc"] = now.isoformat()
        else:
            snap["first_seen_utc"] = st["first_seen_utc"]
        log(
            f"ref={snap['ref']} status={snap['status']} elapsed_h={snap['elapsed_h']} "
            f"publicScore={snap['publicScore'] or '-'}"
        )
        if snap["status"] in ("COMPLETE", "ERROR"):
            snap["terminal"] = True
            snap["finished_utc"] = now.isoformat()
            log(
                f"DONE ref={snap['ref']} wall={snap['elapsed_h']}h "
                f"score={snap['publicScore'] or snap['status']}"
            )
        save_state(snap)
        return 0
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
