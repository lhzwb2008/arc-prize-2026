#!/usr/bin/env python3
"""Wait until the next Kaggle UTC day, then submit the pending kernel version.

State file: notebooks/nvarc_2026/.pending_submit.json
{
  "version": 7,
  "not_before_utc": "2026-09-13T00:05:00+00:00",
  "used_utc_date": "2026-09-12",
  "message": "..."
}
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "notebooks/nvarc_2026/.pending_submit.json"
LOG = ROOT / "notebooks/nvarc_2026/.pending_submit.log"
KAGGLE = ROOT / ".venv/bin/kaggle"
COMP = "arc-prize-2026-arc-agi-2"
KERNEL = "wenbozhang2026/nvarc-qwen3-4b-ttt-2026-ceiling"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def last_submit_utc_date(kaggle: str) -> str | None:
    p = subprocess.run(
        [kaggle, "competitions", "submissions", "-c", COMP, "--csv"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        log("submissions list failed: " + (p.stderr or p.stdout)[-400:])
        return None
    lines = [ln for ln in p.stdout.splitlines() if ln.strip() and not ln.startswith("ref,")]
    if not lines:
        return None
    parts = lines[0].split(",")
    if len(parts) < 3:
        return None
    return parts[2].strip()[:10]


def main() -> int:
    kaggle = sys.argv[1] if len(sys.argv) > 1 else str(KAGGLE)
    if not STATE.exists():
        log(f"FAIL: missing {STATE}")
        return 1
    st = json.loads(STATE.read_text())
    version = int(st["version"])
    not_before = datetime.fromisoformat(st["not_before_utc"].replace("Z", "+00:00"))
    used = st.get("used_utc_date") or ""
    msg = st.get("message") or (
        "NVARC half-cost: A checkpoint + keep-primary + live submission.json (cheap-first both passes)"
    )

    now = datetime.now(timezone.utc)
    if now < not_before:
        wait = (not_before - now).total_seconds()
        log(f"sleeping {wait/3600:.2f}h until {not_before.isoformat()} then submit kernel v{version}")
        time.sleep(max(0, wait) + 30)

    for _ in range(36):
        d = last_submit_utc_date(kaggle)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log(f"last_submit_date={d} today_utc={today} used={used}")
        if d is None:
            log("could not read submissions; retry in 5 min")
            time.sleep(300)
            continue
        if used and d == used:
            log("quota still the UTC day that burned the slot; sleep 10 min")
            time.sleep(600)
            continue
        break
    else:
        log("FAIL: timed out waiting for a new UTC day")
        return 1

    log(f"submitting {KERNEL} -v {version}")
    p = subprocess.run(
        [
            kaggle, "competitions", "submit", "-c", COMP,
            "-k", KERNEL, "-v", str(version),
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
    done = STATE.with_suffix(".done.json")
    done.write_text(json.dumps({**st, "submitted_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
