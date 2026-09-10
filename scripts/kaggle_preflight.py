#!/usr/bin/env python3
"""Fail unless the latest NVARC kernel Save Version actually ran TTT on L4 + Py3.11."""
from __future__ import annotations

import json
import re
import subprocess
import sys

KERNEL = "wenbozhang2026/nvarc-qwen3-4b-ttt-2026-ceiling"
KAGGLE = sys.argv[1] if len(sys.argv) > 1 else "kaggle"


def logs_text() -> str:
    raw = subprocess.check_output([KAGGLE, "kernels", "logs", KERNEL], text=True, stderr=subprocess.DEVNULL)
    try:
        log = json.loads(raw)
    except json.JSONDecodeError:
        log = [json.loads(l.strip(", \n")) for l in raw.splitlines() if l.strip(", \n").startswith("{")]
    return "".join(e.get("data", "") for e in log)


def main() -> int:
    txt = logs_text()
    fails = []
    if "python3.11" not in txt and "python=3.11" not in txt:
        # GATE line uses sys.version like 3.11.13
        if not re.search(r"GATE starter python=3\.11", txt):
            fails.append("Python is not 3.11 (GATE starter python=3.11 missing)")
    if re.search(r"python=3\.12", txt) or "/usr/local/lib/python3.12/" in txt:
        fails.append("Python 3.12 appeared in logs")
    if "Tesla P100" in txt or "sm_60" in txt:
        fails.append("GPU is P100/sm_60")
    if "skip TTT" in txt:
        fails.append("TTT was skipped")
    if "starter_exit=1" in txt:
        fails.append("starter_exit=1")
    if "GATE FAIL" in txt:
        fails.append("GATE FAIL in logs")
    if "decoded=0" in txt or "decoded 0" in txt:
        fails.append("decoded=0")
    if "ImportError" in txt or "Failed to load PyTorch" in txt:
        fails.append("torch import failed")
    gate = re.search(r"GATE starter .+", txt)
    last = re.search(r"GATE last .+", txt)
    print(gate.group(0) if gate else "NO GATE starter line")
    print(last.group(0) if last else "NO GATE last line")
    if "Reload score" in txt:
        m = re.search(r"Reload score:.*", txt)
        print(m.group(0) if m else "Reload score present")
    else:
        print("no Reload score (ok on hidden rerun; required on Save Version smoke)")
    if fails:
        print("PREFLIGHT FAIL:")
        for f in fails:
            print(" -", f)
        return 1
    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
