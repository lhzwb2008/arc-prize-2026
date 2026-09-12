"""1-D peak hunt around half-cost n_train=8 on the public 120.

Reads reports written by run_single_search.sh and prints the next n_train
values to run (or STOP). Improvement bar is +1.0 reload score vs half-A.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WORK = Path(os.getenv("NVARC_WORK", "/opt/work/nvarc"))
SEARCH = WORK / "eval120_search"
HALF = WORK / "eval120_half_a" / "report.json"
FULL = WORK / "eval120" / "report.json"
T3A = WORK / "eval120_t3_a" / "report.json"
MARGIN = 1.0


def load_score(path: Path):
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["score"])


def run_score(n_train: int):
    return load_score(SEARCH / f"n{n_train}_g8" / "report.json")


def main():
    half = load_score(HALF)
    full = load_score(FULL)
    t3a = load_score(T3A)
    table = {
        "baselines": {
            "t3a_n5_g5": t3a,
            "halfA_n8_g8": half,
            "full_n16_g8_c2": full,
        },
        "sweeps": {},
    }
    for n in (4, 5, 6, 10, 12, 16):
        s = run_score(n)
        if s is not None:
            table["sweeps"][f"n{n}_g8"] = s

    s6 = table["sweeps"].get("n6_g8")
    s10 = table["sweeps"].get("n10_g8")
    s4 = table["sweeps"].get("n4_g8")
    s12 = table["sweeps"].get("n12_g8")
    s5 = table["sweeps"].get("n5_g8")
    s16 = table["sweeps"].get("n16_g8")

    next_runs: list[str] = []
    reason = []
    if half is None:
        reason.append("missing half-A baseline")
        next_runs = ["STOP"]
    elif s6 is None or s10 is None:
        reason.append("need n6 and n10 first")
        if s6 is None:
            next_runs.append("6")
        if s10 is None:
            next_runs.append("10")
    else:
        best_left = max(x for x in (s6, half) if x is not None)
        best_right = max(x for x in (s10, half) if x is not None)
        if s6 < half + MARGIN and s10 < half + MARGIN:
            reason.append(
                f"neither neighbor beats half-A {half:.2f} by +{MARGIN:.0f}; "
                f"n6={s6:.2f} n10={s10:.2f} → local max at n=8"
            )
            next_runs = ["STOP"]
        elif s10 >= s6 and s10 >= half:
            reason.append(f"peak to the right (n10={s10:.2f} ≥ n6={s6:.2f}, half={half:.2f})")
            if s12 is None:
                next_runs.append("12")
            elif s12 >= s10 and s16 is None:
                next_runs.append("16")
            else:
                next_runs = ["STOP"]
                reason.append("right side already filled")
        else:
            reason.append(f"peak to the left (n6={s6:.2f} > n10={s10:.2f}, half={half:.2f})")
            if s4 is None:
                next_runs.append("4")
            if s5 is None:
                next_runs.append("5")
            if next_runs == []:
                next_runs = ["STOP"]
                reason.append("left side already filled")

    table["next"] = next_runs
    table["reason"] = reason
    out = SEARCH / "leaderboard.json"
    SEARCH.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table, indent=2))
    print("NEXT " + " ".join(next_runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
