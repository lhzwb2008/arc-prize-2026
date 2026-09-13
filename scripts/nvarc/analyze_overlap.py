#!/usr/bin/env python3
"""Per-task overlap among local NVARC single-pass reports. GPU box paths."""
from __future__ import annotations

import json
import os
from pathlib import Path

WORK = Path(os.getenv("NVARC_WORK", "/opt/work/nvarc"))
DATA = os.getenv("NVARC_DATA", "/opt/data/kaggle/arc-agi_evaluation_challenges.json")
OUT = WORK / "eval120_search" / "overlap.json"


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("per_task") or {}


def val(pt, k):
    return float((pt or {}).get(k, 0) or 0)


def solved(pt, thresh=1.0):
    return {k for k, v in (pt or {}).items() if float(v or 0) >= thresh}


def main():
    keys = list(json.loads(Path(DATA).read_text()))
    runs = {
        "t3A_5x5": load(WORK / "eval120_t3_a/report.json"),
        "t3B_5x5": load(WORK / "eval120_t3_b/report.json"),
        "n5_5x8": load(WORK / "eval120_search/n5_g8/report.json"),
        "n6_6x8": load(WORK / "eval120_search/n6_g8/report.json"),
        "n7_7x8": load(WORK / "eval120_search/n7_g8/report.json"),
        "halfA_8x8": load(WORK / "eval120_half_a/report.json"),
        "halfB_8x8": load(WORK / "eval120_half_b/report.json"),
        "full_16x16": load(WORK / "eval120/report.json"),
        "n6_g6": load(WORK / "eval120_search/ablate_n6/geos6.json"),
        "halfA_g6": load(WORK / "eval120_search/ablate_half_a/geos6.json"),
    }
    summary = {"scores": {}, "full_solves": {}, "unique_full": {}, "notes": []}
    present = {n: pt for n, pt in runs.items() if pt}
    for name, pt in present.items():
        F = [k for k in keys if val(pt, k) >= 1]
        summary["scores"][name] = {
            "score": sum(val(pt, k) for k in keys),
            "n_full": len(F),
            "n_any": sum(1 for k in keys if val(pt, k) > 0),
        }
        summary["full_solves"][name] = F

    mains = [n for n in ("t3A_5x5", "n5_5x8", "n6_6x8", "n7_7x8", "halfA_8x8", "halfB_8x8", "full_16x16") if n in present]
    fulls = {n: set(summary["full_solves"][n]) for n in mains}
    for n in mains:
        others = set().union(*[fulls[m] for m in mains if m != n]) if len(mains) > 1 else set()
        summary["unique_full"][n] = sorted(fulls[n] - others)

    pairs = {}
    for i, a in enumerate(mains):
        for b in mains[i + 1 :]:
            A, B = fulls[a], fulls[b]
            pairs["%s__%s" % (a, b)] = {
                "a_only": sorted(A - B),
                "b_only": sorted(B - A),
                "both": len(A & B),
                "union": len(A | B),
                "net_score": sum(val(present[a], k) - val(present[b], k) for k in keys),
            }
    summary["pairs"] = pairs
    U = set().union(*fulls.values()) if fulls else set()
    summary["union_full"] = sorted(U)
    summary["union_n"] = len(U)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    print("==== scores ====")
    for n, s in summary["scores"].items():
        print("%-16s %6.3f  full=%d any=%d" % (n, s["score"], s["n_full"], s["n_any"]))
    print("==== unique full ====")
    for n, ids in summary["unique_full"].items():
        print("%-16s %d %s" % (n, len(ids), ids))
    print("union_full", summary["union_n"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
