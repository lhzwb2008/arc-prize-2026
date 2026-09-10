#!/usr/bin/env python3
"""Print a one-screen summary of ttt_qwen.py result jsons."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def one(path: Path) -> None:
    d = json.loads(path.read_text())
    solved, total = d.get("solved"), d.get("total")
    acc = d.get("acc")
    hits, inputs = d.get("hits"), d.get("inputs")
    print(f"== {path} ==")
    print(f"  tasks {solved}/{total}  acc={None if acc is None else round(100*acc, 1)}%")
    if hits is not None:
        ia = d.get("input_acc")
        print(f"  inputs {hits}/{inputs}  input_acc={None if ia is None else round(100*ia, 1)}%")
    print(f"  seconds={d.get('seconds')}")
    groups = d.get("groups") or {}
    for name, g in sorted(groups.items()):
        print(f"    {name:20s} tasks {g.get('solved',0)}/{g.get('tasks',0)}  "
              f"inputs {g.get('hits',0)}/{g.get('inputs',0)}")
    oks = [r.get("id") for r in d.get("results") or [] if r.get("ok")]
    if oks:
        print("  solved:", ", ".join(oks[:24]))
    print()


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print("usage: summarize_runs.py file.json ...", file=sys.stderr)
        return 2
    for p in paths:
        if p.exists():
            one(p)
        else:
            print(f"missing {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
