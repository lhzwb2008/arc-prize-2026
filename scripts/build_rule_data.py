#!/usr/bin/env python3
"""Collect natural-language rules for ARC training tasks -> data/rules/task_rules.json.

Sources (both cover the 400 ARC-AGI-1 training tasks, which are inside the
1000-task ARC-AGI-2 training set):
  * BARC seeds  (github.com/xu3kev/BARC, seeds/<task_id>.py): "# concepts:" and
    "# description:" comment headers written by the BARC authors.
  * LARC        (github.com/samacqua/LARC, dataset/summary/*.csv): human
    descriptions; we keep only verified ones that at least one other human could
    build the correct output from.

Output: {task_id: [{"src": "barc_seed"|"larc", "concepts": [...], "rule": "..."}]}

Usage:
    python scripts/build_rule_data.py --barc /tmp/arc-ext/BARC --larc /tmp/arc-ext/LARC
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_text import clean_larc, larc_usable, parse_barc_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def from_barc_seeds(barc_dir: Path):
    out = defaultdict(list)
    for fp in sorted((barc_dir / "seeds").glob("*.py")):
        tid = fp.stem
        if len(tid) != 8:
            continue
        concepts, desc = parse_barc_source(fp.read_text(errors="ignore"))
        if desc:
            out[tid].append({"src": "barc_seed", "concepts": concepts, "rule": desc})
    return out


def from_larc(larc_dir: Path):
    s = larc_dir / "dataset" / "summary"
    tasks = {r["task_id"]: r["task_name"].replace(".json", "") for r in csv.DictReader(open(s / "task.csv"))}
    desc = {r["description_id"]: r for r in csv.DictReader(open(s / "description.csv"))}
    build = {r["build_id"]: r for r in csv.DictReader(open(s / "build.csv"))}
    join = list(csv.DictReader(open(s / "join.csv")))
    built_ok = defaultdict(int)
    for j in join:
        b = build.get(j["build_id"])
        if b and b["is_success"].lower() == "true":
            built_ok[j["description_id"]] += 1
    out = defaultdict(list)
    seen = set()
    for j in join:
        did = j["description_id"]
        d = desc.get(did)
        if not d or did in seen:
            continue
        if d["is_verified"].lower() != "true" or built_ok[did] == 0:
            continue
        rule = clean_larc(d["description_output"])
        if not larc_usable(rule):
            continue
        seen.add(did)
        size = clean_larc(d.get("description_output_grid_size", ""))
        if (size and 8 < len(size) < 80 and "same" not in size.lower()
                and size.lower() not in rule.lower()):
            rule = f"{rule} Output size: {size}"
        out[tasks[j["task_id"]]].append({"src": "larc", "concepts": [], "rule": rule})
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--barc", default="/tmp/arc-ext/BARC")
    p.add_argument("--larc", default="/tmp/arc-ext/LARC")
    p.add_argument("--train-dir", default=str(ROOT / "data" / "full" / "data" / "training"))
    p.add_argument("--out", default=str(ROOT / "data" / "rules" / "task_rules.json"))
    args = p.parse_args()

    rules = defaultdict(list)
    n_seed = n_larc = 0
    if Path(args.barc).exists():
        for tid, lst in from_barc_seeds(Path(args.barc)).items():
            rules[tid].extend(lst)
            n_seed += len(lst)
    if Path(args.larc).exists():
        for tid, lst in from_larc(Path(args.larc)).items():
            rules[tid].extend(lst)
            n_larc += len(lst)
    train_ids = {fp.stem for fp in Path(args.train_dir).glob("*.json")}
    covered = sorted(t for t in rules if t in train_ids)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({t: rules[t] for t in sorted(rules)}, indent=1, ensure_ascii=False))
    print(f"barc_seed rules={n_seed} larc rules={n_larc} tasks={len(rules)} "
          f"in ARC-AGI-2 training={len(covered)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
