#!/usr/bin/env python3
"""Embed src/arc_solver/solver.py into a Kaggle-ready notebook (no imports from the repo)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER = (ROOT / "src" / "arc_solver" / "solver.py").read_text()

NOTEBOOK_DRIVER = r'''
import json
import os
import time
from pathlib import Path

TIME_LIMIT_PER_TASK = float(os.environ.get("ARC_TIME_LIMIT", "30"))

# Kaggle mounts the competition data under /kaggle/input/<competition>/.
# ARC_INPUT_DIR lets the same notebook run locally against data/kaggle/.
input_dir = Path(os.environ.get("ARC_INPUT_DIR", "/kaggle/input"))
candidates = []
if input_dir.exists():
    # Kaggle has used both /kaggle/input/<slug>/ and /kaggle/input/competitions/<slug>/; search recursively.
    candidates.extend(p for p in input_dir.rglob("*.json") if "test" in p.name and "challenges" in p.name)
repo_examples = Path.cwd() / "data" / "examples"


def load_challenges():
    for path in sorted(candidates):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data and "train" not in data:
            print("using", path, "tasks", len(data))
            return data
    tasks = {}
    if repo_examples.exists():
        for file in sorted(repo_examples.glob("*.json")):
            tasks[file.stem] = json.loads(file.read_text())
        print("using local examples", repo_examples, "n=", len(tasks))
        return tasks
    if input_dir.exists():
        for p in sorted(input_dir.rglob("*"))[:200]:
            print("  ", p)
    raise FileNotFoundError("no ARC challenges file found")


challenges = load_challenges()
t0 = time.time()
submission = {}
for n, (task_id, task) in enumerate(challenges.items(), start=1):
    submission[task_id] = predict_task(task, time_limit=TIME_LIMIT_PER_TASK)
    if n % 20 == 0 or n == len(challenges):
        print(f"{n}/{len(challenges)} tasks, {time.time() - t0:.0f}s", flush=True)

# Every task id must be present with both attempts for every test input.
for task_id, task in challenges.items():
    preds = submission.setdefault(task_id, [])
    while len(preds) < len(task["test"]):
        g = task["test"][len(preds)]["input"]
        preds.append({"attempt_1": [row[:] for row in g], "attempt_2": [row[:] for row in g]})
    for k, p in enumerate(preds):
        for key in ("attempt_1", "attempt_2"):
            if not valid(p.get(key)):
                p[key] = [row[:] for row in task["test"][k]["input"]]

out = Path(os.environ.get("ARC_OUTPUT", "/kaggle/working/submission.json"))
if not out.parent.exists():
    out = Path("demo/output/kaggle_submission.json")
    out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(submission, separators=(",", ":")))
print("wrote", out, "tasks", len(submission), f"total {time.time() - t0:.0f}s")
'''


def main() -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ARC Prize 2026 / ARC-AGI-2 — non-ML solver v2\n",
                "\n",
                "DSL program search (depth<=2 whole-grid ops + learned colour maps), per-cell and per-object\n",
                "rule induction, symmetry/periodic completion, split-and-combine truth tables, consensus voting.\n",
                "Pure Python, stdlib only. Internet must stay **off**.\n",
                "\n",
                "Source: https://github.com/lhzwb2008/arc-prize-2026\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in SOLVER.splitlines()],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in NOTEBOOK_DRIVER.strip().splitlines()],
        },
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    out = ROOT / "notebooks" / "kaggle_baseline.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    # plain script with the same content, handy for running the notebook logic locally
    script = ROOT / "notebooks" / "kaggle_baseline_as_script.py"
    script.write_text(SOLVER + "\n\n# ---- notebook driver ----\n" + NOTEBOOK_DRIVER.strip() + "\n", encoding="utf-8")
    print("wrote", out)
    print("wrote", script)


if __name__ == "__main__":
    main()
