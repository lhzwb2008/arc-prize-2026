#!/usr/bin/env python3
"""Embed src/arc_solver/baseline.py into a Kaggle-ready notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER = (ROOT / "src" / "arc_solver" / "baseline.py").read_text()

NOTEBOOK_DRIVER = r'''
from pathlib import Path
import json

INPUT_DIR = Path("/kaggle/input")
candidates = []
if INPUT_DIR.exists():
    for folder in INPUT_DIR.iterdir():
        candidates.extend(folder.glob("*test*challenges*.json"))
        candidates.extend(folder.glob("*test-challenges*.json"))
local_examples = Path("/kaggle/working")
# Local fallback used when this notebook is executed outside Kaggle.
repo_examples = Path.cwd() / "data" / "examples"

def load_challenges():
    for path in candidates:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data and "train" not in data:
            print("using", path)
            return data
    tasks = {}
    if repo_examples.exists():
        for file in sorted(repo_examples.glob("*.json")):
            tasks[file.stem] = json.loads(file.read_text())
        print("using local examples", repo_examples, "n=", len(tasks))
        return tasks
    raise FileNotFoundError("no ARC challenges file found")

challenges = load_challenges()
submission = build_submission(challenges)
out = Path("/kaggle/working/submission.json")
if not Path("/kaggle/working").exists():
    out = Path("demo/output/kaggle_submission.json")
    out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(submission, separators=(",", ":")))
print("wrote", out, "tasks", len(submission))
'''


def main() -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ARC Prize 2026 / ARC-AGI-2 baseline\n",
                "\n",
                "First-pass transform search. Internet must stay **off** when submitting on Kaggle.\n",
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
    print("wrote", out)


if __name__ == "__main__":
    main()
