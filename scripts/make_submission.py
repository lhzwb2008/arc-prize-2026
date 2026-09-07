#!/usr/bin/env python3
"""Build submission.json from a Kaggle-style challenges file or a folder of tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_solver.baseline import build_submission, flatten_challenges  # noqa: E402


def load_challenges(path: Path) -> dict[str, dict]:
    if path.is_dir():
        tasks = {}
        for file in sorted(path.glob("*.json")):
            payload = json.loads(file.read_text())
            tasks.update(flatten_challenges(payload, stem=file.stem))
        return tasks
    payload = json.loads(path.read_text())
    return flatten_challenges(payload, stem=path.stem)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "examples",
        help="challenges JSON or a directory of per-task JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "demo" / "output" / "submission.json",
    )
    args = parser.parse_args()
    challenges = load_challenges(args.input)
    submission = build_submission(challenges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(submission, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.output} with {len(submission)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
