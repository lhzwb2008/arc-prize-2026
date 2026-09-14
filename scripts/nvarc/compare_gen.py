#!/usr/bin/env python3
"""Score + oracle for a pickle dir; append to the generation-queue summary."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arc_decoder import ArcDecoder, score_kgmon, score_mean_quality
from arc_loader import ArcDataset
from rank_pool_search import oracle_score, task_score

DATA = "/opt/data/kaggle/arc-agi_evaluation_challenges.json"
SOL = "/opt/data/kaggle/arc-agi_evaluation_solutions.json"
SUMMARY = "/opt/work/nvarc/eval120_search/gen_queue_summary.json"
BASELINE = "/opt/work/nvarc/eval120_n6x6_a/outputs"


def metrics(dm, replies, outputs: str):
    dec = ArcDecoder(dm, n_guesses=2)
    dec.load_decoded_results(outputs)
    ora, in_pool, n_out = oracle_score(replies, dec.decoded_results)
    mq, mq_hit, _ = task_score(replies, dec.run_selection_algo(score_mean_quality))
    kg, kg_hit, _ = task_score(replies, dec.run_selection_algo(score_kgmon))
    return {
        "oracle": ora,
        "mean_quality": mq,
        "kgmon": kg,
        "gold_in_pool": in_pool,
        "n_out": n_out,
        "mean_quality_top2": mq_hit,
        "kgmon_top2": kg_hit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--summary", default=SUMMARY)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--solutions", default=SOL)
    args = ap.parse_args()

    data = ArcDataset.from_file(args.data)
    data.load_replies(args.solutions)
    dm = data.split_multi_replies()
    replies = dm.replies
    n = 120

    row = metrics(dm, replies, args.outputs)
    base = metrics(dm, replies, args.baseline) if Path(args.baseline).is_dir() else None
    print(f"{args.tag}")
    print(
        f"  mean_q {row['mean_quality']:.3f} ({100 * row['mean_quality'] / n:.2f}%)  "
        f"kgmon {row['kgmon']:.3f}  oracle {row['oracle']:.3f}  "
        f"gold {row['gold_in_pool']}/{row['n_out']}"
    )
    if base:
        print(
            f"  vs 6x6A  mean_q {row['mean_quality'] - base['mean_quality']:+.3f}  "
            f"oracle {row['oracle'] - base['oracle']:+.3f}"
        )

    path = Path(args.summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.exists() else {"baseline_6x6A": base, "runs": []}
    if base and "baseline_6x6A" not in doc:
        doc["baseline_6x6A"] = base
    doc.setdefault("runs", [])
    doc["runs"] = [r for r in doc["runs"] if r.get("tag") != args.tag]
    doc["runs"].append({"tag": args.tag, "outputs": args.outputs, "at": time.strftime("%F %T"), **row})
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
