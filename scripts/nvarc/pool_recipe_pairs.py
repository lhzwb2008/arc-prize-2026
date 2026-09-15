#!/usr/bin/env python3
"""CPU: single-run vs pooled oracle/selection for mixed TTT recipes.

Does not touch the GPU. 8×6 / 8×5 from 8×8A are view-cuts of the same adapter
(nested). Independent TTT dirs (6×6, 5×8, 8×8B) are the complementarity test.
"""
from __future__ import annotations

import json
from pathlib import Path

from arc_decoder import ArcDecoder, hashable, score_kgmon, score_mean_quality
from arc_loader import ArcDataset
from rank_pool_search import oracle_score, task_score
from score_view_subset import keep_files

DATA = "/opt/data/kaggle/arc-agi_evaluation_challenges.json"
SOL = "/opt/data/kaggle/arc-agi_evaluation_solutions.json"
WORK = Path("/opt/work/nvarc")
OUT = WORK / "eval120_search" / "recipe_pool.json"
N = 120

RUNS = {
    "8x8A": (str(WORK / "eval120_half_a/outputs"), None),
    "8x8B": (str(WORK / "eval120_half_b/outputs"), None),
    "8x6A": (str(WORK / "eval120_half_a/outputs"), 6),  # same TTT as 8x8A, fewer views
    "8x5A": (str(WORK / "eval120_half_a/outputs"), 5),
    "6x6A": (str(WORK / "eval120_n6x6_v11_a/outputs"), None),
    "6x8": (str(WORK / "eval120_search/n6_g8/outputs"), None),
    "5x8": (str(WORK / "eval120_search/n5_g8/outputs"), None),
}

PAIRS = [
    ("8x6A", "8x8A", "nested: 8x6 is a view-cut of 8x8A"),
    ("8x6A", "8x8B", "8x6A (cut) + independent 8x8B seed"),
    ("8x6A", "8x5A", "nested: both cuts of 8x8A"),
    ("8x6A", "6x6A", "independent TTT: 8-train/6-view vs 6x6"),
    ("8x6A", "5x8", "independent TTT: 8x6 vs 5-train/8-view"),
    ("8x8A", "6x6A", "independent TTT: 8x8 vs 6x6"),
    ("8x8A", "5x8", "independent TTT: 8x8 vs 5x8"),
    ("8x8A", "6x8", "independent TTT: 8x8 vs 6x8"),
    ("8x8A", "8x8B", "same recipe, two seeds (current v12-style)"),
]


def load_decoded(dm, outputs: str, geos: int | None, prefix: str = ""):
    dec = ArcDecoder(dm, n_guesses=2)
    dec.load_decoded_results(outputs, run_name=prefix)
    decoded = dec.decoded_results
    if geos:
        kept = set(keep_files(outputs, geos, None))
        out = {}
        for bk, samples in decoded.items():
            filt = {}
            for k, v in samples.items():
                fn = k.split(".out")[0]
                if prefix and fn.endswith(prefix):
                    fn = fn[: -len(prefix)]
                if fn in kept:
                    filt[k] = v
            if filt:
                out[bk] = filt
        return out
    return decoded


def merge(a: dict, b: dict) -> dict:
    out = {}
    for bk in set(a) | set(b):
        out[bk] = {**(a.get(bk) or {}), **(b.get(bk) or {})}
    return out


def gold_sets(replies, decoded):
    only = {"a": 0, "b": 0, "both": 0, "none": 0}
    per_task = {}
    for bk, golds in replies.items():
        hg = hashable(golds[0])
        in_a = any(_is_gold(s, hg) for s in (decoded[0].get(bk) or {}).values())
        in_b = any(_is_gold(s, hg) for s in (decoded[1].get(bk) or {}).values())
        if in_a and in_b:
            tag = "both"
        elif in_a:
            tag = "a"
        elif in_b:
            tag = "b"
        else:
            tag = "none"
        only[tag] += 1
        per_task.setdefault(bk.split("_")[0], []).append(tag)
    return only, per_task


def _is_gold(sample, hg) -> bool:
    try:
        return hashable(sample["solution"]) == hg
    except Exception:
        return False


def metrics(replies, decoded):
    ora, in_pool, n_out = oracle_score(replies, decoded)
    mq, mq_hit, _ = task_score(replies, {bk: score_mean_quality(v) for bk, v in decoded.items()})
    kg, kg_hit, _ = task_score(replies, {bk: score_kgmon(v) for bk, v in decoded.items()})
    n_cand = sum(len(v) for v in decoded.values())
    return {
        "oracle": ora,
        "oracle_pct": 100.0 * ora / N,
        "mean_quality": mq,
        "mean_quality_pct": 100.0 * mq / N,
        "kgmon": kg,
        "kgmon_pct": 100.0 * kg / N,
        "gold_in_pool": in_pool,
        "n_out": n_out,
        "n_candidates": n_cand,
        "mean_quality_top2": mq_hit,
        "kgmon_top2": kg_hit,
    }


def pct(x: float) -> str:
    return f"{100.0 * x / N:5.2f}"


def main() -> None:
    data = ArcDataset.from_file(DATA)
    data.load_replies(SOL)
    dm = data.split_multi_replies()
    replies = dm.replies

    loaded = {}
    singles = {}
    print("=== singles ===")
    print(f"{'run':<8} {'mq':>6} {'kg':>6} {'ora':>6}  gold  cands")
    for name, (path, geos) in RUNS.items():
        if not Path(path).is_dir():
            print(f"{name}: MISSING {path}")
            continue
        loaded[name] = load_decoded(dm, path, geos, prefix=f".{name}")
        singles[name] = metrics(replies, loaded[name])
        s = singles[name]
        print(
            f"{name:<8} {s['mean_quality_pct']:6.2f} {s['kgmon_pct']:6.2f} {s['oracle_pct']:6.2f}  "
            f"{s['gold_in_pool']:4d}/{s['n_out']}  {s['n_candidates']:5d}"
        )

    pairs = []
    print("\n=== pairs (mean_q / oracle; gold only-A / only-B / both) ===")
    print(f"{'pair':<22} {'mq':>6} {'ora':>6}  {'d_mq':>6} {'d_ora':>6}  onlyA onlyB both  note")
    for a, b, note in PAIRS:
        if a not in loaded or b not in loaded:
            print(f"{a}+{b}: skip")
            continue
        pooled = merge(loaded[a], loaded[b])
        m = metrics(replies, pooled)
        uniq, _ = gold_sets(replies, (loaded[a], loaded[b]))
        base_ora = max(singles[a]["oracle"], singles[b]["oracle"])
        base_mq = max(singles[a]["mean_quality"], singles[b]["mean_quality"])
        row = {
            "a": a,
            "b": b,
            "note": note,
            **m,
            "gold_only_a": uniq["a"],
            "gold_only_b": uniq["b"],
            "gold_both": uniq["both"],
            "gold_none": uniq["none"],
            "delta_mq_vs_best_single": m["mean_quality"] - base_mq,
            "delta_oracle_vs_best_single": m["oracle"] - base_ora,
        }
        pairs.append(row)
        print(
            f"{a}+{b:<8} {m['mean_quality_pct']:6.2f} {m['oracle_pct']:6.2f}  "
            f"{row['delta_mq_vs_best_single']:+6.2f} {row['delta_oracle_vs_best_single']:+6.2f}  "
            f"{uniq['a']:5d} {uniq['b']:5d} {uniq['both']:4d}  {note}"
        )

    doc = {"n_tasks": N, "singles": singles, "pairs": pairs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
