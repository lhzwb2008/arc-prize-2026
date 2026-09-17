#!/usr/bin/env python3
"""CPU: A-full + expensive-first leftover-B, mixed mean_quality, under 16×16 wall.

Uses existing n8x6 pickle dirs on GPU2. Does not touch the GPU.

Wall cap defaults to the first full-cost 16×16 eval120 pickle span (~13.58h).
Pass A is kept in full; B is an expensive-first prefix. Ranking is mixed
mean_quality over A+leftover-B (keep-primary / pass-pair are diagnostics).

  python simulate_expensive_b.py \
      --outputs /opt/work/nvarc/eval120_n8x6_a/outputs \
      --outputs-b /opt/work/nvarc/eval120_n8x6_b/outputs \
      --full-16x16 /opt/work/nvarc/eval120/outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

N = 120
KAGGLE_H = 12.0
KAGGLE_STOP_MIN = 20.0


def pct(score: float, n: int = N) -> float:
    return 100.0 * float(score) / n


def estimated_work(task, n_train=8, n_geos=6):
    def ntok(g):
        return len(g) * (len(g[0]) + 1) if g and g[0] else 0
    train_tokens = sum(ntok(p["input"]) + ntok(p["output"]) for p in task["train"])
    ratios = [ntok(p["output"]) / max(1, ntok(p["input"])) for p in task["train"]] or [1.0]
    ratios.sort()
    ratio = ratios[len(ratios) // 2]
    test_tokens = sum(ntok(t["input"]) * (1 + ratio) for t in task["test"])
    return train_tokens * n_train + test_tokens * n_geos * len(task["test"])


def pickle_span_h(out_dir: str | Path) -> float | None:
    p = Path(out_dir)
    if not p.is_dir():
        return None
    ts = [f.stat().st_mtime for f in p.iterdir() if f.is_file()]
    if len(ts) < 2:
        return None
    return (max(ts) - min(ts)) / 3600.0


def task_done_mtime(out_dir: str | Path) -> dict[str, float]:
    done: dict[str, float] = {}
    p = Path(out_dir)
    if not p.is_dir():
        return done
    for f in p.iterdir():
        if not f.is_file() or "_" not in f.name:
            continue
        task = f.name.split("_", 1)[0]
        m = f.stat().st_mtime
        if task not in done or m > done[task]:
            done[task] = m
    return done


def sequential_durations(done_mtime: dict[str, float], start_mtime: float) -> dict[str, float]:
    order = sorted(done_mtime, key=lambda k: done_mtime[k])
    dur: dict[str, float] = {}
    prev = start_mtime
    for t in order:
        dur[t] = max(0.0, done_mtime[t] - prev)
        prev = done_mtime[t]
    return dur


def expensive_order(tasks: list[str], work: dict[str, float]) -> list[str]:
    return sorted(tasks, key=lambda k: work.get(k, 0.0), reverse=True)


def hours_for_subset(tasks: list[str], dur: dict[str, float]) -> float:
    return sum(dur.get(t, 0.0) for t in tasks) / 3600.0


def prefix_until_hours(order: list[str], dur: dict[str, float], hours: float) -> list[str]:
    if hours <= 0:
        return []
    keep: list[str] = []
    acc = 0.0
    budget = hours * 3600.0
    for t in order:
        step = dur.get(t, 0.0)
        if keep and acc + step > budget + 1e-9:
            break
        keep.append(t)
        acc += step
        if acc >= budget - 1e-9:
            break
    return keep


def restrict_decoded(decoded: dict, keep_tasks: set[str]) -> dict:
    out = {}
    for bk, samples in decoded.items():
        if bk.split("_")[0] in keep_tasks:
            out[bk] = samples
    return out


def merge_decoded(a: dict, b: dict) -> dict:
    out = {}
    for bk in set(a) | set(b):
        out[bk] = {**(a.get(bk) or {}), **(b.get(bk) or {})}
    return out


def pick_best_within_budget(rows: list[dict], budget_h: float) -> dict | None:
    ok = [r for r in rows if r["a_h"] + r["b_h"] <= budget_h + 1e-6]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["mixed_pct"], -r["b_h"]))


def load_sel(dm, path, run_name=""):
    from arc_decoder import ArcDecoder, score_mean_quality
    dec = ArcDecoder(dm, n_guesses=2)
    dec.load_decoded_results(path, run_name=run_name)
    sel = dec.run_selection_algo(score_mean_quality) if dec.decoded_results else {}
    return dec, sel


def score_pool(replies, decoded_a, decoded_b_slice, sel_a):
    from arc_decoder import ArcDecoder, merge_keep_primary, merge_pass_pair, score_mean_quality
    from rank_pool_search import oracle_score, task_score
    pooled = merge_decoded(decoded_a, decoded_b_slice)
    mix = ArcDecoder(None, n_guesses=2)
    mix.decoded_results = pooled
    sel_m = mix.run_selection_algo(score_mean_quality) if pooled else {}
    sel_b = {}
    if decoded_b_slice:
        bdec = ArcDecoder(None, n_guesses=2)
        bdec.decoded_results = decoded_b_slice
        sel_b = bdec.run_selection_algo(score_mean_quality)
    ora, in_pool, n_out = oracle_score(replies, pooled) if pooled else (0.0, 0, 0)
    mixed_sc, mixed_hit, _ = task_score(replies, sel_m)
    keep_sc, _, _ = task_score(replies, merge_keep_primary(sel_a, sel_m))
    pair_sc, _, _ = task_score(replies, merge_pass_pair(sel_a, sel_b))
    return {
        "mixed": mixed_sc,
        "mixed_pct": pct(mixed_sc),
        "mixed_top2": mixed_hit,
        "keepP": keep_sc,
        "keepP_pct": pct(keep_sc),
        "pair": pair_sc,
        "pair_pct": pct(pair_sc),
        "oracle": ora,
        "oracle_pct": pct(ora),
        "gold_in_pool": in_pool,
        "n_out": n_out,
        "n_b_tasks": len({bk.split("_")[0] for bk in decoded_b_slice}),
    }


def fmt_row(r):
    return (
        f"  nB={r['n_b_tasks']:3d}  work={r['work_frac']:.2f}  "
        f"B={r['b_h']:.2f}h  A+B={r['total_h']:.2f}h  "
        f"mixed {r['mixed_pct']:5.2f}  keepP {r['keepP_pct']:5.2f}  "
        f"pair {r['pair_pct']:5.2f}  ora {r['oracle_pct']:5.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/opt/data/kaggle/arc-agi_evaluation_challenges.json")
    ap.add_argument("--solutions", default="/opt/data/kaggle/arc-agi_evaluation_solutions.json")
    ap.add_argument("--outputs", default="/opt/work/nvarc/eval120_n8x6_a/outputs")
    ap.add_argument("--outputs-b", default="/opt/work/nvarc/eval120_n8x6_b/outputs")
    ap.add_argument("--full-16x16", default="/opt/work/nvarc/eval120/outputs")
    ap.add_argument("--t16-hours", type=float, default=0.0,
                    help="override 16x16 wall hours (0 = pickle span)")
    ap.add_argument("--a-hours", type=float, default=0.0,
                    help="override A wall hours (0 = pickle span)")
    ap.add_argument("--out", default="/opt/work/nvarc/eval120_n8x6_two/expensive_b_sim.json")
    ap.add_argument("--keys-out", default="/opt/work/nvarc/eval120_n8x6_two/b_expensive_keys.json")
    args = ap.parse_args()

    if not (os.path.isdir(args.outputs) and os.path.isdir(args.outputs_b)):
        print("missing pickle dirs; skip")
        return 1

    from arc_loader import ArcDataset
    from rank_pool_search import task_score

    raw = json.loads(Path(args.data).read_text())
    work = {k: float(estimated_work(raw[k])) for k in raw}
    tasks_all = list(raw)

    t16_h = args.t16_hours or pickle_span_h(args.full_16x16) or 13.58
    a_h = args.a_hours or pickle_span_h(args.outputs) or 8.04
    b_span = pickle_span_h(args.outputs_b) or 7.03

    done_b = task_done_mtime(args.outputs_b)
    start_b = min((f.stat().st_mtime for f in Path(args.outputs_b).iterdir() if f.is_file()), default=0.0)
    dur_b = sequential_durations(done_b, start_b)
    # tasks with no B pickle get a work-proportional duration estimate
    rate = (sum(dur_b.values()) / max(1e-9, sum(work[t] for t in dur_b))) if dur_b else 0.0
    for t in tasks_all:
        if t not in dur_b:
            dur_b[t] = work[t] * rate

    exp_order = expensive_order(tasks_all, work)
    mtime_order = sorted(done_b, key=lambda k: done_b[k])
    # Spearman-ish: rank correlation work vs observed completion
    rank_work = {t: i for i, t in enumerate(exp_order)}
    if mtime_order:
        xs = np.array([rank_work.get(t, 0) for t in mtime_order])
        ys = np.arange(len(mtime_order))
        corr = float(np.corrcoef(xs, ys)[0, 1]) if len(mtime_order) > 2 else 0.0
    else:
        corr = 0.0

    data = ArcDataset.from_file(args.data)
    data.load_replies(args.solutions)
    dm = data.split_multi_replies()
    replies = dm.replies

    print("loading pickles...", flush=True)
    dec_a, sel_a = load_sel(dm, args.outputs)
    dec_b, sel_b = load_sel(dm, args.outputs_b, run_name=".p1")
    a_sc, _, _ = task_score(replies, sel_a)
    b_sc, _, _ = task_score(replies, sel_b)
    full = score_pool(replies, dec_a.decoded_results, dec_b.decoded_results, sel_a)
    print(
        f"16x16 wall {t16_h:.2f}h  A {a_h:.2f}h ({pct(a_sc):.2f})  "
        f"B span {b_span:.2f}h ({pct(b_sc):.2f})  "
        f"full mixed {full['mixed_pct']:.2f}  keepP {full['keepP_pct']:.2f}",
        flush=True,
    )
    print(f"B mtime vs expensive-work rank corr {corr:.3f}  nA={len({bk.split('_')[0] for bk in dec_a.decoded_results})} "
          f"nB={len({bk.split('_')[0] for bk in dec_b.decoded_results})}", flush=True)

    a_ok = {}
    b_ok = {}
    for bk, golds in replies.items():
        t = bk.split("_")[0]
        ga = any(np.array_equal(g, golds[0]) for g in (sel_a.get(bk) or [])[:2])
        gb = any(np.array_equal(g, golds[0]) for g in (sel_b.get(bk) or [])[:2])
        a_ok.setdefault(t, []).append(ga)
        b_ok.setdefault(t, []).append(gb)
    a_pts = {t: sum(v) / len(v) for t, v in a_ok.items()}
    b_pts = {t: sum(v) / len(v) for t, v in b_ok.items()}
    b_unique = sorted(t for t in tasks_all if b_pts.get(t, 0) > a_pts.get(t, 0) + 1e-9)
    a_missing = sorted(t for t in tasks_all if t not in {bk.split("_")[0] for bk in dec_a.decoded_results})
    print(f"B-unique wins {len(b_unique)}  A-missing {len(a_missing)}", flush=True)
    for t in b_unique:
        print(
            f"  B-win {t}  expensive_rank={rank_work[t]+1:3d}/120  "
            f"A={a_pts.get(t,0):.2f} B={b_pts.get(t,0):.2f}  "
            f"dur={dur_b.get(t,0)/60:.1f}min",
            flush=True,
        )

    b_budget_local = max(0.0, t16_h - a_h)
    b_budget_kaggle = max(0.0, KAGGLE_H - KAGGLE_STOP_MIN / 60.0 - a_h)
    print(
        f"\nB leftover budgets: local_16x16 {b_budget_local:.2f}h  "
        f"kaggle_12h-20min {b_budget_kaggle:.2f}h (A assumed {a_h:.2f}h)",
        flush=True,
    )

    hour_grid = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, b_budget_kaggle, 5.0, b_budget_local, 6.0, 7.0, b_span]
    hour_grid = sorted({round(h, 3) for h in hour_grid if h >= 0})

    rows = []
    print("\n=== expensive-first leftover-B + mixed mean_quality ===", flush=True)
    seen = set()
    for hours in hour_grid:
        keep = prefix_until_hours(exp_order, dur_b, hours)
        key = tuple(keep)
        if key in seen:
            continue
        seen.add(key)
        decoded_b = restrict_decoded(dec_b.decoded_results, set(keep))
        sc = score_pool(replies, dec_a.decoded_results, decoded_b, sel_a)
        b_h = hours_for_subset(keep, dur_b)
        work_sum = sum(work[t] for t in keep)
        work_all = sum(work.values()) or 1.0
        n_b_unique = sum(1 for t in b_unique if t in keep)
        n_miss = sum(1 for t in a_missing if t in keep)
        row = {
            **sc,
            "kind": "expensive-first",
            "n_b_tasks": len(keep),
            "n_b_decoded": sc["n_b_tasks"],
            "b_h": b_h,
            "a_h": a_h,
            "total_h": a_h + b_h,
            "work_frac": work_sum / work_all,
            "b_unique_in": n_b_unique,
            "a_missing_in": n_miss,
            "within_16x16": a_h + b_h <= t16_h + 1e-6,
            "within_kaggle12": a_h + b_h <= (KAGGLE_H - KAGGLE_STOP_MIN / 60.0) + 1e-6,
            "keys": keep,
            **sc,
            "mixed_pct": sc["mixed_pct"],
            "keepP_pct": sc["keepP_pct"],
            "pair_pct": sc["pair_pct"],
            "oracle_pct": sc["oracle_pct"],
        }
        rows.append(row)
        flag = ""
        if row["within_16x16"]:
            flag += " [<=16x16]"
        if row["within_kaggle12"]:
            flag += " [<=kaggle12]"
        print(fmt_row(row) + f"  B-wins {n_b_unique}/{len(b_unique)}  A-miss {n_miss}/{len(a_missing)}{flag}", flush=True)

    rec_local = pick_best_within_budget(rows, t16_h)
    rec_kaggle = pick_best_within_budget(rows, KAGGLE_H - KAGGLE_STOP_MIN / 60.0)
    print("\n=== recommend (max mixed mean_q, A+B wall <= budget) ===", flush=True)
    for name, rec, bud in (("local_16x16", rec_local, t16_h), ("kaggle_12h", rec_kaggle, KAGGLE_H - KAGGLE_STOP_MIN / 60.0)):
        if not rec:
            print(f"  {name} budget {bud:.2f}h: no row fits (A already {a_h:.2f}h)")
            continue
        delta = rec["mixed_pct"] - pct(a_sc)
        print(
            f"  {name} budget {bud:.2f}h -> nB={rec['n_b_tasks']}  B={rec['b_h']:.2f}h  "
            f"total {rec['total_h']:.2f}h  mixed {rec['mixed_pct']:.2f} "
            f"({delta:+.2f} vs A)  keepP {rec['keepP_pct']:.2f}  pair {rec['pair_pct']:.2f}",
            flush=True,
        )

    rec = rec_local or rec_kaggle
    doc = {
        "t16_h": t16_h,
        "a_h": a_h,
        "b_span_h": b_span,
        "a_pct": pct(a_sc),
        "b_pct": pct(b_sc),
        "full_mixed_pct": full["mixed_pct"],
        "full_keepP_pct": full["keepP_pct"],
        "mtime_vs_expensive_corr": corr,
        "b_unique": b_unique,
        "a_missing": a_missing,
        "b_budget_local_h": b_budget_local,
        "b_budget_kaggle_h": b_budget_kaggle,
        "recommend_local": None if not rec_local else {k: rec_local[k] for k in rec_local if k != "keys"},
        "recommend_kaggle": None if not rec_kaggle else {k: rec_kaggle[k] for k in rec_kaggle if k != "keys"},
        "rows": [{k: v for k, v in r.items() if k != "keys"} for r in rows],
        "ranker": "mixed mean_quality",
        "b_order": "expensive-first estimated_work",
        "note": (
            "Tomorrow stays single-pass 8x6. Day-after: A cheap-first to completion, "
            "B expensive-first until wall cap, mixed mean_quality (no full B, no keep-primary)."
        ),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print("wrote", args.out, flush=True)
    if rec and args.keys_out:
        Path(args.keys_out).parent.mkdir(parents=True, exist_ok=True)
        keys = rec["keys"]
        Path(args.keys_out).write_text(json.dumps(keys, indent=2) + "\n")
        print(f"wrote {len(keys)} B keys -> {args.keys_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
