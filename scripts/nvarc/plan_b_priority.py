#!/usr/bin/env python3
"""CPU: which B queue order lifts the pooled score fastest? Uses GPU2 n8x6 pickles.

Every order is a generic rule over (pass-A own outputs, estimated cost). No task
ids. Clock uses measured B per-task durations; tasks B never decoded still cost
their estimated time and add nothing (conservative).

    python plan_b_priority.py \
      --outputs /opt/work/nvarc/eval120_n8x6_a/outputs \
      --outputs-b /opt/work/nvarc/eval120_n8x6_b/outputs
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

from simulate_expensive_b import (  # noqa: E402
    N,
    bootstrap_delta_pct,
    estimated_work,
    hours_for_subset,
    pct,
    pickle_span_h,
    prefix_until_hours,
    restrict_decoded,
    sequential_durations,
    task_done_mtime,
)

BIG = 1e9


def _hashable(g):
    return tuple(map(tuple, g))


def output_signals(samples: dict) -> dict:
    """Pass-A confidence for one test output from its own candidates."""
    gs = list(samples.values())
    n = len(gs)
    if n == 0:
        return {"n": 0, "share": 0.0, "margin": 0.0, "top_q": -BIG, "n_unique": 0}
    groups: dict = {}
    for g in gs:
        groups.setdefault(_hashable(g["solution"]), []).append(g)
    ranked = sorted(
        ((float(np.mean([-x["beam_score"] - np.mean(x["score_aug"]) for x in grp])), len(grp))
         for grp in groups.values()),
        reverse=True,
    )
    top_q, top_votes = ranked[0]
    margin = top_q - ranked[1][0] if len(ranked) > 1 else BIG
    return {
        "n": n,
        "share": top_votes / n,
        "margin": float(margin),
        "top_q": float(top_q),
        "n_unique": len(groups),
    }


def task_signals(decoded_a: dict, tasks: list[str]) -> dict[str, dict]:
    by_task: dict[str, list] = {t: [] for t in tasks}
    for bk, samples in decoded_a.items():
        t = bk.split("_")[0]
        if t in by_task:
            by_task[t].append(output_signals(samples))
    out = {}
    for t, sigs in by_task.items():
        if not sigs:
            out[t] = {"missing": True, "share": 0.0, "margin": 0.0, "top_q": -BIG, "n_unique": 0}
            continue
        out[t] = {
            "missing": False,
            "share": min(s["share"] for s in sigs),
            "margin": min(s["margin"] for s in sigs),
            "top_q": min(s["top_q"] for s in sigs),
            "n_unique": max(s["n_unique"] for s in sigs),
        }
    return out


def rank_norm(values: dict[str, float]) -> dict[str, float]:
    """0..1 by rank, ties share the mean rank. Lower value -> lower rank."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    out: dict[str, float] = {}
    i = 0
    n = len(items)
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        r = (i + j) / 2.0 / max(1, n - 1)
        for k in range(i, j + 1):
            out[items[k][0]] = r
        i = j + 1
    return out


def build_orders(tasks: list[str], work: dict[str, float], sig: dict[str, dict]) -> dict[str, list[str]]:
    cheap = sorted(tasks, key=lambda t: work[t])
    expensive = list(reversed(cheap))
    missing = [t for t in cheap if sig[t]["missing"]]
    present = [t for t in cheap if not sig[t]["missing"]]

    unc_share = {t: 1.0 - sig[t]["share"] for t in tasks}          # missing -> 1.0
    unc_margin = {t: -sig[t]["margin"] for t in tasks}              # missing -> 0 (> any -margin)
    low_q = {t: -sig[t]["top_q"] for t in tasks}                    # missing -> +BIG

    def per_cost(score: dict[str, float]) -> list[str]:
        return sorted(tasks, key=lambda t: -(score[t] / max(1.0, work[t])))

    def per_cost_rank(score: dict[str, float]) -> list[str]:
        r = rank_norm(score)
        return sorted(tasks, key=lambda t: -(r[t] / max(1.0, work[t])))

    return {
        "cheap": cheap,
        "expensive": expensive,
        "missing_then_cheap": missing + present,
        "unc_share_per_cost": per_cost(unc_share),
        "unc_share": sorted(tasks, key=lambda t: (-unc_share[t], work[t])),
        "margin_per_cost": per_cost_rank(unc_margin),
        "lowq_per_cost": per_cost_rank(low_q),
    }


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    rx = rank_norm({i: v for i, v in enumerate(xs)})
    ry = rank_norm({i: v for i, v in enumerate(ys)})
    a = np.array([rx[i] for i in range(len(xs))])
    b = np.array([ry[i] for i in range(len(xs))])
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/opt/data/kaggle/arc-agi_evaluation_challenges.json")
    ap.add_argument("--solutions", default="/opt/data/kaggle/arc-agi_evaluation_solutions.json")
    ap.add_argument("--outputs", default="/opt/work/nvarc/eval120_n8x6_a/outputs")
    ap.add_argument("--outputs-b", default="/opt/work/nvarc/eval120_n8x6_b/outputs")
    ap.add_argument("--budgets", default="0.5,1,1.5,2,3,4,5,5.3,5.5")
    ap.add_argument("--out", default="/opt/work/nvarc/eval120_n8x6_two/plan_b_priority.json")
    args = ap.parse_args()

    if not (os.path.isdir(args.outputs) and os.path.isdir(args.outputs_b)):
        print("missing pickle dirs; skip")
        return 1

    from arc_loader import ArcDataset
    from rank_pool_search import task_score
    from simulate_expensive_b import load_sel, score_pool

    raw = json.loads(Path(args.data).read_text())
    tasks = list(raw)
    work = {k: float(estimated_work(raw[k])) for k in raw}

    data = ArcDataset.from_file(args.data)
    data.load_replies(args.solutions)
    dm = data.split_multi_replies()
    replies = dm.replies

    print("loading pickles...", flush=True)
    dec_a, sel_a = load_sel(dm, args.outputs)
    dec_b, sel_b = load_sel(dm, args.outputs_b, run_name=".p1")
    a_sc, _, _ = task_score(replies, sel_a)
    a_h = pickle_span_h(args.outputs) or 8.04

    done_b = task_done_mtime(args.outputs_b)
    start_b = min((f.stat().st_mtime for f in Path(args.outputs_b).iterdir() if f.is_file()), default=0.0)
    dur_b = sequential_durations(done_b, start_b)
    rate = (sum(dur_b.values()) / max(1e-9, sum(work[t] for t in dur_b))) if dur_b else 0.0
    for t in tasks:
        dur_b.setdefault(t, work[t] * rate)

    sig = task_signals(dec_a.decoded_results, tasks)
    b_tasks = {bk.split("_")[0] for bk in dec_b.decoded_results}

    # per-task lift when B_i alone joins A (keep-primary and mixed)
    per_a = {}
    for bk, golds in replies.items():
        ok = any(np.array_equal(g, golds[0]) for g in (sel_a.get(bk) or [])[:2])
        per_a.setdefault(bk.split("_")[0], []).append(bool(ok))
    a_pts = {t: sum(v) / len(v) for t, v in per_a.items()}

    lift_keep: dict[str, float] = {}
    lift_mixed: dict[str, float] = {}
    for t in sorted(b_tasks):
        sc = score_pool(replies, dec_a.decoded_results,
                        restrict_decoded(dec_b.decoded_results, {t}), sel_a)
        lift_mixed[t] = sc["mixed_pts"].get(t, 0.0) - a_pts.get(t, 0.0)
        # keep-primary per-task: recompute from selection
        from arc_decoder import ArcDecoder, merge_keep_primary, score_mean_quality
        pooled = {}
        for bk in set(dec_a.decoded_results) | set(restrict_decoded(dec_b.decoded_results, {t})):
            pooled[bk] = {**(dec_a.decoded_results.get(bk) or {}),
                          **(restrict_decoded(dec_b.decoded_results, {t}).get(bk) or {})}
        mix = ArcDecoder(None, n_guesses=2)
        mix.decoded_results = pooled
        sel_k = merge_keep_primary(sel_a, mix.run_selection_algo(score_mean_quality))
        oks = []
        for bk, golds in replies.items():
            if bk.split("_")[0] != t:
                continue
            oks.append(any(np.array_equal(g, golds[0]) for g in (sel_k.get(bk) or [])[:2]))
        lift_keep[t] = (sum(oks) / len(oks) if oks else 0.0) - a_pts.get(t, 0.0)

    print(f"\nA {pct(a_sc):.2f}  A wall {a_h:.2f}h  B decoded {len(b_tasks)}  "
          f"A-missing {sum(1 for t in tasks if sig[t]['missing'])}", flush=True)
    winners = sorted((t for t in lift_keep if lift_keep[t] > 1e-9), key=lambda t: work[t])
    losers = sorted(t for t in lift_mixed if lift_mixed[t] < -1e-9)
    print(f"B lifts keepP on {len(winners)} tasks (+{sum(lift_keep.values()):.2f} pts); "
          f"mixed hurts on {len(losers)} tasks ({sum(lift_mixed[t] for t in losers):.2f})", flush=True)
    cheap_rank = {t: i + 1 for i, t in enumerate(sorted(tasks, key=lambda t: work[t]))}
    print("  task      cheap_rank  dur_min  A_missing  share  margin  top_q  liftK")
    for t in winners:
        s = sig[t]
        print(f"  {t}  {cheap_rank[t]:4d}/120  {dur_b[t]/60:6.1f}  {str(s['missing']):5s}  "
              f"{s['share']:.2f}  {min(s['margin'], 99):6.2f}  {max(s['top_q'], -99):6.2f}  {lift_keep[t]:+.2f}",
              flush=True)

    # does any A signal predict lift?
    bt = sorted(b_tasks)
    lifts = [lift_keep[t] for t in bt]
    print("\nSpearman(signal, keepP lift) over B-decoded tasks:")
    for name, vals in (
        ("1-share", [1 - sig[t]["share"] for t in bt]),
        ("-margin", [-min(sig[t]["margin"], 99) for t in bt]),
        ("-top_q", [-max(sig[t]["top_q"], -99) for t in bt]),
        ("missing", [1.0 if sig[t]["missing"] else 0.0 for t in bt]),
        ("-work", [-work[t] for t in bt]),
        ("-dur", [-dur_b[t] for t in bt]),
    ):
        print(f"  {name:9s} {spearman(vals, lifts):+.3f}", flush=True)
    # lift rate by A-confidence bucket
    conf = sorted((t for t in bt if not sig[t]["missing"]), key=lambda t: sig[t]["share"])
    k = max(1, len(conf) // 3)
    for label, grp in (("low-share", conf[:k]), ("mid-share", conf[k:2 * k]), ("high-share", conf[2 * k:])):
        if grp:
            print(f"  {label:10s} n={len(grp):3d} lift/task {np.mean([lift_keep[t] for t in grp]):+.3f} "
                  f"share[{sig[grp[0]]['share']:.2f}..{sig[grp[-1]]['share']:.2f}]", flush=True)
    miss = [t for t in bt if sig[t]["missing"]]
    if miss:
        print(f"  A-missing  n={len(miss):3d} lift/task {np.mean([lift_keep[t] for t in miss]):+.3f}", flush=True)

    orders = build_orders(tasks, work, sig)
    oracle = sorted(tasks, key=lambda t: (-(lift_keep.get(t, 0.0) / max(60.0, dur_b[t])), work[t]))
    orders["oracle_lift_per_cost"] = oracle

    budgets = sorted({float(x) for x in args.budgets.split(",") if x.strip()})
    print("\n=== keep-primary pct by order x leftover-B hours (A + B prefix) ===")
    hdr = "  order                    " + "".join(f"{b:>7.1f}h" for b in budgets)
    print(hdr, flush=True)
    table: dict[str, dict] = {}
    pts_cache: dict[str, dict] = {}
    for name, order in orders.items():
        row_k = []
        row_m = []
        row_n = []
        for hb in budgets:
            keep = prefix_until_hours(order, dur_b, hb)
            key = tuple(sorted(keep))
            if key not in pts_cache:
                sc = score_pool(replies, dec_a.decoded_results,
                                restrict_decoded(dec_b.decoded_results, set(keep)), sel_a)
                pts_cache[key] = sc
            sc = pts_cache[key]
            row_k.append(sc["keepP_pct"])
            row_m.append(sc["mixed_pct"])
            row_n.append(len(keep))
        table[name] = {"keepP": row_k, "mixed": row_m, "nB": row_n}
        print(f"  {name:25s}" + "".join(f"{v:8.2f}" for v in row_k), flush=True)
    print("\n=== mixed pct ===")
    for name in orders:
        print(f"  {name:25s}" + "".join(f"{v:8.2f}" for v in table[name]["mixed"]), flush=True)
    print("\n=== nB in prefix ===")
    for name in orders:
        print(f"  {name:25s}" + "".join(f"{v:8d}" for v in table[name]["nB"]), flush=True)

    # paired bootstrap vs cheap at each budget (keep-primary points per task)
    print("\n=== rule minus cheap-first, keepP, paired task bootstrap CI90 ===")
    boots = {}
    for name, order in orders.items():
        if name == "cheap":
            continue
        row = []
        for hb in budgets:
            k_rule = set(prefix_until_hours(order, dur_b, hb))
            k_cheap = set(prefix_until_hours(orders["cheap"], dur_b, hb))
            p_rule = {t: a_pts.get(t, 0.0) + (lift_keep.get(t, 0.0) if t in k_rule else 0.0) for t in tasks}
            p_cheap = {t: a_pts.get(t, 0.0) + (lift_keep.get(t, 0.0) if t in k_cheap else 0.0) for t in tasks}
            b = bootstrap_delta_pct(p_cheap, p_rule, iters=1000, seed=1)
            row.append((b["mean_pct"], b["ci90_pct"][0], b["ci90_pct"][1], b["p_lt_0"]))
        boots[name] = row
        print(f"  {name:25s}" + "".join(f" {m:+5.2f}[{lo:+4.1f},{hi:+4.1f}]" for m, lo, hi, _ in row), flush=True)

    doc = {
        "a_pct": pct(a_sc),
        "a_h": a_h,
        "budgets_h": budgets,
        "n_b_decoded": len(b_tasks),
        "n_a_missing": sum(1 for t in tasks if sig[t]["missing"]),
        "winners": {t: {"lift_keep": lift_keep[t], "dur_min": dur_b[t] / 60, "cheap_rank": cheap_rank[t],
                        **{k: (v if abs(v) < 1e6 else None) for k, v in sig[t].items()}} for t in winners},
        "mixed_losers": losers,
        "table": table,
        "bootstrap_vs_cheap": {k: [{"mean": m, "ci90": [lo, hi], "p_lt_0": p} for m, lo, hi, p in v]
                               for k, v in boots.items()},
        "note": "orders use only pass-A outputs + estimated work; oracle row is the ceiling, not a policy",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2, default=float) + "\n")
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
