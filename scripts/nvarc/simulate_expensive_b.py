#!/usr/bin/env python3
"""CPU: leftover-B under the Kaggle-12h-equivalent local wall.

The equivalent wall is max(first 16×16 pickle span, GPU 6×6 two-pass spans).
6×6 two-pass is known COMPLETE on hidden Kaggle 12h, so the longer of those
local walls is the 12h cap; leftover-B is that cap minus pass-A hours.

Uses existing n8x6 pickle dirs on GPU2. Does not touch the GPU.

A stays full; leftover B is a cheap-first prefix until that cap. Ranking is
mixed mean_quality. Expensive-first prefix is reported as a comparison only.
Do not skip a fitted number of expensive tasks — hidden eval is a different 120.

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


def two_pass_hours(a_dir: str | Path, b_dir: str | Path) -> float | None:
    a = pickle_span_h(a_dir)
    b = pickle_span_h(b_dir)
    if a is None or b is None:
        return None
    return a + b


def pick_kaggle_equiv_wall(named_hours: dict[str, float]) -> tuple[str, float]:
    """Longer local wall that still maps to a Kaggle 12h COMPLETE (6x6 two-pass)."""
    named = {k: v for k, v in named_hours.items() if v and v > 0}
    if not named:
        return "none", 0.0
    name = max(named, key=named.get)
    return name, float(named[name])


def leftover_hours(cap_h: float, a_h: float) -> float:
    return max(0.0, float(cap_h) - float(a_h))


def timing_total_hours(path: str | Path) -> float | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    if d.get("total_sec") is not None:
        return float(d["total_sec"]) / 3600.0
    a, b = d.get("pass_a_sec"), d.get("pass_b_sec")
    if a is not None and b is not None:
        return (float(a) + float(b)) / 3600.0
    return None


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


def cheap_order(tasks: list[str], work: dict[str, float]) -> list[str]:
    return sorted(tasks, key=lambda k: work.get(k, 0.0))


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


def bootstrap_delta_pct(pts_base: dict, pts_new: dict, iters: int = 2000, seed: int = 0):
    """Paired task bootstrap of (new - base) in Kaggle percent. Same 120 labels."""
    tasks = sorted(set(pts_base) | set(pts_new))
    d = np.array([pts_new.get(t, 0.0) - pts_base.get(t, 0.0) for t in tasks])
    rng = np.random.default_rng(seed)
    n = len(d)
    draws = np.array([d[rng.integers(0, n, n)].sum() for _ in range(iters)])
    scale = 100.0 / N
    return {
        "mean_pct": float(draws.mean() * scale),
        "p_lt_0": float((draws < 0).mean()),
        "p_eq_0": float((draws == 0).mean()),
        "p_gt_0": float((draws > 0).mean()),
        "ci90_pct": [float(np.percentile(draws, 5) * scale), float(np.percentile(draws, 95) * scale)],
    }


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
    per = {}
    for bk, golds in replies.items():
        ok = any(np.array_equal(g, golds[0]) for g in (sel_m.get(bk) or [])[:2])
        per.setdefault(bk.split("_")[0], []).append(bool(ok))
    mixed_pts = {t: sum(v) / len(v) for t, v in per.items()}
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
        "mixed_pts": mixed_pts,
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
    ap.add_argument("--n6x6-a", default="/opt/work/nvarc/eval120_n6x6_v11_a/outputs")
    ap.add_argument("--n6x6-b", default="/opt/work/nvarc/eval120_n6x6_v11_b/outputs")
    ap.add_argument("--n6x6-old-a", default="/opt/work/nvarc/eval120_n6x6_a/outputs")
    ap.add_argument("--n6x6-old-b", default="/opt/work/nvarc/eval120_n6x6_b/outputs")
    ap.add_argument("--n6x6-timing", default="/opt/work/nvarc/eval120_n6x6_pool/timing.json")
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
    n6_v11 = two_pass_hours(args.n6x6_a, args.n6x6_b)
    n6_old = two_pass_hours(args.n6x6_old_a, args.n6x6_old_b)
    n6_timing = timing_total_hours(args.n6x6_timing)
    walls = {"16x16": t16_h}
    if n6_v11:
        walls["n6x6_v11_A+B"] = n6_v11
    if n6_old:
        walls["n6x6_old_A+B"] = n6_old
    if n6_timing:
        walls["n6x6_old_timing"] = n6_timing
    cap_name, cap_h = pick_kaggle_equiv_wall(walls)
    print("=== local walls (1x3090) vs Kaggle 12h ===", flush=True)
    for k, v in walls.items():
        mark = "  <-- cap" if k == cap_name else ""
        print(f"  {k:18s} {v:6.3f}h{mark}", flush=True)
    print(
        f"  Kaggle 12h equivalent = max(...) = {cap_h:.3f}h ({cap_name}). "
        f"6x6 two-pass is known COMPLETE on hidden 12h.",
        flush=True,
    )

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

    b_budget_equiv = leftover_hours(cap_h, a_h)
    b_budget_local = b_budget_equiv
    b_budget_kaggle = leftover_hours(KAGGLE_H - KAGGLE_STOP_MIN / 60.0, a_h)
    n6_h = max((v for k, v in walls.items() if k.startswith("n6x6") and v), default=0.0)
    b_budget_n6 = leftover_hours(n6_h, a_h) if n6_h else 0.0
    print(
        f"\nB leftover budgets: kaggle-equiv({cap_name}) {b_budget_equiv:.2f}h  "
        f"literal_12h-20min {b_budget_kaggle:.2f}h  "
        f"n6x6-max {b_budget_n6:.2f}h (A {a_h:.2f}h)",
        flush=True,
    )

    a_pts_sel = a_pts
    cheap = cheap_order(tasks_all, work)
    mtime_order = sorted(done_b, key=lambda k: done_b[k])

    def eval_keep(kind, keep):
        decoded_b = restrict_decoded(dec_b.decoded_results, set(keep))
        sc = score_pool(replies, dec_a.decoded_results, decoded_b, sel_a)
        b_h = hours_for_subset(keep, dur_b)
        work_all = sum(work.values()) or 1.0
        n_b_unique = sum(1 for t in b_unique if t in keep)
        n_miss = sum(1 for t in a_missing if t in keep)
        row = {
            **sc,
            "kind": kind,
            "n_b_tasks": len(keep),
            "n_b_decoded": sc["n_b_tasks"],
            "b_h": b_h,
            "a_h": a_h,
            "total_h": a_h + b_h,
            "work_frac": sum(work[t] for t in keep) / work_all,
            "b_unique_in": n_b_unique,
            "a_missing_in": n_miss,
            "within_16x16": a_h + b_h <= t16_h + 1e-6,
            "within_cap": a_h + b_h <= cap_h + 1e-6,
            "within_kaggle12": a_h + b_h <= (KAGGLE_H - KAGGLE_STOP_MIN / 60.0) + 1e-6,
            "keys": keep,
        }
        flag = ""
        if row["within_cap"]:
            flag += " [<=cap]"
        if row["within_16x16"] and abs(t16_h - cap_h) > 1e-6:
            flag += " [<=16x16]"
        if row["within_kaggle12"]:
            flag += " [<=literal12]"
        print(fmt_row(row) + f"  B-wins {n_b_unique}/{len(b_unique)}  A-miss {n_miss}/{len(a_missing)}{flag}", flush=True)
        return row

    hour_grid = [
        0.0, 1.0, 2.0, 3.0, 3.5, 4.0, b_budget_kaggle, 5.0,
        b_budget_n6, b_budget_equiv, 6.0, 7.0, b_span,
    ]
    hour_grid = sorted({round(h, 3) for h in hour_grid if h >= 0})

    rows = []
    print("\n=== cheap-first leftover-B + mixed mean_quality (generic policy) ===", flush=True)
    seen = set()
    for hours in hour_grid:
        keep = prefix_until_hours(cheap, dur_b, hours)
        key = ("cheap",) + tuple(keep)
        if key in seen:
            continue
        seen.add(key)
        rows.append(eval_keep("cheap-first", keep))

    print("\n=== expensive-first leftover-B (generic prefix, not skip-head) ===", flush=True)
    seen_e = set()
    for hours in hour_grid:
        keep = prefix_until_hours(exp_order, dur_b, hours)
        key = ("exp",) + tuple(keep)
        if key in seen_e:
            continue
        seen_e.add(key)
        rows.append(eval_keep("expensive-first", keep))

    cheap_local = eval_keep("cheap-first", prefix_until_hours(cheap, dur_b, b_budget_equiv))
    cheap_kaggle = eval_keep("cheap-first", prefix_until_hours(cheap, dur_b, b_budget_kaggle))
    cheap_n6 = (
        eval_keep("cheap-first", prefix_until_hours(cheap, dur_b, b_budget_n6))
        if n6_h and abs(n6_h - cap_h) > 1e-6
        else None
    )
    exp_local = eval_keep("expensive-first", prefix_until_hours(exp_order, dur_b, b_budget_equiv))
    exp_kaggle = eval_keep("expensive-first", prefix_until_hours(exp_order, dur_b, b_budget_kaggle))
    # eval_keep already printed; those may duplicate a grid row. Fine.

    print("\n=== policy (generic: A done, B cheap-first until wall, mixed mean_q) ===", flush=True)
    policy_rows = [
        (f"cheap-first @kaggle-equiv leftover ({cap_name})", cheap_local, cap_h),
        ("cheap-first @literal-12h leftover", cheap_kaggle, KAGGLE_H - KAGGLE_STOP_MIN / 60.0),
        (f"expensive-first @kaggle-equiv leftover ({cap_name})", exp_local, cap_h),
        ("expensive-first @literal-12h leftover", exp_kaggle, KAGGLE_H - KAGGLE_STOP_MIN / 60.0),
    ]
    if cheap_n6 is not None:
        policy_rows.insert(1, ("cheap-first @6x6-two leftover (sensitivity)", cheap_n6, n6_h))
    for name, rec, bud in policy_rows:
        delta = rec["mixed_pct"] - pct(a_sc)
        print(
            f"  {name} cap {bud:.2f}h -> nB={rec['n_b_tasks']} B={rec['b_h']:.2f}h "
            f"total {rec['total_h']:.2f}h mixed {rec['mixed_pct']:.2f} "
            f"({delta:+.2f} vs A) keepP {rec['keepP_pct']:.2f}",
            flush=True,
        )

    boot_cheap_a = bootstrap_delta_pct(a_pts_sel, cheap_local["mixed_pts"])
    boot_cheap_full = bootstrap_delta_pct(full["mixed_pts"], cheap_local["mixed_pts"])
    boot_exp_a = bootstrap_delta_pct(a_pts_sel, exp_local["mixed_pts"])
    boot_score = bootstrap_delta_pct({t: 0.0 for t in cheap_local["mixed_pts"]}, cheap_local["mixed_pts"])
    print("\n=== paired bootstrap on the same 120 public tasks (Kaggle percent) ===", flush=True)
    print(
        f"  cheap@kaggle-equiv mixed {cheap_local['mixed_pct']:.2f}  "
        f"score CI90 [{boot_score['ci90_pct'][0]:.2f}, {boot_score['ci90_pct'][1]:.2f}]",
        flush=True,
    )
    print(
        f"  cheap@kaggle-equiv minus A: {boot_cheap_a['mean_pct']:+.2f}  "
        f"CI90 [{boot_cheap_a['ci90_pct'][0]:+.2f}, {boot_cheap_a['ci90_pct'][1]:+.2f}]  "
        f"P(<=A) {boot_cheap_a['p_lt_0']+boot_cheap_a['p_eq_0']:.3f}",
        flush=True,
    )
    print(
        f"  cheap@kaggle-equiv minus full mixed: {boot_cheap_full['mean_pct']:+.2f}  "
        f"CI90 [{boot_cheap_full['ci90_pct'][0]:+.2f}, {boot_cheap_full['ci90_pct'][1]:+.2f}]",
        flush=True,
    )
    print(
        f"  exp-prefix@kaggle-equiv minus A: {boot_exp_a['mean_pct']:+.2f}  "
        f"CI90 [{boot_exp_a['ci90_pct'][0]:+.2f}, {boot_exp_a['ci90_pct'][1]:+.2f}]",
        flush=True,
    )
    print(
        "  1 Kaggle point = 1.2/120 tasks. 29 vs 30 is ~1 task; 28 vs 33 is not the same kind of gap.",
        flush=True,
    )
    print(
        "  skip-N-expensive is public-eval overfit; not a hidden-set policy.",
        flush=True,
    )

    def strip_row(r):
        return None if not r else {k: v for k, v in r.items() if k not in ("keys", "mixed_pts")}

    rec = cheap_local
    rec_kaggle = cheap_kaggle
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
        "b_unique_expensive_rank": {t: rank_work[t] + 1 for t in b_unique},
        "a_missing": a_missing,
        "kaggle_equiv": {
            "cap_name": cap_name,
            "cap_h": cap_h,
            "walls": walls,
            "leftover_h": b_budget_equiv,
            "n6_max_h": n6_h,
            "n6_leftover_h": b_budget_n6,
            "note": (
                "6x6 two-pass is COMPLETE on hidden Kaggle 12h. Local cap = "
                "max(16x16 pickle span, GPU 6x6 two-pass spans)."
            ),
        },
        "b_budget_local_h": b_budget_equiv,
        "b_budget_equiv_h": b_budget_equiv,
        "b_budget_kaggle_h": b_budget_kaggle,
        "policy": (
            "A cheap-first to completion, leftover-B cheap-first until "
            "max(16x16, 6x6 two-pass), mixed mean_quality"
        ),
        "recommend_local": strip_row(cheap_local),
        "recommend_kaggle": strip_row(cheap_kaggle),
        "recommend_n6": strip_row(cheap_n6),
        "expensive_prefix_local": strip_row(exp_local),
        "expensive_prefix_kaggle": strip_row(exp_kaggle),
        "bootstrap": {
            "cheap16_score_ci90": boot_score["ci90_pct"],
            "cheap16_minus_A": boot_cheap_a,
            "cheap16_minus_full": boot_cheap_full,
            "exp16_minus_A": boot_exp_a,
        },
        "rows": [strip_row(r) for r in rows],
        "ranker": "mixed mean_quality",
        "b_order": "cheap-first estimated_work (generic); expensive-first prefix for comparison only",
        "note": (
            "Do not skip a fitted number of expensive tasks. Hidden eval is a different "
            "120. Day-after: leftover-B --order cheap until kaggle-equiv wall, mixed mean_quality."
        ),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print("wrote", args.out, flush=True)
    if rec and args.keys_out:
        Path(args.keys_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.keys_out).write_text(json.dumps(rec["keys"], indent=2) + "\n")
        print(f"wrote {len(rec['keys'])} B keys ({rec['kind']}) -> {args.keys_out}", flush=True)
    kg_path = Path(str(args.keys_out).replace(".json", "_kaggle.json")) if args.keys_out else None
    if rec_kaggle and kg_path:
        kg_path.write_text(json.dumps(rec_kaggle["keys"], indent=2) + "\n")
        print(f"wrote {len(rec_kaggle['keys'])} Kaggle B keys ({rec_kaggle['kind']}) -> {kg_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
