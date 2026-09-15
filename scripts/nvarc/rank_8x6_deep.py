#!/usr/bin/env python3
"""CPU: diagnose the 8×6 A+B ranking gap and try richer selectors.

Does not touch the GPU. 8×6 is a view-cut of eval120_half_{a,b}.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from arc_decoder import (
    ArcDecoder,
    hashable,
    score_full_probmul_3,
    score_kgmon,
    score_mean_quality,
    score_sum,
)
from arc_loader import ArcDataset
from finalize import merge_keep_primary
from rank_constraint_filter import apply_filter
from rank_pool_search import RULES, oracle_score, pass_pair_top1, task_score
from score_view_subset import keep_files

DATA = "/opt/data/kaggle/arc-agi_evaluation_challenges.json"
SOL = "/opt/data/kaggle/arc-agi_evaluation_solutions.json"
A_DIR = "/opt/work/nvarc/eval120_half_a/outputs"
B_DIR = "/opt/work/nvarc/eval120_half_b/outputs"
OUT = "/opt/work/nvarc/eval120_search/rank_8x6_deep.json"
N = 120
GEOS = 6


def pct(x: float) -> float:
    return 100.0 * float(x) / N


def load_cut(dm, path: str, geos: int, run_name: str = "") -> dict:
    dec = ArcDecoder(dm, n_guesses=2)
    dec.load_decoded_results(path, run_name=run_name)
    kept = set(keep_files(path, geos, None))
    out = {}
    for bk, samples in dec.decoded_results.items():
        filt = {}
        for k, v in samples.items():
            fn = k.split(".out")[0]
            if run_name and fn.endswith(run_name):
                fn = fn[: -len(run_name)]
            if fn in kept:
                g = dict(v)
                g["_src"] = "B" if run_name else "A"
                g["_key"] = k
                filt[k] = g
        if filt:
            out[bk] = filt
    return out


def merge_decoded(a: dict, b: dict) -> dict:
    out = {}
    for bk in set(a) | set(b):
        out[bk] = {**(a.get(bk) or {}), **(b.get(bk) or {})}
    return out


def _mean_aug(g):
    a = np.asarray(g["score_aug"], dtype=float)
    return float(a.mean()) if a.size else 0.0


def _beam(g):
    return float(g["beam_score"])


def _q(g):
    return -_beam(g) - _mean_aug(g)


def geo_id(key: str) -> str:
    fn = key.split(".out")[0]
    if ".p1" in fn:
        fn = fn.replace(".p1", "")
    if ".permute" in fn:
        return fn.split(".permute")[0]
    return fn


def clusters(samples: dict):
    groups = {}
    for k, g in samples.items():
        try:
            h = hashable(g["solution"])
        except Exception:
            continue
        rec = groups.setdefault(h, {"h": h, "grid": g["solution"], "items": []})
        rec["items"].append((k, g))
    rows = []
    for rec in groups.values():
        items = rec["items"]
        gs = [g for _, g in items]
        srcs = {g.get("_src", "A") for g in gs}
        geos = {geo_id(k) for k, _ in items}
        votes_a = sum(1 for g in gs if g.get("_src") != "B")
        votes_b = sum(1 for g in gs if g.get("_src") == "B")
        qs = [_q(g) for g in gs]
        beams = [_beam(g) for g in gs]
        augs = [_mean_aug(g) for g in gs]
        rows.append(
            {
                "h": rec["h"],
                "grid": rec["grid"],
                "n": len(gs),
                "n_a": votes_a,
                "n_b": votes_b,
                "n_pass": len(srcs),
                "n_geo": len(geos),
                "mean_q": float(np.mean(qs)),
                "best_q": float(np.max(qs)),
                "min_beam": float(np.min(beams)),
                "mean_beam": float(np.mean(beams)),
                "mean_aug": float(np.mean(augs)),
                "min_aug": float(np.min(augs)),
                "shape": tuple(np.shape(rec["grid"])),
                "n_colors": int(len(set(np.asarray(rec["grid"]).ravel().tolist()))),
            }
        )
    tot = sum(r["n"] for r in rows) or 1
    for r in rows:
        r["share"] = r["n"] / tot
    return rows


def order_by(rows, keyfn):
    return [r["grid"] for r in sorted(rows, key=keyfn, reverse=True)]


def score_selected(replies, selected):
    sc, hit, n_out = task_score(replies, selected)
    return sc, hit, n_out


def from_order(decoded, keyfn):
    selected = {}
    for bk, samples in decoded.items():
        selected[bk] = order_by(clusters(samples), keyfn)
    return selected


def rrf_order(decoded, k=60):
    rankers = [
        lambda r: r["mean_q"],
        lambda r: r["n"] - r["mean_aug"],
        lambda r: (r["n_pass"], r["mean_q"]),
        lambda r: (r["n_geo"], r["mean_q"]),
        lambda r: -r["min_beam"],
    ]
    selected = {}
    for bk, samples in decoded.items():
        rows = clusters(samples)
        scores = {r["h"]: 0.0 for r in rows}
        by_h = {r["h"]: r for r in rows}
        for fn in rankers:
            ranked = sorted(rows, key=fn, reverse=True)
            for i, r in enumerate(ranked, 1):
                scores[r["h"]] += 1.0 / (k + i)
        selected[bk] = [
            by_h[h]["grid"] for h, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
    return selected


def both_then_q(r):
    return (r["n_pass"], r["mean_q"], r["n_geo"], r["n"])


def geo_then_q(r):
    return (r["n_geo"], r["mean_q"])


def pass_geo_q(r):
    return (r["n_pass"], r["n_geo"], r["mean_q"])


def votes_q(r):
    return (r["n"], r["mean_q"])


def agree_bonus(r):
    return r["mean_q"] + (1.5 if r["n_pass"] == 2 else 0.0)


def agree_bonus_small(r):
    return r["mean_q"] + (0.4 if r["n_pass"] == 2 else 0.0)


def share_q(r):
    return (r["share"], r["mean_q"])


def lin_combo(w):
    # w: mean_q, votes, both, n_geo, share
    def fn(r):
        return (
            w[0] * r["mean_q"]
            + w[1] * r["n"]
            + w[2] * r["n_pass"]
            + w[3] * r["n_geo"]
            + w[4] * r["share"]
        )

    return fn


def zscore_rows(rows, keys):
    stats = {}
    for k in keys:
        xs = np.array([r[k] for r in rows], dtype=float)
        mu, sd = float(xs.mean()), float(xs.std()) or 1.0
        stats[k] = (mu, sd)
    out = []
    for r in rows:
        z = dict(r)
        for k, (mu, sd) in stats.items():
            z[f"z_{k}"] = (r[k] - mu) / sd
        out.append(z)
    return out


def within_output_z(decoded, keyfn_from_z):
    selected = {}
    for bk, samples in decoded.items():
        rows = zscore_rows(clusters(samples), ["mean_q", "n", "n_pass", "n_geo", "share", "mean_aug", "min_beam"])
        selected[bk] = order_by(rows, keyfn_from_z)
    return selected


def gold_rank_table(replies, decoded):
    hist = Counter()
    misses = []
    n_oracle = n_hit = n_out = 0
    for bk, golds in replies.items():
        n_out += 1
        hg = hashable(golds[0])
        rows = sorted(clusters(decoded.get(bk) or {}), key=lambda r: r["mean_q"], reverse=True)
        rank = None
        gold_row = None
        for i, r in enumerate(rows, 1):
            if r["h"] == hg:
                rank = i
                gold_row = r
                break
        if rank is None:
            hist["absent"] += 1
            continue
        n_oracle += 1
        bucket = str(rank) if rank <= 8 else "9+"
        hist[bucket] += 1
        if rank <= 2:
            n_hit += 1
            continue
        top = rows[:2]
        misses.append(
            {
                "bk": bk,
                "rank": rank,
                "n_cand": len(rows),
                "gold": {
                    "n": gold_row["n"],
                    "n_a": gold_row["n_a"],
                    "n_b": gold_row["n_b"],
                    "n_pass": gold_row["n_pass"],
                    "n_geo": gold_row["n_geo"],
                    "mean_q": gold_row["mean_q"],
                    "share": gold_row["share"],
                    "shape": list(gold_row["shape"]),
                },
                "top1": {
                    "n": top[0]["n"],
                    "n_pass": top[0]["n_pass"],
                    "n_geo": top[0]["n_geo"],
                    "mean_q": top[0]["mean_q"],
                    "share": top[0]["share"],
                    "shape": list(top[0]["shape"]),
                },
                "d_q": gold_row["mean_q"] - top[0]["mean_q"],
                "d_votes": gold_row["n"] - top[0]["n"],
            }
        )
    return dict(hist), misses, n_oracle, n_hit, n_out


def even_odd_score(replies, selected):
    even, odd = {}, {}
    for bk, guesses in selected.items():
        task = bk.split("_")[0]
        (even if int(task[:2], 16) % 2 == 0 else odd)[bk] = guesses
    def sub(sel):
        # score only on tasks that appear in this split, but task_score uses all replies.
        # Restrict replies.
        keys = {bk.split("_")[0] for bk in sel}
        sub_replies = {bk: v for bk, v in replies.items() if bk.split("_")[0] in keys}
        sc, hit, n_out = task_score(sub_replies, sel)
        n_tasks = len({bk.split("_")[0] for bk in sub_replies})
        return sc, n_tasks, hit, n_out
    return {"even": sub(even), "odd": sub(odd)}


def loocv_linear(replies, decoded):
    """Leave-one-task-out: fit 5 weights by least squares on gold vs mean of rest."""
    tasks = sorted({bk.split("_")[0] for bk in replies})
    feats_keys = ["mean_q", "n", "n_pass", "n_geo", "share"]
    per_bk = {}
    for bk, samples in decoded.items():
        per_bk[bk] = clusters(samples)

    def feat(r):
        return np.array([r[k] for k in feats_keys], dtype=float)

    # global standardize using all rows
    all_x = []
    for rows in per_bk.values():
        for r in rows:
            all_x.append(feat(r))
    all_x = np.stack(all_x)
    mu, sd = all_x.mean(0), all_x.std(0)
    sd[sd == 0] = 1.0

    def z(r):
        return (feat(r) - mu) / sd

    # Build design from other tasks: positive = gold cluster, negative = best non-gold
    by_task = defaultdict(list)
    for bk, rows in per_bk.items():
        by_task[bk.split("_")[0]].append((bk, rows))

    selected = {}
    ws = []
    for hold in tasks:
        xs, ys = [], []
        for t, items in by_task.items():
            if t == hold:
                continue
            for bk, rows in items:
                if bk not in replies:
                    continue
                hg = hashable(replies[bk][0])
                golds = [r for r in rows if r["h"] == hg]
                rest = [r for r in rows if r["h"] != hg]
                if not golds or not rest:
                    continue
                xs.append(z(golds[0]))
                ys.append(1.0)
                # hardest negative = highest mean_q distractor
                bad = max(rest, key=lambda r: r["mean_q"])
                xs.append(z(bad))
                ys.append(0.0)
        if len(xs) < 10:
            w = np.array([1.0, 0, 0, 0, 0])
        else:
            X = np.stack(xs)
            y = np.array(ys)
            # ridge
            Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
            reg = 0.5 * np.eye(Xb.shape[1])
            w_full = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
            w = w_full[:-1]
        ws.append(w)
        for bk, rows in by_task[hold]:
            selected[bk] = order_by(rows, lambda r, w=w: float(z(r) @ w))
    w_mean = np.mean(np.stack(ws), axis=0).tolist()
    return selected, dict(zip(feats_keys, w_mean))


def grid_search_holdout(replies, decoded):
    """Tune 3 coefficients on even tasks, eval odd (and reverse)."""
    even_tasks = {bk.split("_")[0] for bk in replies if int(bk.split("_")[0][:2], 16) % 2 == 0}

    def split_sel(selected, even=True):
        keys = even_tasks if even else {bk.split("_")[0] for bk in replies} - even_tasks
        sub_replies = {bk: v for bk, v in replies.items() if bk.split("_")[0] in keys}
        sub_sel = {bk: v for bk, v in selected.items() if bk.split("_")[0] in keys}
        sc, hit, n_out = task_score(sub_replies, sub_sel)
        n_tasks = len({bk.split("_")[0] for bk in sub_replies})
        return sc, n_tasks, hit

    grid = []
    for a in (0.5, 1.0, 1.5, 2.0):
        for b in (0.0, 0.05, 0.1, 0.2, 0.4):
            for c in (0.0, 0.2, 0.5, 1.0, 1.5):
                fn = lin_combo((a, b, c, 0.0, 0.0))
                sel = from_order(decoded, fn)
                tr, ntr, _ = split_sel(sel, even=True)
                te, nte, _ = split_sel(sel, even=False)
                full, _, _ = task_score(replies, sel)
                grid.append(
                    {
                        "w": [a, b, c, 0, 0],
                        "even": pct(tr * N / ntr) if ntr else 0,
                        "odd": pct(te * N / nte) if nte else 0,
                        "full": pct(full),
                        "train_sc": tr,
                    }
                )
    grid.sort(key=lambda d: (d["even"], d["full"]), reverse=True)
    best = grid[0]
    # pick by even, report odd
    return best, grid[:8]


def main():
    ch = json.load(open(DATA))
    data = ArcDataset.from_file(DATA)
    data.load_replies(SOL)
    dm = data.split_multi_replies()
    replies = dm.replies

    print("loading 8x6 cuts...", flush=True)
    dec_a = load_cut(dm, A_DIR, GEOS, "")
    dec_b = load_cut(dm, B_DIR, GEOS, ".p1")
    pooled = merge_decoded(dec_a, dec_b)

    ora, in_pool, n_out = oracle_score(replies, pooled)
    print(f"oracle {ora:.3f}/{N} = {pct(ora):.2f}%  gold_in_pool {in_pool}/{n_out}", flush=True)

    hist, misses, n_oracle, n_hit, _ = gold_rank_table(replies, pooled)
    print("gold rank under mean_q (outputs):", dict(hist), flush=True)
    print(f"  oracle_outputs {n_oracle}  mean_q_top2 {n_hit}  rank_miss {len(misses)}", flush=True)

    rows_out = []

    def add(name, selected, kind="rule"):
        sc, hit, n_out2 = score_selected(replies, selected)
        rec = {"name": name, "kind": kind, "score": sc, "pct": pct(sc), "hit": hit, "n_out": n_out2}
        rows_out.append(rec)
        print(f"{pct(sc):6.2f}  {sc:7.3f}  top2 {hit:3d}/{n_out2}  {name}", flush=True)
        return rec

    # existing pooled rules
    dec = ArcDecoder(dm, n_guesses=2)
    dec.decoded_results = pooled
    for name in ("mean_quality", "kgmon", "probmul3", "votes", "votes_then_aug", "vote_x_quality", "logvotes_minus_aug"):
        add(name, dec.run_selection_algo(RULES[name]), "pool")

    sel_mq = dec.run_selection_algo(score_mean_quality)
    sel_kg = dec.run_selection_algo(score_kgmon)

    decA = ArcDecoder(dm, n_guesses=2)
    decA.decoded_results = dec_a
    decB = ArcDecoder(dm, n_guesses=2)
    decB.decoded_results = dec_b
    sel_a_mq = decA.run_selection_algo(score_mean_quality)
    sel_b_mq = decB.run_selection_algo(score_mean_quality)
    sel_a_kg = decA.run_selection_algo(score_kgmon)
    sel_b_kg = decB.run_selection_algo(score_kgmon)

    add("keepP+mean_q", merge_keep_primary(sel_a_mq, sel_mq), "keep")
    add("keepP+kgmon", merge_keep_primary(sel_a_kg, sel_kg), "keep")
    add("A_top1+B_top1 mq", pass_pair_top1(sel_a_mq, sel_b_mq), "pair")
    add("A_top1+B_top1 kg", pass_pair_top1(sel_a_kg, sel_b_kg), "pair")

    add("both_then_q", from_order(pooled, both_then_q), "struct")
    add("geo_then_q", from_order(pooled, geo_then_q), "struct")
    add("pass_geo_q", from_order(pooled, pass_geo_q), "struct")
    add("votes_then_q", from_order(pooled, votes_q), "struct")
    add("agree_bonus_1.5", from_order(pooled, agree_bonus), "struct")
    add("agree_bonus_0.4", from_order(pooled, agree_bonus_small), "struct")
    add("share_then_q", from_order(pooled, share_q), "struct")
    add("RRF_5", rrf_order(pooled), "ensemble")

    add(
        "z_q+0.3z_pass+0.2z_geo",
        within_output_z(
            pooled,
            lambda r: r["z_mean_q"] + 0.3 * r["z_n_pass"] + 0.2 * r["z_n_geo"],
        ),
        "struct",
    )

    # constraint filter then mean_q
    tmp = ArcDecoder(dm, n_guesses=2)
    tmp.decoded_results = pooled
    for mode in ("shape", "col_all", "shape+col"):
        filtered, dropped, kept, drop_gold = apply_filter(ch, replies, tmp, mode, fallback=True)
        d2 = ArcDecoder(dm, n_guesses=2)
        d2.decoded_results = filtered
        rec = add(f"filter_{mode}+mq", d2.run_selection_algo(score_mean_quality), "filter")
        rec["dropped"] = dropped
        rec["gold_dropped"] = drop_gold

    print("\nLOOCV linear...", flush=True)
    sel_loo, w_mean = loocv_linear(replies, pooled)
    rec = add("LOOCV_linear", sel_loo, "learned")
    rec["w_mean"] = w_mean

    print("\nholdout grid (fit even, the printed full is optimistic if we pick by full)...", flush=True)
    best, top = grid_search_holdout(replies, pooled)
    print("best-on-even", best, flush=True)
    # evaluate that exact weight on full already in best['full'], odd in best['odd']
    add(f"tune_even w={best['w'][:3]}", from_order(pooled, lin_combo(tuple(best["w"]))), "learned")

    # miss analysis summaries
    miss_rank = Counter(str(m["rank"] if m["rank"] <= 6 else "7+") for m in misses)
    gold_both = sum(1 for m in misses if m["gold"]["n_pass"] == 2)
    gold_only = sum(1 for m in misses if m["gold"]["n_pass"] == 1)
    q_better = sum(1 for m in misses if m["d_q"] > 0)
    votes_better = sum(1 for m in misses if m["d_votes"] > 0)
    print("\nmiss summary", {
        "n": len(misses),
        "rank": dict(miss_rank),
        "gold_in_both_passes": gold_both,
        "gold_in_one_pass": gold_only,
        "gold_higher_mean_q_than_top1": q_better,
        "gold_more_votes_than_top1": votes_better,
    }, flush=True)

    rows_out.sort(key=lambda r: r["score"], reverse=True)
    print("\n=== ranked ===", flush=True)
    for r in rows_out[:15]:
        print(f"  {r['pct']:5.2f}  {r['name']}", flush=True)

    doc = {
        "oracle": ora,
        "oracle_pct": pct(ora),
        "gold_in_pool": in_pool,
        "n_out": n_out,
        "mean_q_rank_hist": hist,
        "n_rank_miss": len(misses),
        "misses": misses,
        "rules": rows_out,
        "loocv_w": w_mean,
        "tune_even": best,
        "tune_even_top": top,
        "need_35": 35.0,
        "need_39": pct(ora),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
