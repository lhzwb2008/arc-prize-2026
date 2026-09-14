#!/usr/bin/env python3
"""CPU: drop candidates that violate train shape/color constraints."""
from __future__ import annotations

import json

import numpy as np

from arc_decoder import ArcDecoder, hashable, score_mean_quality
from arc_loader import ArcDataset
from rank_pool_search import oracle_score, task_score

DATA = "/opt/data/kaggle/arc-agi_evaluation_challenges.json"
SOL = "/opt/data/kaggle/arc-agi_evaluation_solutions.json"
W = "/opt/work/nvarc"


def train_stats(ch, task):
    pairs = ch[task]["train"]
    in_sh = [tuple(np.shape(p["input"])) for p in pairs]
    out_sh = [tuple(np.shape(p["output"])) for p in pairs]
    same_out = len(set(out_sh)) == 1
    ratio = {
        (
            o[0] / i[0] if i[0] else None,
            o[1] / i[1] if i[1] else None,
        )
        for i, o in zip(in_sh, out_sh)
    }
    same_ratio = len(ratio) == 1
    delta = {(o[0] - i[0], o[1] - i[1]) for i, o in zip(in_sh, out_sh)}
    same_delta = len(delta) == 1
    cols_out, cols_all = set(), set()
    for p in pairs:
        a = np.asarray(p["input"])
        b = np.asarray(p["output"])
        cols_out |= set(b.ravel().tolist())
        cols_all |= set(a.ravel().tolist()) | set(b.ravel().tolist())
    return {
        "out_sh": out_sh[0] if same_out else None,
        "ratio": next(iter(ratio)) if same_ratio else None,
        "delta": next(iter(delta)) if same_delta else None,
        "cols_out": cols_out,
        "cols_all": cols_all,
        "same_hw": all(i == o for i, o in zip(in_sh, out_sh)),
    }


def pred_shape(ch, task, test_in):
    st = train_stats(ch, task)
    th, tw = np.shape(test_in)
    cands = set()
    if st["out_sh"] is not None:
        cands.add(st["out_sh"])
    if st["ratio"] is not None and None not in st["ratio"]:
        rh, rw = st["ratio"]
        if rh == int(rh) and rw == int(rw) and rh >= 1 and rw >= 1:
            cands.add((int(th * rh), int(tw * rw)))
    if st["delta"] is not None:
        dh, dw = st["delta"]
        hh, ww = th + dh, tw + dw
        if hh >= 1 and ww >= 1:
            cands.add((hh, ww))
    if st["same_hw"]:
        cands.add((th, tw))
    return cands, st


def ok_sample(mode, g, allowed, st):
    sh = tuple(np.shape(g["solution"]))
    cols = set(np.asarray(g["solution"]).ravel().tolist())
    if mode in ("shape", "shape+col") and allowed:
        if sh not in allowed:
            return False
    if mode == "col_out" and not cols <= st["cols_out"]:
        return False
    if mode in ("col_all", "shape+col") and not cols <= st["cols_all"]:
        return False
    return True


def apply_filter(ch, replies, dec, mode, fallback=True):
    new = {}
    dropped = kept = drop_gold = 0
    for bk, samples in dec.decoded_results.items():
        task, idx = bk.split("_")
        test_in = ch[task]["test"][int(idx)]["input"]
        allowed, st = pred_shape(ch, task, test_in)
        gold = hashable(replies[bk][0])
        dest = {}
        for k, g in samples.items():
            if ok_sample(mode, g, allowed, st):
                dest[k] = g
                kept += 1
            else:
                dropped += 1
                try:
                    if hashable(g["solution"]) == gold:
                        drop_gold += 1
                except Exception:
                    pass
        if dest or not fallback:
            new[bk] = dest
        else:
            new[bk] = samples
            kept += len(samples)
    return new, dropped, kept, drop_gold


def main():
    ch = json.load(open(DATA))
    data = ArcDataset.from_file(DATA)
    data.load_replies(SOL)
    dm = data.split_multi_replies()
    replies = dm.replies
    dec = ArcDecoder(dm, n_guesses=2)
    dec.load_decoded_results(f"{W}/eval120_n6x6_a/outputs")
    dec.load_decoded_results(f"{W}/eval120_n6x6_b/outputs", run_name=".p1")

    killed_shape = killed_col_out = killed_col_all = 0
    n_shape = 0
    for task, obj in ch.items():
        st = train_stats(ch, task)
        for i, t in enumerate(obj["test"]):
            bk = f"{task}_{i}"
            gold = np.asarray(replies[bk][0])
            allowed, _ = pred_shape(ch, task, t["input"])
            if allowed:
                n_shape += 1
                if tuple(gold.shape) not in allowed:
                    killed_shape += 1
            cout = set(gold.ravel().tolist())
            if not cout <= st["cols_out"]:
                killed_col_out += 1
            if not cout <= st["cols_all"]:
                killed_col_all += 1
    print("gold safety (strict filter would drop the gold grid)")
    print(f"  shape-applicable {n_shape}/172  gold_shape_miss {killed_shape}")
    print(f"  gold not subset train-out colors {killed_col_out}/172")
    print(f"  gold not subset train in+out colors {killed_col_all}/172")

    base = dec.run_selection_algo(score_mean_quality)
    sc, _, _ = task_score(replies, base)
    ora, _, _ = oracle_score(replies, dec.decoded_results)
    print(f"\nbaseline mean_q {sc:.3f} ({100 * sc / 120:.2f}%) oracle {ora:.3f}")

    for fallback in (True, False):
        print(f"\nfallback={fallback} (empty pool {'keeps all' if fallback else 'scores 0'})")
        for mode in ("shape", "col_out", "col_all", "shape+col"):
            filtered, dropped, kept, drop_gold = apply_filter(
                ch, replies, dec, mode, fallback=fallback
            )
            d2 = ArcDecoder(dm, n_guesses=2)
            d2.decoded_results = filtered
            sel = d2.run_selection_algo(score_mean_quality)
            sc2, hit, _ = task_score(replies, sel)
            ora2, _, _ = oracle_score(replies, filtered)
            empty = sum(1 for v in filtered.values() if not v)
            print(
                f"  {mode:10} mean_q {sc2:.3f} ({100 * sc2 / 120:.2f}%) oracle {ora2:.3f} "
                f"drop {dropped} keep {kept} gold_dropped {drop_gold} empty {empty}"
            )


if __name__ == "__main__":
    main()
