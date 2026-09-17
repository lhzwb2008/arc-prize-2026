"""Atomically write submission.json from pickle dirs.

Pass-A is --outputs. Later passes are --outputs-extra.
Default pool mode is pass-pair: rank each pass with mean_quality, then
attempt_1 = A top-1, attempt_2 = B top-1 (A top-2 if B is missing/same).

A hard kill cannot tear the JSON: we fsync a temp file, then os.replace.
Does not overwrite an existing scored file with an all-[[0]] result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arc_decoder import (
    ArcDecoder,
    merge_keep_primary,
    merge_pass_pair,
    score_mean_quality,
)
from arc_loader import ArcDataset


def n_nonzero(submission) -> int:
    n = 0
    for attempts in submission.values():
        for a in attempts:
            for g in a.values():
                if g != [[0]]:
                    n += 1
    return n


def atomic_write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(obj)
    with open(tmp, "w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def resolve_pool_mode(keep_primary: bool, pool_mode: str | None) -> str:
    if keep_primary:
        return "keep-primary"
    mode = (pool_mode or os.getenv("NVARC_POOL_MODE") or "pair").strip().lower().replace("_", "-")
    if mode in ("keep", "keep-primary"):
        return "keep-primary"
    if mode in ("mixed", "pool", "mean-quality", "mean_quality"):
        return "mixed"
    return "pair"


def rank_dir(dataset, path, run_name=""):
    dec = ArcDecoder(dataset, n_guesses=2)
    if path and os.path.isdir(path):
        dec.load_decoded_results(path, run_name=run_name)
    n = sum(len(v) for v in dec.decoded_results.values())
    sel = dec.run_selection_algo(score_mean_quality) if dec.decoded_results else {}
    return dec, sel, n


def _merge_decoded(a: dict, b: dict) -> dict:
    out = {}
    for bk in set(a) | set(b):
        out[bk] = {**(a.get(bk) or {}), **(b.get(bk) or {})}
    return out


def build_submission(
    data_path: str,
    primary: str,
    extras: list[str],
    keep_primary: bool = False,
    pool_mode: str | None = None,
):
    data = ArcDataset.from_file(data_path)
    dm = data.split_multi_replies()
    if not os.path.isdir(primary):
        raise SystemExit(f"primary pickle dir missing: {primary}")
    mode = resolve_pool_mode(keep_primary, pool_mode)
    dec_a, sel_a, n_primary = rank_dir(dm, primary)

    extra_paths = []
    for extra in extras or []:
        if extra and os.path.isdir(extra) and any(Path(extra).iterdir()):
            extra_paths.append(extra)

    dec_b = ArcDecoder(dm, n_guesses=2)
    n_extra = 0
    for i, extra in enumerate(extra_paths, 1):
        before = sum(len(v) for v in dec_b.decoded_results.values())
        dec_b.load_decoded_results(extra, run_name=f".p{i}")
        n_extra += sum(len(v) for v in dec_b.decoded_results.values()) - before
    sel_b = dec_b.run_selection_algo(score_mean_quality) if dec_b.decoded_results else {}

    selected = None
    if mode == "pair":
        if sel_a or sel_b:
            selected = merge_pass_pair(sel_a, sel_b)
    else:
        mixed = ArcDecoder(dm, n_guesses=2)
        mixed.decoded_results = _merge_decoded(dec_a.decoded_results, dec_b.decoded_results)
        sel_p = mixed.run_selection_algo(score_mean_quality) if mixed.decoded_results else None
        if mode == "keep-primary" and sel_p is not None:
            selected = merge_keep_primary(sel_a, sel_p)
        else:
            selected = sel_p

    submission = data.get_submission(selected)
    n_decoded = len(set(dec_a.decoded_results) | set(dec_b.decoded_results))
    print(f"pool_mode={mode}", flush=True)
    return submission, n_decoded, n_primary, n_extra


def maybe_write(submission, dest: str) -> bool:
    dest = str(dest)
    new_n = n_nonzero(submission)
    if os.path.exists(dest):
        try:
            old = json.loads(Path(dest).read_text())
            old_n = n_nonzero(old)
        except Exception:
            old_n = 0
        if new_n == 0 and old_n > 0:
            print(f"checkpoint skip: new has 0 nonzero, keep existing {old_n}", flush=True)
            return False
    atomic_write_json(dest, submission)
    print(f"checkpoint wrote {dest} tasks={len(submission)} nonzero={new_n}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="challenges json; inferred from gate.json if omitted")
    ap.add_argument("--outputs", required=True, help="pass-A pickle dir")
    ap.add_argument("--outputs-extra", action="append", default=[], help="later-pass pickle dirs")
    ap.add_argument("--keep-primary", action="store_true",
                    help="legacy: mixed mean_quality then force pass-A top-1")
    ap.add_argument("--pool-mode", default="",
                    help="pair (default), mixed, or keep-primary. pair = A_top1+B_top1")
    ap.add_argument("--submission", default="/kaggle/working/submission.json")
    args = ap.parse_args()

    data_path = args.data
    if not data_path:
        gp = Path("/kaggle/working/gate.json")
        if gp.exists():
            data_path = json.loads(gp.read_text()).get("path") or ""
    if not data_path:
        try:
            from arc_paths import competition_file
            hidden = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
            data_path = competition_file(
                "arc-agi_test_challenges.json" if hidden else "arc-agi_evaluation_challenges.json"
            )
        except Exception as e:
            raise SystemExit(f"need --data ({e})")

    mode = resolve_pool_mode(args.keep_primary, args.pool_mode or None)
    submission, n_decoded, n_primary, n_extra = build_submission(
        data_path, args.outputs, args.outputs_extra,
        keep_primary=args.keep_primary, pool_mode=mode,
    )
    print(
        f"checkpoint decoded={n_decoded} samples_a={n_primary} samples_extra={n_extra} "
        f"pool_mode={mode} keep_primary={args.keep_primary}",
        flush=True,
    )
    if n_decoded == 0:
        print("checkpoint: nothing decoded yet", flush=True)
        return
    maybe_write(submission, args.submission)


def run_live(primary, extras, dest, data_path="", keep_primary=False, pool_mode=None):
    """In-process merge. Same work as CLI, without a transformers relaunch."""
    if not data_path:
        gp = Path("/kaggle/working/gate.json")
        if gp.exists():
            data_path = json.loads(gp.read_text()).get("path") or ""
    if not data_path:
        data_path = os.getenv("NVARC_DATA") or ""
    if not data_path:
        raise RuntimeError("need data_path")
    extras = [p for p in (extras or []) if p]
    mode = resolve_pool_mode(keep_primary, pool_mode)
    submission, n_decoded, n_primary, n_extra = build_submission(
        data_path, primary, extras, keep_primary, pool_mode=mode,
    )
    print(
        f"checkpoint decoded={n_decoded} samples_a={n_primary} samples_extra={n_extra} "
        f"pool_mode={mode} keep_primary={keep_primary}",
        flush=True,
    )
    if n_decoded == 0:
        print("checkpoint: nothing decoded yet", flush=True)
        return False
    return maybe_write(submission, dest)


def live_checkpoint(tag="live"):
    """Notebook starter.py live merge: flock + run_live. Env-driven."""
    primary = os.getenv("NVARC_CHECKPOINT_PRIMARY", "")
    if not primary:
        return
    sub = os.getenv("NVARC_CHECKPOINT_SUB", "")
    if not sub:
        return
    extras = [p for p in os.getenv("NVARC_CHECKPOINT_EXTRAS", "").split(":") if p]
    out = os.getenv("NVARC_OUT", "")
    if out and out != primary and out not in extras:
        extras.append(out)
    try:
        import fcntl
        with open(sub + ".lock", "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                print(f"live checkpoint {tag} primary={primary} extra={extras}", flush=True)
                run_live(primary, extras, sub, os.getenv("NVARC_DATA") or "", False)
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"live checkpoint {tag}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
