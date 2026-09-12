"""Atomically write submission.json from pickle dirs.

Pass-A is --outputs (keep-primary). Later passes are --outputs-extra.
A hard kill cannot tear the JSON: we fsync a temp file, then os.replace.

Does not overwrite an existing scored file with an all-[[0]] result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from arc_decoder import ArcDecoder, score_kgmon
from arc_loader import ArcDataset


def merge_keep_primary(sel_a, sel_p):
    """Pooled ranking, but pass-A top-1 is always one of the two attempts."""
    selected = {}
    n_forced = 0
    for bk in set(sel_a) | set(sel_p):
        a1 = (sel_a.get(bk) or [None])[0]
        top = list(sel_p.get(bk) or [])[:2]
        if a1 is not None and not any(np.array_equal(a1, g) for g in top):
            top = (top[:1] + [a1]) if top else [a1]
            n_forced += 1
        selected[bk] = top
    print(f"keep-primary: forced pass-A top-1 back on {n_forced} outputs", flush=True)
    return selected


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


def build_submission(data_path: str, primary: str, extras: list[str], keep_primary: bool):
    data = ArcDataset.from_file(data_path)
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    if not os.path.isdir(primary):
        raise SystemExit(f"primary pickle dir missing: {primary}")
    decoder.load_decoded_results(primary)
    n_primary = sum(len(v) for v in decoder.decoded_results.values())
    sel_primary = decoder.run_selection_algo(score_kgmon) if keep_primary else None
    n_extra = 0
    for i, extra in enumerate(extras, 1):
        if not extra or not os.path.isdir(extra):
            continue
        if not any(Path(extra).iterdir()):
            continue
        before = sum(len(v) for v in decoder.decoded_results.values())
        decoder.load_decoded_results(extra, run_name=f".p{i}")
        n_extra += sum(len(v) for v in decoder.decoded_results.values()) - before
    selected = decoder.run_selection_algo(score_kgmon) if decoder.decoded_results else None
    if sel_primary is not None and selected is not None:
        selected = merge_keep_primary(sel_primary, selected)
    submission = data.get_submission(selected)
    n_decoded = len(decoder.decoded_results)
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
    ap.add_argument("--outputs", required=True, help="pass-A pickle dir (keep-primary source)")
    ap.add_argument("--outputs-extra", action="append", default=[], help="later-pass pickle dirs")
    ap.add_argument("--keep-primary", action="store_true",
                    help="force pass-A top-1 to remain among the two attempts")
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

    submission, n_decoded, n_primary, n_extra = build_submission(
        data_path, args.outputs, args.outputs_extra, args.keep_primary,
    )
    print(
        f"checkpoint decoded={n_decoded} samples_a={n_primary} samples_extra={n_extra} "
        f"keep_primary={args.keep_primary}",
        flush=True,
    )
    if n_decoded == 0:
        print("checkpoint: nothing decoded yet", flush=True)
        return
    maybe_write(submission, args.submission)


if __name__ == "__main__":
    main()
