#!/usr/bin/env python3
"""Push the NVARC kernel (single-pass 8×6 notebook) and record version_number."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "notebooks" / "nvarc_2026"
OUT = Path(os.environ.get("NVARC_SINGLE_KERNEL", "/opt/work/nvarc/single8x6_kernel.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(FOLDER))
    ap.add_argument("--accelerator", default="NvidiaL4")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    folder = Path(args.folder)
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    nb = json.loads((folder / meta["code_file"]).read_text())
    text = "\n".join("".join(c.get("source") or []) for c in nb["cells"])
    if 'SCHEDULE = "single"' not in text:
        raise SystemExit("FAIL: notebook SCHEDULE is not single (refusing to push)")
    if "ARC_N_TRAIN_AUG\"] = \"8\"" not in text and "ARC_N_TRAIN_AUG'] = '8'" not in text:
        if 'COMMON["ARC_N_TRAIN_AUG"] = "8"' not in text:
            raise SystemExit("FAIL: notebook is not 8 train-aug")
    print(f"notebook ok: SCHEDULE=single 8×6  folder={folder}", flush=True)
    if args.dry_run:
        print("dry-run: not pushing")
        return 0

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    resp = api.kernels_push(str(folder), acc=args.accelerator)
    payload = resp.to_dict() if hasattr(resp, "to_dict") else {
        "ref": getattr(resp, "ref", None),
        "url": getattr(resp, "url", None),
        "version_number": getattr(resp, "version_number", None),
        "error": getattr(resp, "error", None),
    }
    print(json.dumps(payload, indent=2, default=str), flush=True)
    if payload.get("error"):
        raise SystemExit(f"FAIL push error: {payload['error']}")
    version = payload.get("version_number")
    if version is None:
        raise SystemExit("FAIL: push returned no version_number")
    doc = {
        "kernel": meta.get("id"),
        "version_number": int(version),
        "url": payload.get("url"),
        "ref": payload.get("ref"),
        "pushed_utc": datetime.now(timezone.utc).isoformat(),
        "accelerator": args.accelerator,
        "schedule": "single",
        "recipe": "8x6",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT} version={version}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
