#!/usr/bin/env python3
"""Rewrite the NVARC notebook driver/starter for leftover-B uncertainty/cost."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks/nvarc_2026/nvarc_qwen3_4b_ttt_2026.ipynb"
QUEUE = ROOT / "scripts/nvarc/b_priority_queue.py"

MD = """# ARC Prize 2026 — NVARC 8×6 leftover-B (v17)

Fork of Ivan Sorokin's notebook (`sorokin/qwen3_4b_grids15_sft139`). **Two** 8 train-aug / 6 decode-view LoRA TTT passes.

**Pass A** cheap-first (seed 42). **Pass B** leftover until 12h−20min, queued by pass-A uncertainty per unit cost: `(1 - A top-1 vote share) / estimated_work`. No task ids. Pooling is **keep-primary**. Live rewrite only when pickles change and ≥5 min since last write; hard write at 12h−5min.

**v17.** Driver `SCHEDULE` is leftover. Save Version smoke is still A-only (4 eval keys). Hidden rerun does A then leftover-B.
"""

DRIVER = r'''import os, sys, time, json, subprocess, threading, fcntl
from pathlib import Path

# v17: 8x6 leftover-B. A cheap-first; B ordered by A-uncertainty/cost.
# Save Version (not hidden): A-only 4-key smoke. Hidden: A then leftover-B.
# Live rewrite gated (pickle change + 5min gap). Hard write at 12h-5min.
SCHEDULE = "leftover"

COMMON = {
    "UNSLOTH_DISABLE_STATISTICS": "1",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "OMP_NUM_THREADS": "12",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "ARC_N_EVAL_AUG": "1",
    "NVARC_TASK_LIMIT": "0",
    "NVARC_DFS_LIMIT": "0",
    "NVARC_CHECKPOINT_EVERY": "90",
    "NVARC_CHECKPOINT_MIN_GAP": "300",
    "NVARC_CHECKPOINT_SUB": "/kaggle/working/submission.json",
    "NVARC_HARD_MERGE_TIME": str(hard_merge_time),
    "NVARC_POOL_MODE": "keep-primary",
}
COMMON["ARC_N_TRAIN_AUG"] = "8"
COMMON["ARC_N_EVAL_GEOS"] = "6"
os.environ.update(COMMON)
print(
    f"RECIPE v17 n_train={os.environ['ARC_N_TRAIN_AUG']} geos={os.environ['ARC_N_EVAL_GEOS']} "
    f"eval_aug={os.environ['ARC_N_EVAL_AUG']} schedule={SCHEDULE} ranker=keep-primary "
    f"b_order=A-uncertainty/cost hard_merge_in={(hard_merge_time-time.time())/60:.1f}min "
    f"ckpt_min_gap={os.environ['NVARC_CHECKPOINT_MIN_GAP']}s",
    flush=True,
)

SEED_A = dict(ARC_LORA_SEED=42, ARC_TRAIN_AUG_SEED=1, ARC_EVAL_AUG_SEED=2, ARC_SCORE_SEED_OFFSET=0)
SEED_B = dict(ARC_LORA_SEED=137, ARC_TRAIN_AUG_SEED=17, ARC_EVAL_AUG_SEED=29, ARC_SCORE_SEED_OFFSET=7)

PASSES = [
    dict(name="A", out="/kaggle/inference_outputs", seeds=SEED_A, order="cheap"),
    dict(name="B", out="/kaggle/inference_outputs_b", seeds=SEED_B, order="file"),
]
RESERVE_PER_LATER = 0

SUB = "/kaggle/working/submission.json"
PRIMARY = PASSES[0]["out"]


def remaining():
    return global_end_time - time.time()


def done_keys(out_dir):
    p = Path(out_dir)
    if not p.exists():
        return set()
    return {fn.split("_", 1)[0] for fn in os.listdir(p) if "_" in fn}


def gate_hidden():
    gp = Path("/kaggle/working/gate.json")
    if gp.exists():
        try:
            return bool(json.loads(gp.read_text()).get("hidden"))
        except Exception:
            return False
    return bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))


def gate_data_path():
    gp = Path("/kaggle/working/gate.json")
    if gp.exists():
        try:
            return json.loads(gp.read_text()).get("path") or ""
        except Exception:
            return ""
    return ""


def pool_dirs():
    dirs = [PRIMARY]
    for spec in PASSES[1:]:
        p = Path(spec["out"])
        if p.exists() and any(p.iterdir()):
            dirs.append(spec["out"])
    return dirs


def checkpoint(dirs, tag):
    cmd = [sys.executable, "nvarc_checkpoint.py",
           "--outputs", dirs[0], "--submission", SUB]
    data_path = gate_data_path()
    if data_path:
        cmd.extend(["--data", data_path])
    for extra in dirs[1:]:
        if Path(extra).exists() and any(Path(extra).iterdir()):
            cmd.extend(["--outputs-extra", extra])
    print(f"[checkpoint {tag}] {' '.join(cmd)}", flush=True)
    lock_path = SUB + ".lock"
    with open(lock_path, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            rc = subprocess.call(cmd)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    print(f"[checkpoint {tag}] rc={rc} remaining={remaining()/60:.1f} min exists={Path(SUB).exists()}", flush=True)
    return rc


def run_pass(spec, end_time, keys_file="", skip_done=False, extra_dirs=None):
    name, out_dir = spec["name"], spec["out"]
    if remaining() < 15 * 60:
        print(f"[pass {name}] skipped, remaining={remaining()/60:.1f} min", flush=True)
        return 0
    env = dict(os.environ)
    env.update({k: str(v) for k, v in spec["seeds"].items()})
    env["NVARC_CHECKPOINT_PRIMARY"] = PRIMARY
    extras = [d for d in (extra_dirs or []) if d != PRIMARY]
    env["NVARC_CHECKPOINT_EXTRAS"] = ":".join(extras)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "starter.py",
        "--end-time", str(end_time),
        "--out", out_dir,
        "--order", spec["order"],
    ]
    if keys_file:
        cmd.extend(["--keys-file", keys_file])
    if skip_done:
        cmd.append("--skip-done")
    print(f"[pass {name}] start remaining={remaining()/60:.1f} min out={out_dir} order={spec['order']} extra={spec['seeds']}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    nfiles = len(list(Path(out_dir).glob("*"))) if Path(out_dir).exists() else 0
    print(f"[pass {name}] rc={rc} took={(time.time()-t0)/60:.1f} min files={nfiles} remaining={remaining()/60:.1f} min", flush=True)
    print(f"starter_exit={rc}", flush=True)
    return rc


def build_b_queue(out_dir):
    keys_file = "/kaggle/working/b_priority_keys.json"
    data_path = gate_data_path()
    cmd = [
        sys.executable, "b_priority_queue.py",
        "--data", data_path,
        "--outputs", PRIMARY,
        "--skip-done", out_dir,
        "--keys-out", keys_file,
    ]
    print("[b-queue]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0 or not Path(keys_file).exists():
        print(f"[b-queue] failed rc={rc}; B falls back to cheap-first", flush=True)
        return "", "cheap"
    n = len(json.loads(Path(keys_file).read_text()))
    print(f"[b-queue] {n} keys -> {keys_file}", flush=True)
    return keys_file, "file"


def _hard_merge_watchdog():
    while time.time() < hard_merge_time:
        time.sleep(5)
    print("[hard-merge] 12h-5min reached; writing keep-primary pool", flush=True)
    try:
        checkpoint(pool_dirs(), "hard-Tminus5")
    except Exception as e:
        print(f"[hard-merge] {type(e).__name__}: {e}", flush=True)


threading.Thread(target=_hard_merge_watchdog, daemon=True).start()

done_dirs = []
for i, spec in enumerate(PASSES):
    later = len(PASSES) - i - 1
    end_time = global_end_time - later * RESERVE_PER_LATER
    if end_time < time.time() + 20 * 60:
        end_time = global_end_time
    keys_file = ""
    spec = dict(spec)
    if spec["name"] == "B":
        if not gate_hidden():
            print("[pass B] Save Version smoke: skip leftover-B", flush=True)
            continue
        keys_file, order = build_b_queue(spec["out"])
        spec["order"] = order
    rc = run_pass(spec, end_time, keys_file=keys_file, extra_dirs=done_dirs + [spec["out"]])
    done_dirs.append(spec["out"])
    if spec["name"] == "A":
        qp = Path("/kaggle/working/queued_keys.json")
        queued = json.loads(qp.read_text()) if qp.exists() else []
        unprocessed = [k for k in queued if k not in done_keys(PRIMARY)]
        print(f"[catchup] A reached {len(queued)-len(unprocessed)}/{len(queued)} unprocessed={len(unprocessed)}", flush=True)
        if unprocessed:
            ckeys = "/kaggle/working/keys_catchup.json"
            json.dump(unprocessed, open(ckeys, "w"))
            catchup = dict(name="catchup", out=PRIMARY, seeds=SEED_A, order="file")
            run_pass(catchup, global_end_time, keys_file=ckeys, skip_done=True, extra_dirs=[PRIMARY])
    checkpoint(done_dirs, f"after-{spec['name']}")
    print(f"pass-{spec['name']} leftover files={len(list(Path(spec['out']).glob('*'))) if Path(spec['out']).exists() else 0}", flush=True)

print(
    f"passes done remaining={remaining()/60:.1f} min sub={Path(SUB).exists()} "
    f"wall_h={(time.time()-T0)/3600:.3f} schedule={SCHEDULE}",
    flush=True,
)
'''

STARTER_CKPT = '''
def pickle_fingerprint(dirs):
    n = 0
    newest = 0.0
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        with os.scandir(d) as it:
            for e in it:
                if e.is_file():
                    n += 1
                    m = e.stat().st_mtime
                    if m > newest:
                        newest = m
    return (n, newest)


_CKPT_STATE = {"fp": None, "t": 0.0}


def checkpoint_due(dirs, now, min_gap_s, force=False):
    fp = pickle_fingerprint(dirs)
    if force:
        _CKPT_STATE.update(fp=fp, t=now)
        return True
    if fp == _CKPT_STATE["fp"]:
        return False
    if now - _CKPT_STATE["t"] < min_gap_s:
        return False
    _CKPT_STATE.update(fp=fp, t=now)
    return True


def live_checkpoint(tag="live", force=True):
    primary = os.getenv("NVARC_CHECKPOINT_PRIMARY", "")
    if not primary:
        return
    sub = os.getenv("NVARC_CHECKPOINT_SUB", "/kaggle/working/submission.json")
    extras = [p for p in os.getenv("NVARC_CHECKPOINT_EXTRAS", "").split(":") if p]
    out = os.getenv("NVARC_OUT", "")
    if out and out != primary and out not in extras:
        extras.append(out)
    min_gap = float(os.getenv("NVARC_CHECKPOINT_MIN_GAP", "300"))
    if not checkpoint_due([primary] + extras, time.time(), min_gap, force=force):
        return
    try:
        import fcntl
        from nvarc_checkpoint import run_live
        with open(sub + ".lock", "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                print(f"live checkpoint {tag} primary={primary} extra={extras}", flush=True)
                run_live(primary, extras, sub, os.getenv("NVARC_DATA") or "", False)
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"live checkpoint {tag}: {type(e).__name__}: {e}", flush=True)
'''


def as_source(text: str):
    if not text.endswith("\n"):
        text += "\n"
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]]


def cell_src(cell) -> str:
    return "".join(cell.get("source") or [])


def find_cell(nb, needle: str, cell_type: str | None = None) -> int:
    for i, cell in enumerate(nb["cells"]):
        if cell_type and cell.get("cell_type") != cell_type:
            continue
        if needle in cell_src(cell):
            return i
    raise SystemExit(f"cell not found: {needle[:80]!r} type={cell_type}")


def main():
    nb = json.loads(NB.read_text())
    starter_i = find_cell(nb, "%%writefile starter.py")
    starter = cell_src(nb["cells"][starter_i])
    old = '''def live_checkpoint(tag="live"):
    primary = os.getenv("NVARC_CHECKPOINT_PRIMARY", "")
    if not primary:
        return
    sub = os.getenv("NVARC_CHECKPOINT_SUB", "/kaggle/working/submission.json")
    extras = [p for p in os.getenv("NVARC_CHECKPOINT_EXTRAS", "").split(":") if p]
    out = os.getenv("NVARC_OUT", "")
    if out and out != primary and out not in extras:
        extras.append(out)
    try:
        import fcntl
        from nvarc_checkpoint import run_live
        with open(sub + ".lock", "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                print(f"live checkpoint {tag} primary={primary} extra={extras}", flush=True)
                run_live(primary, extras, sub, os.getenv("NVARC_DATA") or "", False)
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"live checkpoint {tag}: {type(e).__name__}: {e}", flush=True)
'''
    if "def pickle_fingerprint" in starter and "checkpoint_due(" in starter:
        print("starter already has gated checkpoints")
    elif old not in starter:
        raise SystemExit("starter live_checkpoint block not found")
    else:
        starter = starter.replace(old, STARTER_CKPT.lstrip("\n"))
        starter = starter.replace(
            "            live_checkpoint(\"tick\")\n",
            "            live_checkpoint(\"tick\", force=False)\n",
        )
        nb["cells"][starter_i]["source"] = as_source(starter)

    md_i = find_cell(nb, "# ARC Prize 2026", cell_type="markdown")
    nb["cells"][md_i]["source"] = as_source(MD)

    already_queue = any("%%writefile b_priority_queue.py" in cell_src(c) for c in nb["cells"])
    driver_i = find_cell(nb, "PASSES =", cell_type="code")
    if "%%writefile" in cell_src(nb["cells"][driver_i]):
        raise SystemExit("driver cell lookup hit a writefile cell")
    if not already_queue:
        queue_cell = {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "source": as_source("%%writefile b_priority_queue.py\n" + QUEUE.read_text()),
            "execution_count": None,
            "outputs": [],
        }
        nb["cells"].insert(driver_i, queue_cell)
        driver_i += 1
    nb["cells"][driver_i]["source"] = as_source(DRIVER)

    last_i = find_cell(nb, "WALL hours=", cell_type="code")
    last = cell_src(nb["cells"][last_i])
    last = last.replace(
        'print(f"WALL hours={(time.time()-T0)/3600:.3f} schedule=single 8x6 A-only", flush=True)',
        'print(f"WALL hours={(time.time()-T0)/3600:.3f} schedule=leftover 8x6 A+B-unc", flush=True)',
    )
    last = last.replace("schedule=single 8x6 A-only", "schedule=leftover 8x6 A+B-unc")
    nb["cells"][last_i]["source"] = as_source(last)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    src = "\n".join(cell_src(c) for c in nb["cells"])
    assert 'SCHEDULE = "leftover"' in src
    assert "b_priority_queue.py" in src
    assert "NVARC_CHECKPOINT_MIN_GAP" in src
    assert "Save Version smoke: skip leftover-B" in src
    assert 'live_checkpoint("tick", force=False)' in src
    print("patched", NB, "cells", len(nb["cells"]))


if __name__ == "__main__":
    main()
