"""Local replay of the Kaggle v14 scheduler (after pass A).

Catchup on A leftovers, then full 8×6 B with cheap-first order, 90s live
mean_quality merge, and a hard write at T0+12h-5min. B itself is not killed
at 12h-20min so the local eval120 run can finish A+B; the hard-merge path
still fires on the Kaggle clock so logs show the same checkpoint code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.getenv("NVARC_WORK", "/opt/work/nvarc"))
DATA = os.getenv("NVARC_DATA", "/opt/data/kaggle/arc-agi_evaluation_challenges.json")
SOL = os.getenv("NVARC_SOL", "/opt/data/kaggle/arc-agi_evaluation_solutions.json")
PYTHON = os.getenv("NVARC_PYTHON", sys.executable)

A_NAME = os.getenv("NVARC_N8X6_A", "eval120_n8x6_a")
B_NAME = os.getenv("NVARC_N8X6_B", "eval120_n8x6_b")
SUM = WORK / os.getenv("NVARC_N8X6_SUM", "eval120_n8x6_two")
PRIMARY = str(WORK / A_NAME / "outputs")
B_OUT = str(WORK / B_NAME / "outputs")
SUB = str(SUM / "submission.json")
QUEUED = str(SUM / "queued_keys.json")


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


t0_raw = os.getenv("NVARC_T0", "")
if t0_raw:
    T0 = float(t0_raw)
else:
    p = SUM / "t0.epoch"
    T0 = float(p.read_text()) if p.exists() else time.time()

global_end_time = T0 + 12 * 3600 - 20 * 60
hard_merge_time = T0 + 12 * 3600 - 5 * 60
finish_all = os.getenv("NVARC_FINISH_ALL", "1") not in ("0", "false", "no")

COMMON = {
    "UNSLOTH_DISABLE_STATISTICS": "1",
    "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "12"),
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "ARC_N_EVAL_AUG": "1",
    "ARC_N_TRAIN_AUG": "8",
    "ARC_N_EVAL_GEOS": "6",
    "NVARC_TASK_LIMIT": "0",
    "NVARC_DFS_LIMIT": "0",
    "NVARC_CHECKPOINT_EVERY": "90",
    "NVARC_CHECKPOINT_SUB": SUB,
    "NVARC_HARD_MERGE_TIME": str(hard_merge_time),
    "NVARC_DATA": DATA,
    "NVARC_QUEUED_KEYS": QUEUED,
}
if Path("/usr/local/cuda/bin/ptxas").exists():
    COMMON["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ.update(COMMON)

SEED_A = dict(ARC_LORA_SEED=42, ARC_TRAIN_AUG_SEED=1, ARC_EVAL_AUG_SEED=2, ARC_SCORE_SEED_OFFSET=0)
SEED_B = dict(ARC_LORA_SEED=137, ARC_TRAIN_AUG_SEED=17, ARC_EVAL_AUG_SEED=29, ARC_SCORE_SEED_OFFSET=7)

PASSES = [
    dict(name="A", out=PRIMARY, seeds=SEED_A, order="cheap"),
    dict(name="B", out=B_OUT, seeds=SEED_B, order="cheap"),
]


def remaining():
    return global_end_time - time.time()


def done_keys(out_dir):
    p = Path(out_dir)
    if not p.exists():
        return set()
    return {fn.split("_", 1)[0] for fn in os.listdir(p) if "_" in fn}


def pool_dirs():
    dirs = [PRIMARY]
    p = Path(B_OUT)
    if p.exists() and any(p.iterdir()):
        dirs.append(B_OUT)
    return dirs


def checkpoint(dirs, tag):
    cmd = [PYTHON, str(HERE / "checkpoint.py"), "--outputs", dirs[0],
           "--submission", SUB, "--data", DATA]
    for extra in dirs[1:]:
        if Path(extra).exists() and any(Path(extra).iterdir()):
            cmd.extend(["--outputs-extra", extra])
    print(f"[checkpoint {tag}] {' '.join(cmd)} remaining={remaining()/60:.1f} min", flush=True)
    lock_path = SUB + ".lock"
    Path(SUB).parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with open(lock_path, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            rc = subprocess.call(cmd)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    print(
        f"[checkpoint {tag}] rc={rc} remaining={remaining()/60:.1f} min exists={Path(SUB).exists()}",
        flush=True,
    )
    return rc


def run_pass(spec, end_time, keys_file="", skip_done=False, extra_dirs=None):
    name, out_dir = spec["name"], spec["out"]
    env = dict(os.environ)
    env.update({k: str(v) for k, v in spec["seeds"].items()})
    env["NVARC_CHECKPOINT_PRIMARY"] = PRIMARY
    extras = [d for d in (extra_dirs or []) if d != PRIMARY]
    env["NVARC_CHECKPOINT_EXTRAS"] = ":".join(extras)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(HERE / "starter.py"),
        "--data", DATA,
        "--out", out_dir,
        "--order", spec["order"],
        "--no-timeouts",
    ]
    if end_time and not finish_all:
        cmd.extend(["--end-time", str(end_time)])
    if keys_file:
        cmd.extend(["--keys-file", keys_file])
    if skip_done:
        cmd.append("--skip-done")
    print(
        f"[pass {name}] start remaining={remaining()/60:.1f} min out={out_dir} "
        f"order={spec['order']} extra={spec['seeds']} finish_all={finish_all}",
        flush=True,
    )
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    nfiles = len(list(Path(out_dir).glob("*"))) if Path(out_dir).exists() else 0
    print(
        f"[pass {name}] rc={rc} took={(time.time()-t0)/60:.1f} min files={nfiles} "
        f"remaining={remaining()/60:.1f} min",
        flush=True,
    )
    print(f"starter_exit={rc}", flush=True)
    return rc


def _hard_merge_watchdog():
    while time.time() < hard_merge_time:
        time.sleep(5)
    print("[hard-merge] 12h-5min reached; writing A+B pool", flush=True)
    try:
        checkpoint(pool_dirs(), "hard-Tminus5")
    except Exception as e:
        print(f"[hard-merge] {type(e).__name__}: {e}", flush=True)


def finalize_pass(outputs, submission, report, extra=""):
    cmd = [
        PYTHON, str(HERE / "finalize.py"),
        "--data", DATA, "--solutions", SOL,
        "--outputs", outputs,
        "--submission", submission,
        "--report", report,
    ]
    if extra:
        cmd.extend(["--outputs-extra", extra])
    print("[finalize]", " ".join(cmd), flush=True)
    subprocess.call(cmd)


def score_outputs(tag, outputs):
    cmd = [PYTHON, str(HERE / "compare_gen.py"), "--tag", tag, "--outputs", outputs,
           "--summary", str(SUM / "compare.json")]
    subprocess.call(cmd)


def main():
    SUM.mkdir(parents=True, exist_ok=True)
    (SUM / "t0.epoch").write_text(str(T0) + "\n")
    log(
        f"RECIPE v14 local n_train=8 geos=6 ranker=mean_quality finish_all={finish_all} "
        f"T0={time.strftime('%F %T', time.localtime(T0))} "
        f"hard_merge_in={(hard_merge_time-time.time())/60:.1f}min "
        f"workers_stop_in={(global_end_time-time.time())/60:.1f}min"
    )

    threading.Thread(target=_hard_merge_watchdog, daemon=True).start()

    with open(DATA) as f:
        all_keys = sorted(json.load(f))
    a_done = done_keys(PRIMARY)
    unprocessed = [k for k in all_keys if k not in a_done]
    log(f"[catchup] A reached {len(a_done)}/{len(all_keys)} unprocessed={len(unprocessed)}")
    if unprocessed:
        keys_file = str(SUM / "keys_catchup.json")
        json.dump(unprocessed, open(keys_file, "w"))
        catchup = dict(name="catchup", out=PRIMARY, seeds=SEED_A, order="file")
        run_pass(catchup, global_end_time, keys_file=keys_file, skip_done=True, extra_dirs=[PRIMARY])

    a_work = WORK / A_NAME
    finalize_pass(PRIMARY, str(a_work / "submission.json"), str(a_work / "report.json"))
    checkpoint([PRIMARY], "after-A")
    score_outputs(A_NAME, PRIMARY)
    log(f"pass-A leftover files={len(list(Path(PRIMARY).glob('*'))) if Path(PRIMARY).exists() else 0}")

    t_b0 = time.time()
    run_pass(PASSES[1], global_end_time, skip_done=True, extra_dirs=[PRIMARY, B_OUT])
    t_b1 = time.time()
    b_work = WORK / B_NAME
    b_work.mkdir(parents=True, exist_ok=True)
    finalize_pass(B_OUT, str(b_work / "submission.json"), str(b_work / "report.json"))
    (b_work / "done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    checkpoint(pool_dirs(), "after-B")
    log(f"pass-B leftover files={len(list(Path(B_OUT).glob('*'))) if Path(B_OUT).exists() else 0}")
    score_outputs(B_NAME, B_OUT)
    finalize_pass(PRIMARY, str(SUM / "submission.json"), str(SUM / "report.json"), extra=B_OUT)

    a_rep = json.loads((WORK / A_NAME / "report.json").read_text()) if (WORK / A_NAME / "report.json").exists() else {}
    b_rep = json.loads((WORK / B_NAME / "report.json").read_text()) if (WORK / B_NAME / "report.json").exists() else {}
    pool_rep = json.loads((SUM / "report.json").read_text()) if (SUM / "report.json").exists() else {}
    n = int(pool_rep.get("n_tasks") or a_rep.get("n_tasks") or 120)

    def row(tag, score):
        if score is None:
            print(f"{tag:22s}    n/a", flush=True)
            return
        print(f"{tag:22s} {score:7.3f}/{n} = {100.0 * score / n:5.2f}%", flush=True)

    print("--- native 8x6 A+B timed pool (v14 local) ---", flush=True)
    row("pass A seed42", a_rep.get("score"))
    row("pass B seed137", b_rep.get("score"))
    row("pool mean_quality", pool_rep.get("score"))
    print(f"B wall {(t_b1 - t_b0)/3600:.2f}h  remaining={remaining()/60:.1f} min", flush=True)

    (SUM / "timing.json").write_text(json.dumps({
        "recipe": {
            "n_train_aug": 8, "n_eval_geos": 6, "n_eval_aug": 1,
            "lr": 5e-5, "epochs": 1, "lora_r": 256,
            "pooled": True, "ranker": "mean_quality", "keep_primary": False,
            "live_every_s": 90, "hard_merge_time": hard_merge_time,
            "finish_all": finish_all,
        },
        "t0": T0,
        "pass_a": {"name": A_NAME, "report_score": a_rep.get("score"),
                   "report_pct": None if a_rep.get("score") is None else 100.0 * a_rep["score"] / n,
                   "decoded": len(a_rep.get("decoded_tasks") or [])},
        "pass_b": {"name": B_NAME, "wall_sec": t_b1 - t_b0,
                   "report_score": b_rep.get("score"),
                   "report_pct": None if b_rep.get("score") is None else 100.0 * b_rep["score"] / n,
                   "decoded": len(b_rep.get("decoded_tasks") or [])},
        "pool": {"score": pool_rep.get("score"),
                 "pct": None if pool_rep.get("score") is None else 100.0 * pool_rep["score"] / n},
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2) + "\n")
    (SUM / "done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("n8x6 A+B timed pool done")


if __name__ == "__main__":
    main()
