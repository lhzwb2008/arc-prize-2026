"""Local port of the NVARC starter (Kaggle notebook cell `starter.py`).

Same worker loop as the Kaggle version, but paths / GPU count / task list are
arguments instead of hardcoded Kaggle locations.

Live pooling (Kaggle v14): set NVARC_CHECKPOINT_PRIMARY / _EXTRAS / _SUB
and optional NVARC_HARD_MERGE_TIME (epoch seconds). None sentinels are queued
only after every worker has started.
"""
import argparse
import json
import os
import signal
import threading
import time
import traceback

import torch
import torch.multiprocessing as mp


def estimated_work(task):
    def ntok(g):
        return len(g) * (len(g[0]) + 1) if g and g[0] else 0
    train_tokens = sum(ntok(p["input"]) + ntok(p["output"]) for p in task["train"])
    ratios = [ntok(p["output"]) / max(1, ntok(p["input"])) for p in task["train"]] or [1.0]
    ratios.sort()
    ratio = ratios[len(ratios) // 2]
    test_tokens = sum(ntok(t["input"]) * (1 + ratio) for t in task["test"])
    n_train = int(os.getenv("ARC_N_TRAIN_AUG", "8"))
    n_geos = int(os.getenv("ARC_N_EVAL_GEOS", "6"))
    return train_tokens * n_train + test_tokens * n_geos * len(task["test"])


def live_checkpoint(tag="live"):
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
        from checkpoint import run_live
        with open(sub + ".lock", "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                print(f"live checkpoint {tag} primary={primary} extra={extras}", flush=True)
                run_live(primary, extras, sub, os.getenv("NVARC_DATA") or "", False)
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"live checkpoint {tag}: {type(e).__name__}: {e}", flush=True)


def local_worker(rank, queue, end_time, sync_dir):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    if rank > 0:
        waited = 0
        marker = os.path.join(sync_dir, f"worker{rank-1}")
        while not os.path.exists(marker) and waited < 900:
            time.sleep(5)
            waited += 5

    from arc_solver import worker

    with open(os.path.join(sync_dir, f"worker{rank}"), "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!", flush=True)
    worker(rank, queue, end_time)
    print(f"[Rank {rank}] done!", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="arc-agi_*_challenges.json")
    parser.add_argument("--out", required=True, help="dir for per-subkey pickles")
    parser.add_argument("--model", default=os.getenv("NVARC_MODEL", "/opt/models/qwen3_4b_grids15_sft139"))
    parser.add_argument("--nprocs", type=int, default=torch.cuda.device_count() or 1)
    parser.add_argument("--hours", type=float, default=12.0, help="wall budget; 0 = no cutoff")
    parser.add_argument("--end-time", type=float, default=0.0, help="absolute epoch; overrides --hours")
    parser.add_argument("--no-timeouts", action="store_true",
                        help="finish every task: no wall/DFS/per-task caps")
    parser.add_argument("--keys", default="", help="comma list of task ids (default: all)")
    parser.add_argument("--keys-file", default="", help="file with one task id per line")
    parser.add_argument("--skip-done", action="store_true", help="skip tasks that already have pickles in --out")
    parser.add_argument("--order", default="sorted", choices=["cheap", "sorted", "expensive", "file"])
    args = parser.parse_args()

    if args.end_time > 0:
        end_time = args.end_time
    elif args.no_timeouts or args.hours <= 0:
        os.environ["NVARC_TASK_LIMIT"] = "0"
        os.environ["NVARC_DFS_LIMIT"] = "0"
        end_time = time.time() + 365 * 24 * 3600
    else:
        end_time = time.time() + args.hours * 3600 - 600

    if args.no_timeouts or args.hours <= 0:
        os.environ["NVARC_TASK_LIMIT"] = "0"
        os.environ["NVARC_DFS_LIMIT"] = "0"

    os.environ["NVARC_MODEL"] = args.model
    os.environ["NVARC_DATA"] = args.data
    os.environ["NVARC_OUT"] = args.out
    os.makedirs(args.out, exist_ok=True)

    with open(args.data, "r") as f:
        data = json.load(f)

    if args.keys:
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    elif args.keys_file:
        keys = json.load(open(args.keys_file)) if args.keys_file.endswith(".json") else [
            l.strip() for l in open(args.keys_file) if l.strip()
        ]
    else:
        keys = sorted(data.keys())
    keys = [k for k in keys if k in data]

    if args.order == "cheap":
        keys = sorted(keys, key=lambda k: estimated_work(data[k]))
    elif args.order == "expensive":
        keys = sorted(keys, key=lambda k: estimated_work(data[k]), reverse=True)
    elif args.order == "sorted":
        keys = sorted(keys)

    if args.skip_done:
        done = {fn.split("_", 1)[0] for fn in os.listdir(args.out) if "_" in fn}
        keys = [k for k in keys if k not in done]

    for key in keys:
        if key not in data:
            raise SystemExit(f"unknown task id {key}")

    queued_path = os.getenv("NVARC_QUEUED_KEYS", "")
    if queued_path:
        with open(queued_path, "w") as f:
            json.dump(keys, f)

    sync_dir = args.out.rstrip("/") + ".sync"
    os.makedirs(sync_dir, exist_ok=True)
    for fn in os.listdir(sync_dir):
        os.remove(os.path.join(sync_dir, fn))

    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)

    nprocs = args.nprocs

    def put_sentinels():
        deadline = end_time if end_time > 0 else time.time() + 3600
        for i in range(nprocs):
            marker = os.path.join(sync_dir, f"worker{i}")
            while not os.path.exists(marker) and time.time() < deadline:
                time.sleep(1)
        for _ in range(nprocs):
            queue.put(None)
        print(f"starter: queued {nprocs} sentinels after worker markers", flush=True)

    print(
        f"starter: nprocs={nprocs} cuda={torch.cuda.device_count()} "
        f"cap=sm_{''.join(map(str, torch.cuda.get_device_capability()))} "
        f"nkeys={len(keys)} order={args.order} budget_h={(end_time-time.time())/3600:.2f} "
        f"data={args.data} out={args.out}",
        flush=True,
    )

    stop = threading.Event()
    hard_merge = float(os.getenv("NVARC_HARD_MERGE_TIME") or "0")

    def _on_term(signum, frame):
        print(f"starter signal {signum}: checkpoint then stop", flush=True)
        live_checkpoint(f"signal-{signum}")

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    def _loop():
        interval = float(os.getenv("NVARC_CHECKPOINT_EVERY", "90"))
        while not stop.wait(interval):
            live_checkpoint("tick")
            if hard_merge > 0 and time.time() >= hard_merge:
                live_checkpoint("hard-Tminus5")
                break

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    threading.Thread(target=put_sentinels, daemon=True).start()
    try:
        live_checkpoint("start")
        mp.spawn(local_worker, args=(queue, end_time, sync_dir), nprocs=nprocs)
    except Exception as e:
        print(f"[starter] spawn finished with error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    finally:
        stop.set()
        live_checkpoint("end")
    print("[starter] finished.", flush=True)
