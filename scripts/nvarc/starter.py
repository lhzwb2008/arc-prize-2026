"""Local port of the NVARC starter (Kaggle notebook cell `starter.py`).

Same worker loop as the Kaggle version, but paths / GPU count / task list are
arguments instead of hardcoded Kaggle locations. Per-task budgets inside
arc_solver.py (TTT, 1200s decode, 540s DFS) are left untouched so that local
numbers stay comparable with the 4xL4 Kaggle run.

Example (3090, 120 eval tasks, overnight):
    NVARC_MODEL=/opt/models/qwen3_4b_grids15_sft139 \
    python starter.py --data /opt/data/kaggle/arc-agi_evaluation_challenges.json \
        --out /opt/work/nvarc/eval120/outputs --hours 30
"""
import argparse
import json
import os
import time

import torch
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time, sync_dir):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue: stagger worker start-up.
    if rank > 0:
        while not os.path.exists(os.path.join(sync_dir, f"worker{rank-1}")):
            time.sleep(5)

    from arc_solver import worker

    with open(os.path.join(sync_dir, f"worker{rank}"), "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")

    worker(rank, queue, end_time)

    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="arc-agi_*_challenges.json")
    parser.add_argument("--out", required=True, help="dir for per-subkey pickles")
    parser.add_argument("--model", default=os.getenv("NVARC_MODEL", "/opt/models/qwen3_4b_grids15_sft139"))
    parser.add_argument("--nprocs", type=int, default=torch.cuda.device_count() or 1)
    parser.add_argument("--hours", type=float, default=12.0, help="wall budget; Kaggle uses 12h-10min")
    parser.add_argument("--end-time", type=float, default=0.0, help="absolute epoch; overrides --hours")
    parser.add_argument("--keys", default="", help="comma list of task ids (default: all)")
    parser.add_argument("--keys-file", default="", help="file with one task id per line")
    parser.add_argument("--skip-done", action="store_true", help="skip tasks that already have pickles in --out")
    args = parser.parse_args()

    end_time = args.end_time or (time.time() + args.hours * 3600 - 600)

    os.environ["NVARC_MODEL"] = args.model
    os.environ["NVARC_DATA"] = args.data
    os.environ["NVARC_OUT"] = args.out
    os.makedirs(args.out, exist_ok=True)

    with open(args.data, "r") as f:
        data = json.load(f)

    if args.keys:
        wanted = [k.strip() for k in args.keys.split(",") if k.strip()]
    elif args.keys_file:
        wanted = [l.strip() for l in open(args.keys_file) if l.strip()]
    else:
        wanted = sorted(data.keys())

    if args.skip_done:
        done = {fn.split("_", 1)[0] for fn in os.listdir(args.out) if "_" in fn}
        wanted = [k for k in wanted if k not in done]

    # sibling dir: ArcDecoder.load_decoded_results() treats every entry of --out as a pickle
    sync_dir = args.out.rstrip("/") + ".sync"
    os.makedirs(sync_dir, exist_ok=True)
    for fn in os.listdir(sync_dir):
        os.remove(os.path.join(sync_dir, fn))

    queue = mp.Manager().Queue()
    for key in wanted:
        if key not in data:
            raise SystemExit(f"unknown task id {key}")
        queue.put(key)
    for _ in range(args.nprocs):
        queue.put(None)

    print(f"starter: nprocs={args.nprocs} cuda={torch.cuda.device_count()} "
          f"cap=sm_{''.join(map(str, torch.cuda.get_device_capability()))} "
          f"nkeys={len(wanted)} budget_h={(end_time-time.time())/3600:.2f} data={args.data} out={args.out}")

    mp.spawn(local_worker, args=(queue, end_time, sync_dir), nprocs=args.nprocs)
