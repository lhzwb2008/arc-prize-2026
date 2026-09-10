#!/usr/bin/env python3
"""Offline ("weak") SFT: one global LoRA on the ARC-AGI-2 training tasks.

Goal: teach the base model the *format* and the basic few-shot induction habit
(look at demo pairs, emit a grid of the right shape) before per-task TTT.

Data sources
  * data/full/data/training/*.json  (1000 tasks, train+test pairs all labelled)
  * optional RE-ARC dump: <dir>/tasks/<id>.json, a list of {"input","output"}
    (michaelhodel/re-arc; 400 ARC-1 tasks x 1000 verified examples)

  * optional BARC-Heavy jsonl (barc0/200k_HEAVY_...): {"source": code with
    "# concepts:"/"# description:" header, "examples": [[in, out], ...]}

Each SFT sample = one augmented mini-task rendered as a multi-turn chat
(shared renderer in ttt_qwen.py): random dihedral transform, random colour
permutation, shuffled pair order, loss on every assistant grid.

--rule-mode: "state the rule, then draw the grid". Only tasks that have a
natural-language rule are used (data/rules/task_rules.json for ARC/RE-ARC ids,
the description header for BARC-Heavy); the final answer is
'Concepts: ..\\nRule: ..\\nOutput:\\n<grid>'. Rule text is rewritten to follow the
colour permutation, and the dihedral transform is restricted to those the text
can follow (rule_text.rewrite_rule).
--strip-rules: identical sample selection/augmentation but plain grid answers ->
the control arm of the ablation.

Usage:
    python scripts/sft_qwen.py --model /opt/models/Qwen3.5-4B \
        --augs-per-task 8 --max-len 4096 --out-dir /opt/work/sft
    # then
    python scripts/ttt_qwen.py --split evaluation --sft-adapter /opt/work/sft/adapter
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_text import allowed_geoms, format_rule_block, parse_barc_source, rewrite_rule  # noqa: E402
from ttt_qwen import (  # noqa: E402
    GEOM,
    color_perm,
    encode_conversation,
    load_base,
    random_color_table,
    sparse_ce_loss,
)

ROOT = Path(__file__).resolve().parents[1]

# A task is (task_id, pairs, rules) with rules = [(concepts, rule_text), ...] (may be empty).


def load_rules(path: str):
    if not path or not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text())
    return {tid: [(e.get("concepts", []), e["rule"]) for e in lst if e.get("rule")] for tid, lst in raw.items()}


def load_arc_tasks(folder: Path, rules=None):
    rules = rules or {}
    tasks = []
    for fp in sorted(folder.glob("*.json")):
        if fp.name.startswith("._"):
            continue
        t = json.loads(fp.read_text())
        pairs = [(ex["input"], ex["output"]) for ex in t["train"] + t["test"] if ex.get("output")]
        if len(pairs) >= 2:
            tasks.append((fp.stem, pairs, rules.get(fp.stem, [])))
    return tasks


def load_rearc_tasks(folder: Path, rules=None):
    """RE-ARC dump: tasks/<id>.json -> list of {"input","output"}."""
    rules = rules or {}
    tasks_dir = folder / "tasks" if (folder / "tasks").exists() else folder
    tasks = []
    for fp in sorted(tasks_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        pairs = [(ex["input"], ex["output"]) for ex in data if ex.get("output")]
        if len(pairs) >= 2:
            tasks.append((f"rearc_{fp.stem}", pairs, rules.get(fp.stem, [])))
    return tasks


def _grid_ok(g) -> bool:
    return (
        isinstance(g, list) and 0 < len(g) <= 30 and isinstance(g[0], list) and 0 < len(g[0]) <= 30
        and all(len(r) == len(g[0]) for r in g)
        and all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r)
    )


def load_barc_heavy(path: Path, max_records: int, skip: int = 0):
    """Stream the BARC-Heavy jsonl; each record is a synthetic task with ~30 pairs + a description."""
    tasks = []
    with path.open() as f:
        for li, line in enumerate(f):
            if li < skip:
                continue
            if max_records and len(tasks) >= max_records:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            concepts, rule = parse_barc_source(rec.get("source", ""))
            pairs = []
            for ex in rec.get("examples", []):
                if isinstance(ex, dict):
                    a, b = ex.get("input"), ex.get("output")
                else:
                    a, b = ex[0], ex[1]
                if _grid_ok(a) and _grid_ok(b):
                    pairs.append((a, b))
            if len(pairs) >= 2 and rule:
                tasks.append((f"barc_{li}", pairs, [(concepts, rule)]))
    return tasks


def make_sample(tok, pairs, rng: random.Random, args, rules=None):
    """One augmented mini-task -> encoded conversation, or None if it doesn't fit.

    rules: [(concepts, rule_text)]; used when args.rule_mode or args.strip_rules.
    The random draws are identical in both arms so the ablation sees the same grids.
    """
    pairs = list(pairs)
    rng.shuffle(pairs)
    n = min(len(pairs), rng.randint(args.min_pairs, args.max_pairs))
    pairs = pairs[:n]
    with_rules = bool(rules) and (args.rule_mode or args.strip_rules)
    concepts, rule = rng.choice(rules) if with_rules else ([], None)
    geoms = GEOM
    if with_rules:
        ok = set(allowed_geoms(rule))
        geoms = [g for g in GEOM if g[0] in ok] or [GEOM[0]]
    name, fn = rng.choice(geoms)
    mapped = [(fn(a), fn(b)) for a, b in pairs]
    table = None
    if rng.random() < args.color_prob:
        table = random_color_table(rng)
        mapped = [(color_perm(a, table), color_perm(b, table)) for a, b in mapped]
    block = None
    rule_mode = bool(args.rule_mode) and with_rules
    if rule_mode:
        # geoms were filtered by allowed_geoms, so the direction rewrite cannot fail
        rw = rewrite_rule(rule, name, table) or rule
        block = format_rule_block(rw, concepts)
    query_in, query_out = mapped[-1]
    ctx = mapped[:-1]
    return encode_conversation(
        tok, ctx, query_in, query_out, args.max_len, loss=args.loss,
        min_ctx=1, allow_truncate=False, rule=block, rule_mode=rule_mode,
    )


def pack_encoded(encoded, max_len: int):
    """Concatenate short conversations so each sequence is close to max_len.

    Median ARC chat is ~1k tokens; padding them to 4096 wastes most of the GPU math.
    """
    packed = []
    cur_ids, cur_lab = [], []

    def flush():
        nonlocal cur_ids, cur_lab
        if not cur_ids:
            return
        packed.append({
            "input_ids": torch.tensor(cur_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(cur_ids), dtype=torch.long),
            "labels": torch.tensor(cur_lab, dtype=torch.long),
        })
        cur_ids, cur_lab = [], []

    for ex in encoded:
        ids = ex["input_ids"].tolist()
        lab = ex["labels"].tolist()
        if len(ids) > max_len:
            ids, lab = ids[-max_len:], lab[-max_len:]
        if cur_ids and len(cur_ids) + len(ids) > max_len:
            flush()
        cur_ids.extend(ids)
        cur_lab.extend(lab)
    flush()
    return packed


def collate(batch):
    max_len = max(x["input_ids"].size(0) for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in batch:
        pad = max_len - x["input_ids"].size(0)
        out["input_ids"].append(torch.nn.functional.pad(x["input_ids"], (0, pad), value=0))
        out["attention_mask"].append(torch.nn.functional.pad(x["attention_mask"], (0, pad), value=0))
        out["labels"].append(torch.nn.functional.pad(x["labels"], (0, pad), value=-100))
    return {k: torch.stack(v) for k, v in out.items()}


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base * (1.0 + math.cos(math.pi * t))


def sft_train_loop(model, encoded, args, device: str, log_path: Path):
    model.train()
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    n = len(encoded)
    bs = max(1, args.batch)
    accum = max(1, args.grad_accum)
    steps_per_epoch = max(1, -(-n // (bs * accum)))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = int(total_steps * args.warmup_ratio)
    print(f"optimizer steps={total_steps} warmup={warmup} packed_seqs={n} batch={bs} accum={accum}", flush=True)
    t0 = time.time()
    last_loss = 0.0
    cursor = 0
    opt.zero_grad(set_to_none=True)
    for step in range(total_steps):
        lr = cosine_lr(step, total_steps, warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        step_loss = 0.0
        for _micro in range(accum):
            batch = collate([encoded[(cursor + j) % n] for j in range(bs)])
            cursor = (cursor + bs) % n
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            try:
                loss = sparse_ce_loss(model, batch) / accum
                loss.backward()
                step_loss += float(loss.detach() * accum)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at step {step+1}, skipping microbatch seq_len={batch['input_ids'].size(1)}", flush=True)
                opt.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        last_loss = step_loss / accum
        if (step + 1) % 10 == 0 or step == 0:
            rec = {
                "step": step + 1,
                "elapsed_s": round(time.time() - t0),
                "loss": last_loss,
                "lr": lr,
                "epoch": round((step + 1) / total_steps, 4),
            }
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    return {"train_runtime": round(time.time() - t0, 1), "train_loss": last_loss, "global_step": total_steps}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/opt/models/Qwen3.5-4B")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--train-dir", default=str(ROOT / "data" / "full" / "data" / "training"))
    p.add_argument("--augs-per-task", type=int, default=8)
    p.add_argument("--rearc-dir", default="", help="optional RE-ARC dump (dir with tasks/*.json)")
    p.add_argument("--rearc-per-task", type=int, default=0)
    p.add_argument("--barc-heavy", default="", help="optional BARC-Heavy jsonl (data_100k.jsonl)")
    p.add_argument("--barc-heavy-max", type=int, default=0, help="records to read from the jsonl")
    p.add_argument("--barc-heavy-skip", type=int, default=0)
    p.add_argument("--barc-heavy-per-task", type=int, default=1)
    p.add_argument("--rules-json", default=str(ROOT / "data" / "rules" / "task_rules.json"),
                   help="task_id -> natural-language rules (scripts/build_rule_data.py)")
    p.add_argument("--rule-mode", action="store_true",
                   help="'Rule: ... Output: grid' answers; keeps only tasks that have a rule")
    p.add_argument("--strip-rules", action="store_true",
                   help="control arm: same tasks/augmentations as --rule-mode, plain grid answers")
    p.add_argument("--exclude-ids", default="",
                   help="file with task ids (one per line) to hold out from ARC/RE-ARC sources")
    p.add_argument("--min-pairs", type=int, default=3, help="pairs per mini-task incl. the query")
    p.add_argument("--max-pairs", type=int, default=8)
    p.add_argument("--color-prob", type=float, default=0.8)
    p.add_argument("--loss", default="all", choices=["all", "last"])
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--pack", action="store_true", default=True,
                   help="concatenate short samples up to max-len (much less padding)")
    p.add_argument("--no-pack", action="store_false", dest="pack")
    p.add_argument("--grad-ckpt", action="store_true", default=True)
    p.add_argument("--no-grad-ckpt", action="store_false", dest="grad_ckpt")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--dump-samples", type=int, default=0, help="write N decoded samples to out-dir/samples.txt")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="/opt/work/sft")
    p.add_argument("--merged-dir", default="", help="also save merged full weights here (optional)")
    args = p.parse_args()
    if args.rule_mode and args.strip_rules:
        p.error("--rule-mode and --strip-rules are the two arms of the ablation; pick one")
    need_rules = args.rule_mode or args.strip_rules

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    tok, model = load_base(args.model, args.device)

    # ---- build data ------------------------------------------------------
    rules = load_rules(args.rules_json) if need_rules else {}
    sources = []
    arc_tasks = load_arc_tasks(Path(args.train_dir), rules)
    if args.augs_per_task > 0:
        sources.append(("arc_training", arc_tasks, args.augs_per_task))
    if args.rearc_dir and args.rearc_per_task > 0:
        rearc = load_rearc_tasks(Path(args.rearc_dir), rules)
        sources.append(("rearc", rearc, args.rearc_per_task))
    if args.barc_heavy and args.barc_heavy_max > 0:
        t_b = time.time()
        barc = load_barc_heavy(Path(args.barc_heavy), args.barc_heavy_max, args.barc_heavy_skip)
        print(f"barc_heavy: read {len(barc)} records in {time.time() - t_b:.0f}s", flush=True)
        sources.append(("barc_heavy", barc, args.barc_heavy_per_task))
    if need_rules:
        # both arms train on exactly the tasks that have a rule
        sources = [(n, [t for t in tasks if t[2]], k) for n, tasks, k in sources]
    if args.exclude_ids:
        excl = {ln.strip() for ln in Path(args.exclude_ids).read_text().splitlines() if ln.strip()}
        before = sum(len(t) for _, t, _ in sources)
        sources = [(n, [t for t in tasks if t[0].replace("rearc_", "") not in excl], k)
                   for n, tasks, k in sources]
        print(f"excluded {before - sum(len(t) for _, t, _ in sources)} tasks via {args.exclude_ids}", flush=True)
    t0 = time.time()
    encoded = []
    stats = {}
    for src_name, tasks, per_task in sources:
        kept = skipped = 0
        for _tid, pairs, task_rules in tasks:
            for _ in range(per_task):
                ex = make_sample(tok, pairs, rng, args, rules=task_rules)
                if ex is None:
                    skipped += 1
                    continue
                encoded.append(ex)
                kept += 1
        stats[src_name] = {"tasks": len(tasks), "samples": kept, "skipped_too_long": skipped}
        print(f"{src_name}: tasks={len(tasks)} samples={kept} skipped={skipped}", flush=True)
    rng.shuffle(encoded)
    if args.max_samples:
        encoded = encoded[: args.max_samples]
    if args.dump_samples:
        with (out_dir / "samples.txt").open("w") as f:
            for ex in encoded[: args.dump_samples]:
                ids, lab = ex["input_ids"].tolist(), ex["labels"].tolist()
                sup = tok.decode([t for t, l in zip(ids, lab) if l != -100])
                f.write("=" * 78 + "\n" + tok.decode(ids) + "\n--- supervised tokens ---\n" + sup + "\n")
        print(f"wrote {min(args.dump_samples, len(encoded))} decoded samples to {out_dir / 'samples.txt'}", flush=True)
    lengths = sorted(int(e["input_ids"].numel()) for e in encoded)
    sup = sum(int((e["labels"] != -100).sum()) for e in encoded)
    stats["total"] = {
        "samples": len(encoded),
        "tokens": sum(lengths),
        "supervised_tokens": sup,
        "len_median": lengths[len(lengths) // 2] if lengths else 0,
        "len_p90": lengths[int(len(lengths) * 0.9)] if lengths else 0,
        "len_max": lengths[-1] if lengths else 0,
        "build_seconds": round(time.time() - t0, 1),
    }
    if args.pack:
        before = len(encoded)
        encoded = pack_encoded(encoded, args.max_len)
        pack_lens = sorted(int(e["input_ids"].numel()) for e in encoded)
        stats["packed"] = {
            "seqs": len(encoded),
            "from_samples": before,
            "len_median": pack_lens[len(pack_lens) // 2] if pack_lens else 0,
            "len_p90": pack_lens[int(len(pack_lens) * 0.9)] if pack_lens else 0,
            "len_min": pack_lens[0] if pack_lens else 0,
        }
        print(f"packed {before} -> {len(encoded)} seqs median={stats['packed']['len_median']}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)
    (out_dir / "data_stats.json").write_text(json.dumps({"args": vars(args), "stats": stats}, indent=2))
    if not encoded:
        print("no samples", flush=True)
        return 1

    # ---- LoRA -------------------------------------------------------------
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.train()

    t1 = time.time()
    train_out = sft_train_loop(model, encoded, args, args.device, log_path)
    print(train_out, flush=True)

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    summary = {
        "args": vars(args),
        "stats": stats,
        "train_seconds": round(time.time() - t1, 1),
        "train_loss": train_out.get("train_loss"),
        "adapter": str(adapter_dir),
        "train_out": train_out,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved adapter to {adapter_dir}", flush=True)

    if args.merged_dir:
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merged_dir, safe_serialization=True)
        tok.save_pretrained(args.merged_dir)
        print(f"saved merged weights to {args.merged_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
