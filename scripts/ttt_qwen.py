#!/usr/bin/env python3
"""Per-task LoRA test-time fine-tuning for ARC-AGI-2, NVARC/ARChitects style.

Each hidden task is an independent tiny dataset: fit a LoRA adapter on the
train pairs (plus geometric augmentations), then generate the test grid.

Usage:
    python scripts/ttt_qwen.py --split evaluation --limit 8 --device cpu
    python scripts/ttt_qwen.py --split evaluation --limit 8 --device cuda --model /data/models/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
GRID_RE = re.compile(r"^[0-9]+(?:\n[0-9]+)*$", re.M)


def shape(g):
    return (len(g), len(g[0])) if g and g[0] else (0, 0)


def rot90(g):
    h, w = shape(g)
    return [[g[h - 1 - j][i] for j in range(h)] for i in range(w)]


def rot180(g):
    return [row[::-1] for row in g[::-1]]


def rot270(g):
    h, w = shape(g)
    return [[g[j][w - 1 - i] for j in range(h)] for i in range(w)]


def flip_h(g):
    return [row[::-1] for row in g]


def flip_v(g):
    return g[::-1]


GEOM = [
    ("id", lambda g: [r[:] for r in g]),
    ("r90", rot90),
    ("r180", rot180),
    ("r270", rot270),
    ("fh", flip_h),
    ("fv", flip_v),
]


def grid_to_text(g) -> str:
    return "\n".join("".join(str(int(c)) for c in row) for row in g)


def parse_grid(text: str):
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:[^\n]*)\n?", "", text)
    # Prefer the last digit-only block (spaces between digits allowed).
    compact_lines = []
    for ln in text.splitlines():
        compact = re.sub(r"[^0-9]", "", ln.strip())
        if compact:
            compact_lines.append(compact)
        elif compact_lines and any(compact_lines):
            # blank line ends a block; keep collecting, split later
            compact_lines.append("")
    blocks = []
    cur = []
    for ln in compact_lines:
        if not ln:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(ln)
    if cur:
        blocks.append(cur)
    if not blocks:
        return None
    lines = blocks[-1]
    if not lines:
        return None
    width = len(lines[0])
    if width == 0 or any(len(ln) != width for ln in lines):
        return None
    if width > 30 or len(lines) > 30:
        return None
    return [[int(ch) for ch in ln] for ln in lines]


def color_perm(g, table):
    return [[table[v] for v in row] for row in g]


def random_color_table(rng: random.Random):
    rest = list(range(1, 10))
    rng.shuffle(rest)
    table = [0] + rest  # keep 0 as background-ish
    return table


def augment_pairs(pairs, rng: random.Random, n_color: int = 4):
    out = []
    for name, fn in GEOM:
        mapped = []
        ok = True
        for inp, oup in pairs:
            try:
                mapped.append((fn(inp), fn(oup)))
            except Exception:
                ok = False
                break
        if ok:
            out.append((name, mapped))
    for i in range(n_color):
        table = random_color_table(rng)
        mapped = [(color_perm(a, table), color_perm(b, table)) for a, b in pairs]
        out.append((f"col{i}", mapped))
    return out


def build_messages(inp, oup=None, thinking=False):
    user = grid_to_text(inp)
    msgs = [
        {
            "role": "system",
            "content": (
                "You solve ARC abstraction puzzles. "
                "Reply with only the output grid: digits 0-9, one row per line, no other text."
            ),
        },
        {"role": "user", "content": user},
    ]
    if oup is not None:
        msgs.append({"role": "assistant", "content": grid_to_text(oup)})
    return msgs


class ArcPairs(Dataset):
    def __init__(self, encoded):
        self.encoded = encoded

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, i):
        return {k: v.clone() for k, v in self.encoded[i].items()}


def encode_example(tokenizer, inp, oup, max_len: int):
    prompt_msgs = build_messages(inp)

    def apply(msgs, add_gen: bool):
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=add_gen,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_gen
            )

    prompt = apply(prompt_msgs, True)
    answer = grid_to_text(oup)
    if tokenizer.eos_token:
        answer = answer + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    if len(full_ids) > max_len:
        full_ids = full_ids[-max_len:]
        labels = labels[-max_len:]
    if not any(x != -100 for x in labels):
        labels = list(full_ids)
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate(batch):
    max_len = max(x["input_ids"].size(0) for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in batch:
        pad = max_len - x["input_ids"].size(0)
        out["input_ids"].append(
            torch.nn.functional.pad(x["input_ids"], (0, pad), value=0)
        )
        out["attention_mask"].append(
            torch.nn.functional.pad(x["attention_mask"], (0, pad), value=0)
        )
        out["labels"].append(
            torch.nn.functional.pad(x["labels"], (0, pad), value=-100)
        )
    return {k: torch.stack(v) for k, v in out.items()}


def load_base(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device == "cpu":
        n = max(1, os.cpu_count() or 1)
        torch.set_num_threads(n)
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))
    name = model_id.lower()
    big_cpu = device == "cpu" and any(tag in name for tag in ("4b", "7b", "8b", "14b"))
    if device == "cuda":
        dtype = torch.bfloat16
    elif big_cpu:
        # fp16 GEMM is emulated and extremely slow on CPU; bf16 uses AVX512_BF16.
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=dtype,
    )
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    print(f"dtype={dtype} device_map={kwargs.get('device_map')}", flush=True)
    return tok, model


def attach_lora(model, r: int, device: str):
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    try:
        model = get_peft_model(model, cfg)
    except ValueError:
        cfg.target_modules = None
        model = get_peft_model(model, LoraConfig(
            r=r, lora_alpha=2 * r, bias="none", task_type="CAUSAL_LM",
            target_modules="all-linear",
        ))
    if device == "cuda":
        model.to("cuda")
    model.train()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def generate_grid(model, tok, inp, device: str, max_new: int = 384):
    msgs = build_messages(inp)
    try:
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device if hasattr(model, "device") else device) for k, v in inputs.items()}
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].size(1) :]
    text = tok.decode(gen, skip_special_tokens=True)
    return parse_grid(text), text


def invert_geom(name, g):
    if g is None:
        return None
    if name == "id":
        return g
    if name == "r90":
        return rot270(g)
    if name == "r180":
        return rot180(g)
    if name == "r270":
        return rot90(g)
    if name == "fh":
        return flip_h(g)
    if name == "fv":
        return flip_v(g)
    return g


def predict_with_vote(model, tok, test_inp, device: str, vote: bool = True, max_new: int = 384):
    votes = []
    raws = []
    geoms = GEOM if vote else [GEOM[0]]
    for name, fn in geoms:
        g, text = generate_grid(model, tok, fn(test_inp), device, max_new=max_new)
        raws.append((name, text[:200]))
        back = invert_geom(name, g)
        if back is not None:
            votes.append(tuple(tuple(r) for r in back))
    if not votes:
        return None, raws
    best = Counter(votes).most_common(1)[0][0]
    return [list(r) for r in best], raws


def ttt_one_task(base_model, tok, task, args, device: str):
    rng = random.Random(args.seed)
    pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
    views = augment_pairs(pairs, rng, n_color=args.color_augs)
    examples = []
    for _name, mapped in views:
        for inp, oup in mapped:
            examples.append(encode_example(tok, inp, oup, args.max_len))
    if not examples:
        return [None] * len(task["test"]), []

    n_sup = sum(int((ex["labels"] != -100).sum()) for ex in examples)
    print(f"  train_pairs={len(examples)} supervised_tokens={n_sup}", flush=True)

    model = attach_lora(base_model, args.lora_r, device)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    ds = ArcPairs(examples)
    targs = TrainingArguments(
        output_dir=str(Path(args.work_dir) / "run"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=args.steps,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        fp16=False,
        bf16=device == "cuda",
        dataloader_pin_memory=device == "cuda",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        use_cpu=device == "cpu",
        disable_tqdm=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()
    model.eval()
    preds = []
    debug = []
    for t in task["test"]:
        grid, raws = predict_with_vote(
            model, tok, t["input"], device, vote=not args.no_vote, max_new=args.max_new
        )
        preds.append(grid)
        debug.append(raws)
    if isinstance(model, PeftModel):
        model.unload()
        try:
            model.delete_adapter("default")
        except Exception:
            pass
    del trainer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return preds, debug


def grids_equal(a, b) -> bool:
    return a is not None and b is not None and a == b


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/models/Qwen3.5-4B")
    p.add_argument("--split", default="evaluation", choices=["evaluation", "training", "examples"])
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--color-augs", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", default="/data/arc-prize-2026/work")
    p.add_argument("--out", default="")
    p.add_argument("--zero-shot", action="store_true", help="skip TTT, generate from frozen model")
    p.add_argument("--no-vote", action="store_true", help="skip geometric vote at inference")
    p.add_argument("--max-new", type=int, default=384)
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "cuda":
            free = torch.cuda.get_device_properties(0).total_memory
            print(f"cuda memory total={free/1024**3:.2f} GiB", flush=True)
            if free < 6 * 1024**3:
                print("GPU < 6GiB; 4B TTT will not fit. Use --device cpu or a smaller model.", flush=True)

    if args.split == "examples":
        folder = ROOT / "data" / "examples"
    else:
        folder = ROOT / "data" / "full" / "data" / args.split
    files = sorted(
        p
        for p in folder.glob("*.json")
        if p.name != "README.md" and not p.name.startswith("._")
    )
    files = files[args.offset :]
    if args.limit:
        files = files[: args.limit]
    print(f"model={args.model} device={args.device} tasks={len(files)} steps={args.steps}", flush=True)

    tok, model = load_base(args.model, args.device)
    print("loaded", type(model).__name__, "params", sum(x.numel() for x in model.parameters()) / 1e6, "M", flush=True)

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    results = []
    solved = 0
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        task = json.loads(fp.read_text())
        golds = [t.get("output") for t in task["test"]]
        t1 = time.time()
        if args.zero_shot:
            preds, debug = [], []
            model.eval()
            for t in task["test"]:
                g, raws = predict_with_vote(
                    model,
                    tok,
                    t["input"],
                    args.device,
                    vote=not args.no_vote,
                    max_new=args.max_new,
                )
                preds.append(g)
                debug.append(raws)
        else:
            if isinstance(model, PeftModel):
                model = model.get_base_model()
            preds, debug = ttt_one_task(model, tok, task, args, args.device)
        hits = [grids_equal(pred, gold) for pred, gold in zip(preds, golds)]
        ok = all(hits) and golds and all(g is not None for g in golds)
        solved += int(ok)
        rec = {
            "id": fp.stem,
            "ok": ok,
            "hits": hits,
            "seconds": round(time.time() - t1, 1),
            "pred": preds,
            "raw": [
                [(name, txt[:240]) for name, txt in item]
                for item in debug[:2]
            ],
        }
        results.append(rec)
        print(
            f"[{i}/{len(files)}] {fp.stem} ok={ok} hits={hits} {rec['seconds']}s  "
            f"running={solved}/{i}",
            flush=True,
        )
        if debug and debug[0]:
            print(f"  raw={debug[0][0][1][:160]!r}", flush=True)
        elapsed = time.time() - t0
        summary = {
            "model": args.model,
            "device": args.device,
            "split": args.split,
            "solved": solved,
            "total": i,
            "planned": len(files),
            "acc": solved / max(i, 1),
            "seconds": round(elapsed, 1),
            "args": {k: getattr(args, k) for k in vars(args)},
            "results": results,
        }
        out = Path(args.out) if args.out else Path(args.work_dir) / f"ttt_{args.split}.json"
        out.write_text(json.dumps(summary, indent=2))

    elapsed = time.time() - t0
    summary = {
        "model": args.model,
        "device": args.device,
        "split": args.split,
        "solved": solved,
        "total": len(files),
        "acc": solved / max(len(files), 1),
        "seconds": round(elapsed, 1),
        "args": {k: getattr(args, k) for k in vars(args)},
        "results": results,
    }
    out = Path(args.out) if args.out else Path(args.work_dir) / f"ttt_{args.split}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"\n{args.split}: {solved}/{len(files)} ({100 * summary['acc']:.1f}%)  "
        f"wall {elapsed:.0f}s  wrote {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
