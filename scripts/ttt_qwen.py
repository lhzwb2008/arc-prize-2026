#!/usr/bin/env python3
"""Per-task LoRA test-time fine-tuning for ARC-AGI-2, NVARC/ARChitects style.

Each hidden task is an independent tiny dataset: fit a LoRA adapter on the
train pairs (plus geometric/color augmentations), then generate the test grid.

Two prompt formats:
  --format context (default): demos are in-context user/assistant turns, TTT is
      leave-one-out over the demos; matches scripts/sft_qwen.py.
  --format single: legacy, one input->output pair per sample, no demos shown.

Usage:
    python scripts/ttt_qwen.py --split evaluation --limit 8 --device cuda --model /opt/models/Qwen3.5-4B
    python scripts/ttt_qwen.py --split evaluation --sft-adapter /opt/work/sft/adapter --device cuda
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


def transpose(g):
    h, w = shape(g)
    return [[g[j][i] for j in range(h)] for i in range(w)]


def anti_transpose(g):
    return rot180(transpose(g))


# Full dihedral group of the square (8 elements).
GEOM = [
    ("id", lambda g: [r[:] for r in g]),
    ("r90", rot90),
    ("r180", rot180),
    ("r270", rot270),
    ("fh", flip_h),
    ("fv", flip_v),
    ("tr", transpose),
    ("atr", anti_transpose),
]
GEOM_FN = dict(GEOM)


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


SYSTEM_PROMPT = (
    "You solve ARC abstraction puzzles. "
    "Reply with only the output grid: digits 0-9, one row per line, no other text."
)


def build_messages(inp, oup=None, thinking=False):
    user = grid_to_text(inp)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if oup is not None:
        msgs.append({"role": "assistant", "content": grid_to_text(oup)})
    return msgs


# ---------------------------------------------------------------------------
# In-context ("conversation") format shared by SFT and TTT.
#
# Every demonstration pair is one user/assistant turn; the query input is the
# final user turn. We render Qwen ChatML by hand so that every assistant turn
# looks identical (empty think block, then the grid) and so we can put loss on
# every assistant answer, not just the last one.
# ---------------------------------------------------------------------------
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_HEAD = f"{IM_START}assistant\n<think>\n\n</think>\n\n"


def _segments(ctx_pairs, inp, oup=None, loss: str = "all"):
    """Return [(text, supervised)] for one conversation."""
    segs = [(f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n", False)]
    for ci, co in ctx_pairs:
        segs.append((f"{IM_START}user\n{grid_to_text(ci)}{IM_END}\n{ASSISTANT_HEAD}", False))
        segs.append((f"{grid_to_text(co)}{IM_END}", loss == "all"))
        segs.append(("\n", False))
    segs.append((f"{IM_START}user\n{grid_to_text(inp)}{IM_END}\n{ASSISTANT_HEAD}", False))
    if oup is not None:
        segs.append((f"{grid_to_text(oup)}{IM_END}", True))
    return segs


def render_prompt(ctx_pairs, inp) -> str:
    return "".join(t for t, _ in _segments(ctx_pairs, inp))


def _tokenize_segments(tokenizer, segs):
    ids, labels = [], []
    for text, sup in segs:
        t = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.extend(t)
        labels.extend(t if sup else [-100] * len(t))
    return ids, labels


def fit_context(tokenizer, ctx_pairs, inp, oup, max_len: int, loss: str = "all", min_ctx: int = 0):
    """Drop context pairs (from the end) until the conversation fits max_len.

    Returns (ctx_pairs_used, ids, labels) or None if even min_ctx pairs do not fit.
    """
    ctx = list(ctx_pairs)
    while True:
        ids, labels = _tokenize_segments(tokenizer, _segments(ctx, inp, oup, loss))
        if len(ids) <= max_len:
            return ctx, ids, labels
        if len(ctx) <= min_ctx:
            return None
        ctx = ctx[:-1]


def encode_conversation(tokenizer, ctx_pairs, inp, oup, max_len: int, loss: str = "all",
                        min_ctx: int = 0, allow_truncate: bool = False):
    fitted = fit_context(tokenizer, ctx_pairs, inp, oup, max_len, loss, min_ctx)
    if fitted is None:
        if not allow_truncate:
            return None
        ids, labels = _tokenize_segments(tokenizer, _segments([], inp, oup, loss))
        ids, labels = ids[-max_len:], labels[-max_len:]
    else:
        _ctx, ids, labels = fitted
    if not any(x != -100 for x in labels):
        return None
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.ones(len(ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


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


def _causal_body(model):
    """Inner transformer + lm_head. LoRA is injected in-place, so this still trains."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def sparse_ce_loss(model, inputs, num_items_in_batch=None):
    """Cross-entropy only on labelled tokens; avoids a [T, vocab] logit tensor."""
    labels = inputs["labels"]
    decoder, lm_head = _causal_body(model)
    out = decoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    hidden = out.last_hidden_state[:, :-1]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    logits = lm_head(hidden[mask]).float()
    loss = torch.nn.functional.cross_entropy(logits, shift_labels[mask], reduction="sum")
    denom = num_items_in_batch if num_items_in_batch is not None else mask.sum()
    return loss / torch.as_tensor(denom, device=loss.device, dtype=loss.dtype).clamp_min(1)


class SparseLogitsTrainer(Trainer):
    """Only run lm_head on supervised positions.

    Qwen3.5 has a 248k vocab, so full [T, V] fp32 logits for a 4k sequence cost
    ~4 GiB (+ grads). ARC answers are a fraction of the sequence, so we gather
    the hidden states at label positions first and project only those.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = sparse_ce_loss(model, inputs, num_items_in_batch)
        return (loss, None) if return_outputs else loss


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


def configure_cuda():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def load_base(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device == "cpu":
        n = max(1, os.cpu_count() or 1)
        torch.set_num_threads(n)
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))
    elif device == "cuda":
        configure_cuda()
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
        attn_implementation="sdpa",
    )
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = {"": "cpu"}
        kwargs.pop("attn_implementation", None)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    # Cache off for training; generate() turns it back on.
    model.config.use_cache = False
    print(f"dtype={dtype} device_map={kwargs.get('device_map')}", flush=True)
    return tok, model


def merge_sft_adapter(model, adapter_dir: str):
    """Load an offline-SFT LoRA adapter and fold it into the base weights."""
    print(f"merging SFT adapter {adapter_dir}", flush=True)
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged.config.use_cache = False
    return merged


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


def generate_grid(model, tok, inp, device: str, max_new: int = 384, ctx_pairs=None,
                  max_len: int = 0):
    if ctx_pairs is None:
        msgs = build_messages(inp)
        try:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        ctx = list(ctx_pairs)
        if max_len:
            # Leave room for the answer; drop demos from the end until it fits.
            budget = max(max_len - max_new, 256)
            fitted = fit_context(tok, ctx, inp, None, budget, min_ctx=0)
            ctx = fitted[0] if fitted is not None else []
        prompt = render_prompt(ctx, inp)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
    dev = model.device if hasattr(model, "device") else device
    inputs = {k: v.to(dev, non_blocking=True) for k, v in inputs.items()}
    eos = [i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0]
    eos = list(dict.fromkeys(eos)) or tok.eos_token_id
    was_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos,
            )
    finally:
        model.config.use_cache = was_cache
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
    if name in ("tr", "atr"):
        return GEOM_FN[name](g)  # self-inverse
    return g


def predict_with_vote(model, tok, test_inp, device: str, vote: bool = True, max_new: int = 384,
                      ctx_pairs=None, max_len: int = 0):
    votes = []
    raws = []
    geoms = GEOM if vote else [GEOM[0]]
    for name, fn in geoms:
        ctx = None
        if ctx_pairs is not None:
            ctx = [(fn(a), fn(b)) for a, b in ctx_pairs]
        g, text = generate_grid(
            model, tok, fn(test_inp), device, max_new=max_new, ctx_pairs=ctx, max_len=max_len
        )
        raws.append((name, text[:200]))
        back = invert_geom(name, g)
        if back is not None:
            votes.append(tuple(tuple(r) for r in back))
    if not votes:
        return None, raws
    best = Counter(votes).most_common(1)[0][0]
    return [list(r) for r in best], raws


def ttt_train_loop(model, examples, args, device: str):
    """Tight LoRA loop: no HuggingFace Trainer, no per-step logging."""
    model.train()
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    n = len(examples)
    bs = max(1, args.batch)
    accum = max(1, args.grad_accum)
    last_loss = None
    opt.zero_grad(set_to_none=True)

    def step_once(step, cur_bs):
        batch = collate([examples[(step * cur_bs + j) % n] for j in range(cur_bs)])
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss = sparse_ce_loss(model, batch) / accum
        loss.backward()
        return float(loss.detach() * accum)

    for step in range(args.steps):
        try:
            last_loss = step_once(step, bs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            print(f"  OOM at batch={bs}, falling back to 1", flush=True)
            opt.zero_grad(set_to_none=True)
            bs = 1
            last_loss = step_once(step, bs)
        if (step + 1) % accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
    if args.steps % accum != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    if args.grad_ckpt:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    model.eval()
    model.config.use_cache = True
    print(f"  last_loss={last_loss:.4f} batch={bs}", flush=True)


def ttt_one_task(base_model, tok, task, args, device: str):
    rng = random.Random(args.seed)
    pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
    views = augment_pairs(pairs, rng, n_color=args.color_augs)
    examples = []
    in_context = args.format == "context"
    for _name, mapped in views:
        if in_context:
            # Leave-one-out: every demo becomes the query once, the rest are context.
            for i in range(len(mapped)):
                ctx = mapped[:i] + mapped[i + 1 :]
                rng.shuffle(ctx)
                ex = encode_conversation(
                    tok, ctx, mapped[i][0], mapped[i][1], args.max_len,
                    loss=args.ttt_loss, allow_truncate=True,
                )
                if ex is not None:
                    examples.append(ex)
        else:
            for inp, oup in mapped:
                examples.append(encode_example(tok, inp, oup, args.max_len))
    if not examples:
        return [None] * len(task["test"]), []

    n_sup = sum(int((ex["labels"] != -100).sum()) for ex in examples)
    print(f"  train_pairs={len(examples)} supervised_tokens={n_sup}", flush=True)

    model = attach_lora(base_model, args.lora_r, device)
    ttt_train_loop(model, examples, args, device)
    preds = []
    debug = []
    for t in task["test"]:
        grid, raws = predict_with_vote(
            model, tok, t["input"], device, vote=not args.no_vote, max_new=args.max_new,
            ctx_pairs=pairs if in_context else None, max_len=args.max_len,
        )
        preds.append(grid)
        debug.append(raws)
    if isinstance(model, PeftModel):
        model.unload()
        try:
            model.delete_adapter("default")
        except Exception:
            pass
    gc.collect()
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
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--grad-ckpt", action="store_true",
                   help="activation checkpointing (saves VRAM, slows TTT a lot)")
    p.add_argument("--max-len", type=int, default=4096,
                   help="token budget per sample (context format drops demos to fit)")
    p.add_argument("--color-augs", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", default="/data/arc-prize-2026/work")
    p.add_argument("--out", default="")
    p.add_argument("--zero-shot", action="store_true", help="skip TTT, generate from frozen model")
    p.add_argument("--no-vote", action="store_true", help="skip geometric vote at inference")
    p.add_argument("--max-new", type=int, default=384)
    p.add_argument(
        "--format", default="context", choices=["single", "context"],
        help="single: one pair per sample (legacy); context: demos in-context, leave-one-out TTT",
    )
    p.add_argument("--ttt-loss", default="all", choices=["all", "last"],
                   help="context format: supervise every assistant turn or only the query answer")
    p.add_argument("--sft-adapter", default="", help="offline SFT LoRA dir to merge into the base first")
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
    if args.sft_adapter:
        model = merge_sft_adapter(model, args.sft_adapter)
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
            ctx = [(ex["input"], ex["output"]) for ex in task["train"]]
            for t in task["test"]:
                g, raws = predict_with_vote(
                    model,
                    tok,
                    t["input"],
                    args.device,
                    vote=not args.no_vote,
                    max_new=args.max_new,
                    ctx_pairs=ctx if args.format == "context" else None,
                    max_len=args.max_len,
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
