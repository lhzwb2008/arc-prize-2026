#!/bin/bash
# "State the rule, then draw the grid" ablation. Runs on the RTX 3090 box after the
# baseline queue (SFT-A + bare TTT evals) has released the GPU.
#
#   arm R : sft_qwen.py --rule-mode    (Rule: ... Output: grid)
#   arm C : sft_qwen.py --strip-rules  (same tasks, same augmentations, plain grid)
# both arms: 371 rule-labelled ARC training tasks + their RE-ARC variants + 6000
# BARC-Heavy synthetic tasks, minus 40 held-out seed tasks (data/rules/holdout_ids.txt).
#
# Kept (diagnostic):
#   1. ConceptARC zero-shot R then C   -> does writing a rule help on core concepts?
#   2. 40 held-out tasks, TTT: oracle / proposed / none
# Dropped (120 public eval is 0 under every current recipe, no gradient):
#   zs_C_eval, ttt_R_eval, ttt_C_eval  (touched as *.done on the GPU box)
#
# Every step is skipped when its output exists, so the script is restartable.
set -u
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/opt/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOTDIR=${ROOTDIR:-/opt/arc-rule}
PY=${PY:-/opt/venv/bin/python}
W=${W:-/opt/work/rule}
MODEL=${MODEL:-/opt/models/Qwen3.5-4B}
WAIT_PID=${WAIT_PID:-}
cd "$ROOTDIR"
mkdir -p "$W"

log() { echo "[$(date '+%F %T')] $*"; }

if [ -n "$WAIT_PID" ]; then
  log "waiting for pid $WAIT_PID (baseline queue)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "baseline queue finished, GPU free"
fi

RULES=data/rules/task_rules.json
HOLD=data/rules/holdout_ids.txt
SFT_DATA="--augs-per-task 6 --rearc-dir /opt/rearc/re_arc --rearc-per-task 6 \
  --barc-heavy /opt/barc/data_100k.jsonl --barc-heavy-max 6000 --barc-heavy-per-task 1 \
  --rules-json $RULES --exclude-ids $HOLD --seed 0"
SFT_OPT="--max-len 2048 --lora-r 32 --lr 1e-4 --batch 1 --grad-accum 8 --save-steps 1000 --dump-samples 20"
TTT_OPT="--device cuda --format context --steps 32 --lora-r 16 --max-len 4096 --batch 1 --grad-ckpt \
  --color-augs 2 --no-vote --max-new 768 --work-dir $W"
ZS_OPT="--device cuda --format context --zero-shot --max-len 4096 --no-vote --max-new 768 --work-dir $W"

run() {  # run <name> <outfile> <cmd...>   (json outputs are partial until $name.done exists)
  local name=$1 out=$2; shift 2
  if [ -e "$W/$name.done" ]; then log "skip $name (done)"; return; fi
  log "start $name"
  "$@" > "$W/$name.log" 2>&1
  local rc=$?
  log "exit $name rc=$rc"
  [ $rc -eq 0 ] && [ -e "$out" ] && touch "$W/$name.done"
}

# ---- 1. two SFT arms ---------------------------------------------------------
run sft_R "$W/sft_R/adapter/adapter_model.safetensors" \
  $PY -u scripts/sft_qwen.py --model $MODEL --rule-mode   $SFT_DATA $SFT_OPT --out-dir $W/sft_R
run zs_R_eval "$W/zs_R_eval.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_R/adapter --rule-mode \
    --split evaluation --limit 0 $ZS_OPT --out $W/zs_R_eval.json
run zs_R_concept "$W/zs_R_concept.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_R/adapter --rule-mode \
    --tasks-dir /opt/ext/ConceptARC/corpus --limit 0 $ZS_OPT --out $W/zs_R_concept.json

run sft_C "$W/sft_C/adapter/adapter_model.safetensors" \
  $PY -u scripts/sft_qwen.py --model $MODEL --strip-rules $SFT_DATA $SFT_OPT --out-dir $W/sft_C
run zs_C_concept "$W/zs_C_concept.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_C/adapter \
    --tasks-dir /opt/ext/ConceptARC/corpus --limit 0 $ZS_OPT --out $W/zs_C_concept.json

# ---- 2. 40 held-out training tasks: oracle vs proposed vs none ------------------
run ttt_R_hold_oracle "$W/ttt_R_hold_oracle.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_R/adapter --rule-mode \
    --rules-json $RULES --split training --ids-file $HOLD --limit 0 $TTT_OPT --out $W/ttt_R_hold_oracle.json
run ttt_R_hold_proposed "$W/ttt_R_hold_proposed.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_R/adapter --rule-mode \
    --split training --ids-file $HOLD --limit 0 $TTT_OPT --out $W/ttt_R_hold_proposed.json
run ttt_C_hold "$W/ttt_C_hold.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_C/adapter \
    --split training --ids-file $HOLD --limit 0 $TTT_OPT --out $W/ttt_C_hold.json

log "queue done"
