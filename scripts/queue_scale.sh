#!/bin/bash
# After queue_rule.sh: remaining diagnostics, then overnight GPU so the 3090
# does not sit idle after vote.
#
#   zs_C_concept          C-arm ConceptARC zero-shot vs zs_R_concept 12/160
#   vote32                8-view vote on ConceptARC first 32 (SFT-R)
#   ttt_R_eval_proposed   120 public eval, SFT-R + proposed-rule TTT (beat DSL 1/120?)
#   ttt_R_concept         ConceptARC full TTT vs the 7.5% zero-shot
#   ttt_R_hold_propose8   40 holdout, 8 sampled rules (default was 3) vs 10/40
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
  log "waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "previous queue finished, GPU free"
fi

HOLD=data/rules/holdout_ids.txt
ZS_OPT="--device cuda --format context --zero-shot --max-len 4096 --max-new 768 --work-dir $W"
TTT_OPT="--device cuda --format context --steps 32 --lora-r 16 --max-len 4096 --batch 1 --grad-ckpt \
  --color-augs 2 --no-vote --max-new 768 --work-dir $W"

run() {
  local name=$1 out=$2; shift 2
  if [ -e "$W/$name.done" ]; then log "skip $name (done)"; return; fi
  log "start $name"
  "$@" > "$W/$name.log" 2>&1
  local rc=$?
  log "exit $name rc=$rc"
  [ $rc -eq 0 ] && [ -e "$out" ] && touch "$W/$name.done"
}

# ---- C-arm ConceptARC zero-shot (skipped by the running queue_rule.sh) ----
run zs_C_concept "$W/zs_C_concept.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_C/adapter \
    --tasks-dir /opt/ext/ConceptARC/corpus --limit 0 --no-vote $ZS_OPT \
    --out $W/zs_C_concept.json

# ---- 8-view vote lift on a ConceptARC subset? ----
run zs_R_concept_vote "$W/zs_R_concept_vote.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_R/adapter --rule-mode \
    --tasks-dir /opt/ext/ConceptARC/corpus --limit 32 $ZS_OPT \
    --out $W/zs_R_concept_vote.json

# ---- overnight: can proposed-rule TTT beat DSL's 1/120 on public eval? ----
run ttt_R_eval_proposed "$W/ttt_R_eval_proposed.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_R/adapter --rule-mode \
    --split evaluation --limit 0 $TTT_OPT --out $W/ttt_R_eval_proposed.json

# ---- overnight: does TTT lift ConceptARC above the 12/160 zero-shot? ----
run ttt_R_concept "$W/ttt_R_concept.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_R/adapter --rule-mode \
    --tasks-dir /opt/ext/ConceptARC/corpus --limit 0 $TTT_OPT \
    --out $W/ttt_R_concept.json

# ---- overnight: more rule samples on the 40 holdout (close the 10 vs 11 gap?) ----
run ttt_R_hold_propose8 "$W/ttt_R_hold_propose8.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL --sft-adapter $W/sft_R/adapter --rule-mode \
    --split training --ids-file $HOLD --limit 0 --rule-samples 8 --rule-temp 0.9 \
    $TTT_OPT --out $W/ttt_R_hold_propose8.json

log "scale+night queue done (next is queue_nvarc.sh)"
