#!/bin/bash
# After rule ablation + vote probe: run NVARC Qwen3-4B TTT locally on the 3090.
# Competition submit is separate (Kaggle L4x4). This queue is the local diagnostic loop.
set -u
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/opt/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOTDIR=${ROOTDIR:-/opt/arc-rule}
PY=${PY:-/opt/venv/bin/python}
W=${W:-/opt/work/nvarc}
MODEL=${MODEL:-/opt/models/qwen3_4b_grids15_sft139}
WAIT_PID=${WAIT_PID:-}
cd "$ROOTDIR"
mkdir -p "$W"

log() { echo "[$(date '+%F %T')] $*"; }

if [ -n "$WAIT_PID" ]; then
  log "waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "previous queue finished, GPU free"
fi

if [ ! -f "$MODEL/config.json" ]; then
  # download may unpack one extra directory level
  if [ -f "$MODEL/transformers/bfloat16/1/config.json" ]; then
    MODEL="$MODEL/transformers/bfloat16/1"
  else
    log "NVARC weights missing at $MODEL — skip"
    exit 1
  fi
fi

HOLD=data/rules/holdout_ids.txt
# First local pass: reuse our TTT loop as a connectivity check (custom 16-token
# tokenizer will likely need the Unsloth notebook path; this still verifies load).
ZS_OPT="--device cuda --format context --zero-shot --max-len 4096 --no-vote --max-new 768 --work-dir $W"
TTT_OPT="--device cuda --format context --steps 16 --lora-r 16 --max-len 4096 --batch 1 --grad-ckpt \
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

run zs_nvarc_concept "$W/zs_nvarc_concept.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL \
    --tasks-dir /opt/ext/ConceptARC/corpus --limit 32 $ZS_OPT \
    --out $W/zs_nvarc_concept.json

run ttt_nvarc_hold "$W/ttt_nvarc_hold.json" \
  $PY -u scripts/ttt_qwen.py --model $MODEL \
    --split training --ids-file $HOLD --limit 8 $TTT_OPT \
    --out $W/ttt_nvarc_hold.json

log "nvarc queue done"
