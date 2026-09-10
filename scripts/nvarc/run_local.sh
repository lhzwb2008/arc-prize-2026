#!/usr/bin/env bash
# Run the NVARC TTT pipeline locally (3090 box) and score it.
#
#   bash run_local.sh <run_name> <hours> [starter extra args...]
#   e.g. bash run_local.sh smoke4 2 --keys 0934a4d8,36a08778,981571dc,aa4ec2a5
#        bash run_local.sh eval120 30
#
# Env overrides: NVARC_VENV, NVARC_MODEL, NVARC_DATA, NVARC_SOL, NVARC_WORK
set -euo pipefail

RUN=${1:?run name}; HOURS=${2:?hours}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${NVARC_VENV:-/opt/venv_nvarc}
export NVARC_MODEL=${NVARC_MODEL:-/opt/models/qwen3_4b_grids15_sft139}
DATA=${NVARC_DATA:-/opt/data/kaggle/arc-agi_evaluation_challenges.json}
SOL=${NVARC_SOL:-/opt/data/kaggle/arc-agi_evaluation_solutions.json}
WORK=${NVARC_WORK:-/opt/work/nvarc}/$RUN
mkdir -p "$WORK"

export UNSLOTH_DISABLE_STATISTICS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
export TOKENIZERS_PARALLELISM=false
[ -x /usr/local/cuda/bin/ptxas ] && export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

cd "$HERE"
echo "run=$RUN hours=$HOURS work=$WORK extra=$*"
date
"$VENV/bin/python" starter.py --data "$DATA" --out "$WORK/outputs" --hours "$HOURS" "$@"
echo "starter_exit=$?"
date
FINALIZE_KEYS=()
prev=""
for a in "$@"; do
  if [ "$prev" = "--keys" ]; then FINALIZE_KEYS=(--keys "$a"); fi
  prev=$a
done
"$VENV/bin/python" finalize.py --data "$DATA" --solutions "$SOL" \
    --outputs "$WORK/outputs" --submission "$WORK/submission.json" --report "$WORK/report.json" \
    "${FINALIZE_KEYS[@]}"
touch "$WORK/done"
