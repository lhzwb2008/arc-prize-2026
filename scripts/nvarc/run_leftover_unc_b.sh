#!/usr/bin/env bash
# GPU2: leftover-B by A-uncertainty/cost until the Kaggle-12h-equivalent wall.
# Does not finish all of B. Does not touch eval120_n8x6_b.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
VENV=${NVARC_VENV:-/opt/venv_nvarc}
B_NAME=${NVARC_PARTIAL_B:-eval120_n8x6_b_unc}
SUM_NAME=${NVARC_PARTIAL_SUM:-eval120_n8x6_unc}
SUM=$WORK/$SUM_NAME
SESSION=${NVARC_TMUX_SESSION:-leftover-unc-b}

if [ "$B_NAME" = "eval120_n8x6_b" ]; then
  echo "refusing to overwrite eval120_n8x6_b"
  exit 1
fi
if pgrep -f "schedule_partial_b.py" >/dev/null 2>&1; then
  echo "schedule_partial_b.py already running"
  pgrep -af "schedule_partial_b.py" || true
  exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists"
  exit 0
fi

mkdir -p "$SUM"
export NVARC_PYTHON="$VENV/bin/python"
export NVARC_PARTIAL_B="$B_NAME"
export NVARC_PARTIAL_SUM="$SUM_NAME"
export NVARC_POOL_MODE=keep-primary

echo "==== dry-run cap ===="
"$VENV/bin/python" -u "$HERE/schedule_partial_b.py" --dry-run

echo "==== start tmux $SESSION ===="
tmux new-session -d -s "$SESSION" -c "$HERE" \
  "export NVARC_PYTHON=$VENV/bin/python NVARC_PARTIAL_B=$B_NAME NVARC_PARTIAL_SUM=$SUM_NAME NVARC_POOL_MODE=keep-primary NVARC_WORK=$WORK;
   $VENV/bin/python -u $HERE/schedule_partial_b.py 2>&1 | tee -a $SUM/console.log;
   echo leftover_unc_b_exit=\$? | tee -a $SUM/console.log"
echo "started tmux $SESSION  log=$SUM/console.log"
tmux ls
