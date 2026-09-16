#!/usr/bin/env bash
# After n8x6-A finishes, stop the old unpooled two.sh and start B with v14 pooling.
# Does not touch the running A workers.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${NVARC_WORK:-/opt/work/nvarc}
A_NAME=${NVARC_N8X6_A:-eval120_n8x6_a}
B_NAME=${NVARC_N8X6_B:-eval120_n8x6_b}
SUM=$WORK/eval120_n8x6_two
LOGDIR=$WORK/eval120_n8x6_two_logs
VENV=${NVARC_VENV:-/opt/venv_nvarc}
A_DONE=$WORK/$A_NAME/done
LOCK=$SUM/switch.lock

mkdir -p "$SUM" "$LOGDIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] switch already running" | tee -a "$LOGDIR/switch.log"
  exit 0
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/switch.log"; }

if [ ! -f "$SUM/t0.epoch" ]; then
  log "missing $SUM/t0.epoch; refusing to guess T0"
  exit 1
fi

log "waiting for $A_DONE (will not interrupt A)"
while [ ! -f "$A_DONE" ]; do
  sleep 15
done
log "A done; intercepting old run_n8x6_two.sh before unpooled B"

# Parent two.sh: after A/done it scores A then starts unpooled B.
pkill -f "bash run_n8x6_two.sh" || true
sleep 2
# If unpooled B already spawned, replace it with the pooled starter.
if pgrep -f "eval120_n8x6_b" >/dev/null 2>&1; then
  log "unpooled B already running; stopping it so pooled B can start"
  pkill -f "run_local.sh ${B_NAME}" || true
  pkill -f "starter.py .*${B_NAME}/outputs" || true
  sleep 5
fi
if pgrep -f "schedule_local_v14.py" >/dev/null 2>&1; then
  log "schedule_local_v14.py already running; exit"
  exit 0
fi

export NVARC_T0=$(cat "$SUM/t0.epoch")
export NVARC_FINISH_ALL=${NVARC_FINISH_ALL:-1}
export NVARC_PYTHON="$VENV/bin/python"
export NVARC_WORK="$WORK"
cd "$HERE"
log "starting schedule_local_v14.py T0=$NVARC_T0 finish_all=$NVARC_FINISH_ALL"
exec "$VENV/bin/python" -u "$HERE/schedule_local_v14.py" >>"$LOGDIR/console.log" 2>&1
