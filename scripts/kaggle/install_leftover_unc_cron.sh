#!/usr/bin/env bash
# GPU2: push leftover-B 8×6 kernel (optional), install 08:00 CST 19 Sep submit + 30min watch.
set -euo pipefail
ROOT=${NVARC_REPO:-/root/workspace/arc-prize-2026}
VENV=${NVARC_KAGGLE_VENV:-/opt/venv_kaggle}
WORKDIR=/opt/work/nvarc
SUBMIT_LOCK=$WORKDIR/leftover_unc_submit.cron.lock
WATCH_LOCK=$WORKDIR/leftover_unc_watch.cron.lock
SUBMIT_LOG=$WORKDIR/leftover_unc_submit.log
WATCH_LOG=$WORKDIR/leftover_unc_watch.log
SUBMIT_MARK='nvarc leftover_unc auto-submit'
WATCH_MARK='nvarc leftover_unc submission watch'
PUSH=1
if [ "${1:-}" = "--no-push" ]; then
  PUSH=0
fi

mkdir -p "$WORKDIR"
chmod +x \
  "$ROOT/scripts/kaggle/submit_leftover_unc.py" \
  "$ROOT/scripts/kaggle/watch_submission.py" \
  "$ROOT/scripts/kaggle/smoke_leftover_unc.py" \
  "$ROOT/scripts/kaggle/push_leftover_unc.py"

export KAGGLE_CONFIG_DIR=/root/.kaggle
export NVARC_KAGGLE=$VENV/bin/kaggle
export NVARC_LEFTOVER_KERNEL=$WORKDIR/leftover_unc_kernel.json
export NVARC_LEFTOVER_STATE=$WORKDIR/leftover_unc_submit_state.json
export NVARC_WATCH_STATE=$WORKDIR/leftover_unc_watch.json
export NVARC_WATCH_LABEL=leftover_unc

echo "==== smoke (pre-push notebook / credentials) ===="
"$VENV/bin/python" "$ROOT/scripts/kaggle/smoke_leftover_unc.py" \
  --kaggle "$VENV/bin/kaggle" --allow-missing-kernel-info
if [ "$PUSH" = "1" ]; then
  echo "==== push kernel NvidiaL4 ===="
  "$VENV/bin/python" "$ROOT/scripts/kaggle/push_leftover_unc.py" --folder "$ROOT/notebooks/nvarc_2026"
fi

echo "==== install crons ===="
TMP=$(mktemp)
crontab -l 2>/dev/null \
  | grep -v "$SUBMIT_MARK" \
  | grep -v "$WATCH_MARK" \
  | grep -v 'nvarc single8x6 auto-submit' \
  | grep -v 'nvarc single8x6 submission watch' \
  | grep -v 'nvarc v13 auto-submit' \
  | grep -v 'nvarc v14 submission watch' \
  | grep -v 'nvarc v12 variance A/B' \
  | grep -v 'nvarc kaggle v12 auto-submit' \
  > "$TMP" || true
# 10-min poll; Python no-ops until 08:00 CST (2026-09-19T00:00:00Z)
# flock file is leftover_unc_submit.cron.lock; fcntl lock is /tmp/nvarc_leftover_unc_submit.lock
printf '%s\n' "*/10 * * * * KAGGLE_CONFIG_DIR=/root/.kaggle NVARC_LEFTOVER_KERNEL=$WORKDIR/leftover_unc_kernel.json NVARC_LEFTOVER_STATE=$WORKDIR/leftover_unc_submit_state.json NVARC_LEFTOVER_LOCK=/tmp/nvarc_leftover_unc_submit.lock /usr/bin/flock -n $SUBMIT_LOCK $VENV/bin/python $ROOT/scripts/kaggle/submit_leftover_unc.py --once --kaggle $VENV/bin/kaggle >> $SUBMIT_LOG 2>&1  # $SUBMIT_MARK" >> "$TMP"
# 30-min duration watch; waits until submit writes a ref
printf '%s\n' "*/30 * * * * KAGGLE_CONFIG_DIR=/root/.kaggle NVARC_WATCH_STATE=$WORKDIR/leftover_unc_watch.json NVARC_SINGLE_STATE=$WORKDIR/leftover_unc_submit_state.json NVARC_WATCH_LOCK=/tmp/nvarc_leftover_unc_watch.lock NVARC_WATCH_LABEL=leftover_unc /usr/bin/flock -n $WATCH_LOCK $VENV/bin/python $ROOT/scripts/kaggle/watch_submission.py --once --kaggle $VENV/bin/kaggle >> $WATCH_LOG 2>&1  # $WATCH_MARK" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "crontab:"
crontab -l

echo "==== dry-run submit (should be too early or kernel RUNNING) ===="
"$VENV/bin/python" "$ROOT/scripts/kaggle/submit_leftover_unc.py" --once --dry-run --kaggle "$VENV/bin/kaggle" || true
echo "==== watch once (should wait for ref) ===="
NVARC_WATCH_STATE=$WORKDIR/leftover_unc_watch.json \
NVARC_SINGLE_STATE=$WORKDIR/leftover_unc_submit_state.json \
NVARC_WATCH_LOCK=/tmp/nvarc_leftover_unc_watch.lock \
NVARC_WATCH_LABEL=leftover_unc \
  "$VENV/bin/python" "$ROOT/scripts/kaggle/watch_submission.py" --once --kaggle "$VENV/bin/kaggle" || true

echo "==== smoke (post-install) ===="
"$VENV/bin/python" "$ROOT/scripts/kaggle/smoke_leftover_unc.py" --kaggle "$VENV/bin/kaggle"
echo "kernel info:"
cat "$WORKDIR/leftover_unc_kernel.json" 2>/dev/null || echo "(none)"
