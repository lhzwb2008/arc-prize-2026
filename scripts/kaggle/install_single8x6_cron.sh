#!/usr/bin/env bash
# GPU2: push single-pass 8×6 kernel (optional), install 08:00 CST submit + 30min watch.
set -euo pipefail
ROOT=${NVARC_REPO:-/root/workspace/arc-prize-2026}
VENV=${NVARC_KAGGLE_VENV:-/opt/venv_kaggle}
WORKDIR=/opt/work/nvarc
SUBMIT_LOCK=$WORKDIR/single8x6_submit.cron.lock
WATCH_LOCK=$WORKDIR/single8x6_watch.cron.lock
SUBMIT_LOG=$WORKDIR/single8x6_submit.log
WATCH_LOG=$WORKDIR/single8x6_watch.log
SUBMIT_MARK='nvarc single8x6 auto-submit'
WATCH_MARK='nvarc single8x6 submission watch'
PUSH=1
if [ "${1:-}" = "--no-push" ]; then
  PUSH=0
fi

mkdir -p "$WORKDIR"
chmod +x \
  "$ROOT/scripts/kaggle/submit_single8x6.py" \
  "$ROOT/scripts/kaggle/watch_submission.py" \
  "$ROOT/scripts/kaggle/smoke_single8x6.py" \
  "$ROOT/scripts/kaggle/push_single8x6.py"

export KAGGLE_CONFIG_DIR=/root/.kaggle
export NVARC_KAGGLE=$VENV/bin/kaggle

echo "==== smoke (pre-push notebook / credentials) ===="
"$VENV/bin/python" "$ROOT/scripts/kaggle/smoke_single8x6.py" \
  --kaggle "$VENV/bin/kaggle" --allow-missing-kernel-info
if [ "$PUSH" = "1" ]; then
  echo "==== push kernel NvidiaL4 ===="
  "$VENV/bin/python" "$ROOT/scripts/kaggle/push_single8x6.py" --folder "$ROOT/notebooks/nvarc_2026"
fi

echo "==== install crons ===="
TMP=$(mktemp)
crontab -l 2>/dev/null \
  | grep -v "$SUBMIT_MARK" \
  | grep -v "$WATCH_MARK" \
  | grep -v 'nvarc v13 auto-submit' \
  | grep -v 'nvarc v14 submission watch' \
  | grep -v 'nvarc v12 variance A/B' \
  | grep -v 'nvarc kaggle v12 auto-submit' \
  > "$TMP" || true
# 10-min poll; Python no-ops until 08:00 CST (2026-09-18T00:00:00Z)
printf '%s\n' "*/10 * * * * KAGGLE_CONFIG_DIR=/root/.kaggle /usr/bin/flock -n $SUBMIT_LOCK $VENV/bin/python $ROOT/scripts/kaggle/submit_single8x6.py --once --kaggle $VENV/bin/kaggle >> $SUBMIT_LOG 2>&1  # $SUBMIT_MARK" >> "$TMP"
# 30-min duration watch; waits until submit writes a ref
printf '%s\n' "*/30 * * * * KAGGLE_CONFIG_DIR=/root/.kaggle /usr/bin/flock -n $WATCH_LOCK $VENV/bin/python $ROOT/scripts/kaggle/watch_submission.py --once --kaggle $VENV/bin/kaggle >> $WATCH_LOG 2>&1  # $WATCH_MARK" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "crontab:"
crontab -l

echo "==== dry-run submit (should be too early or kernel RUNNING) ===="
"$VENV/bin/python" "$ROOT/scripts/kaggle/submit_single8x6.py" --once --dry-run --kaggle "$VENV/bin/kaggle" || true
echo "==== watch once (should wait for ref) ===="
"$VENV/bin/python" "$ROOT/scripts/kaggle/watch_submission.py" --once --kaggle "$VENV/bin/kaggle" || true

echo "==== smoke (post-install) ===="
"$VENV/bin/python" "$ROOT/scripts/kaggle/smoke_single8x6.py" --kaggle "$VENV/bin/kaggle"
echo "kernel info:"
cat "$WORKDIR/single8x6_kernel.json" 2>/dev/null || echo "(none)"
