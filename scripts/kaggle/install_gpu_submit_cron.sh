#!/usr/bin/env bash
# Install a UTC cron poller on the GPU box for Kaggle competition submit.
# Credentials must already be at /root/.kaggle/credentials.json (OAuth file).
set -euo pipefail
ROOT=${NVARC_REPO:-/opt/arc-prize-2026}
VENV=${NVARC_KAGGLE_VENV:-/opt/venv_kaggle}
STATE_DIR=$ROOT/notebooks/nvarc_2026
LOCK=/opt/work/nvarc/kaggle_v12_submit.lock
LOG=/opt/work/nvarc/kaggle_v12_submit.log
CRON_MARK='nvarc kaggle v12 auto-submit'

python3 -m venv "$VENV"
WHEELS=/tmp/kaggle_linux
if [ -d "$WHEELS" ] && ls "$WHEELS"/kaggle-*.whl >/dev/null 2>&1; then
  "$VENV/bin/pip" install -q --no-index --find-links "$WHEELS" 'kaggle==2.2.4'
else
  "$VENV/bin/pip" install -q -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com 'kaggle==2.2.4'
fi

mkdir -p /root/.kaggle /opt/work/nvarc "$STATE_DIR"
chmod 700 /root/.kaggle
if [ ! -f /root/.kaggle/credentials.json ]; then
  echo "FAIL: missing /root/.kaggle/credentials.json" >&2
  exit 1
fi
chmod 600 /root/.kaggle/credentials.json

"$VENV/bin/kaggle" kernels status wenbozhang2026/nvarc-qwen3-4b-ttt-2026-ceiling

LINE="*/10 * * * * KAGGLE_CONFIG_DIR=/root/.kaggle /usr/bin/flock -n $LOCK $VENV/bin/python $ROOT/scripts/kaggle/submit_half2_when_quota.py --once --kaggle $VENV/bin/kaggle >> $LOG 2>&1  # $CRON_MARK"
TMP=$(mktemp)
crontab -l 2>/dev/null | grep -v "$CRON_MARK" > "$TMP" || true
printf '%s\n' "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "installed cron:"
crontab -l | grep "$CRON_MARK"
echo "dry-run --once"
KAGGLE_CONFIG_DIR=/root/.kaggle "$VENV/bin/python" \
  "$ROOT/scripts/kaggle/submit_half2_when_quota.py" --once --kaggle "$VENV/bin/kaggle"
