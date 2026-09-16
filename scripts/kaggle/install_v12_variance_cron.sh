#!/usr/bin/env bash
# Install 10-min cron on the GPU box for the v13 competition submit
# (08:02 CST 17 Sep 2026 = 00:02 UTC). Does not push a new kernel version.
set -euo pipefail
ROOT=${NVARC_REPO:-/opt/arc-prize-2026}
VENV=${NVARC_KAGGLE_VENV:-/opt/venv_kaggle}
LOCK=/opt/work/nvarc/v13_submit.cron.lock
LOG=/opt/work/nvarc/v13_submit.log
CRON_MARK='nvarc v13 auto-submit'
VERSION=${NVARC_KERNEL_VERSION:-13}

mkdir -p /opt/work/nvarc
chmod +x "$ROOT/scripts/kaggle/submit_v12_variance.py"

# Cron flock file MUST differ from the script's fcntl lock (/tmp/nvarc_v13_submit.lock).
LINE="*/10 * * * * KAGGLE_CONFIG_DIR=/root/.kaggle NVARC_KERNEL_VERSION=$VERSION /usr/bin/flock -n $LOCK $VENV/bin/python $ROOT/scripts/kaggle/submit_v12_variance.py --once --kaggle $VENV/bin/kaggle >> $LOG 2>&1  # $CRON_MARK"
TMP=$(mktemp)
crontab -l 2>/dev/null \
  | grep -v "$CRON_MARK" \
  | grep -v 'nvarc v12 variance A/B' \
  | grep -v 'nvarc kaggle v12 auto-submit' \
  > "$TMP" || true
printf '%s\n' "$LINE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "installed cron VERSION=$VERSION :"
crontab -l
echo "dry-run --once"
KAGGLE_CONFIG_DIR=/root/.kaggle NVARC_KERNEL_VERSION="$VERSION" "$VENV/bin/python" \
  "$ROOT/scripts/kaggle/submit_v12_variance.py" --once --dry-run --kaggle "$VENV/bin/kaggle"
