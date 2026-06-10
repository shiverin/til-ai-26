#!/usr/bin/env bash
# Watch weights/base/checkpoint_best_ema.pth; on every fresh save (new mtime),
# snapshot + build + submit under an auto-numbered tag. Each submission then
# cools down for SUBMIT_COOLDOWN seconds (default 600) before we re-arm the
# watcher — so chained best-saves can't queue overlapping submissions.
#
# Usage:
#   ./autosubmit.sh [tag-prefix]            # default prefix: rfdetr-best
#   SUBMIT_COOLDOWN=300 ./autosubmit.sh     # 5-min cooldown instead of 10
#
# Run in a separate terminal alongside the training. Ctrl-C to stop.
set -euo pipefail

PREFIX="${1:-rfdetr-best}"
COOLDOWN="${SUBMIT_COOLDOWN:-600}"
POLL="${WATCH_POLL_SEC:-15}"

CV_QH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVE="$CV_QH_DIR/weights/base/checkpoint_best_ema.pth"

echo "[autosubmit] watching $LIVE (poll=${POLL}s, cooldown=${COOLDOWN}s, prefix=$PREFIX)"
echo "[autosubmit] Ctrl-C to stop."

# Initial mtime (0 if file doesn't exist yet). New saves bump this.
last_mtime=0
if [[ -f "$LIVE" ]]; then
    last_mtime=$(stat -c %Y "$LIVE")
fi

counter=1
while true; do
    if [[ -f "$LIVE" ]]; then
        cur_mtime=$(stat -c %Y "$LIVE")
        if [[ "$cur_mtime" -gt "$last_mtime" ]]; then
            suffix="$(printf '%02d' "$counter")"
            echo
            echo "[autosubmit] new best detected (mtime $cur_mtime > $last_mtime) — submitting as ${PREFIX}-${suffix}"
            # Re-use submit_epoch.sh's snapshot+build+submit+sleep pipeline.
            # It already prints progress and honours SUBMIT_COOLDOWN.
            SUBMIT_COOLDOWN="$COOLDOWN" "$CV_QH_DIR/submit_epoch.sh" "${PREFIX}-${suffix}" || {
                echo "[autosubmit] submission ${PREFIX}-${suffix} failed; will retry on next save" >&2
            }
            # Re-read mtime AFTER the cooldown — the file may have been overwritten
            # during the sleep, in which case we want to fire again immediately.
            last_mtime=$(stat -c %Y "$LIVE")
            counter=$((counter + 1))
        fi
    fi
    sleep "$POLL"
done
