#!/usr/bin/env bash
# Submit the 3 no-retrain variants of best_v8m_ep1.pt (0.727 baseline)
# with 660s (11 min) cooldown between to respect the leaderboard pacing.
set -euo pipefail

TAGS=(
    v8m-ep1-tta-1536
    v8m-ep1-bilateral-1536
    v8m-ep1-tta-bilateral-1536
)

SLEEP_BETWEEN=660

for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    echo
    echo "===== $(date +%H:%M:%S) submitting [$((i+1))/${#TAGS[@]}] $tag ====="
    til submit cv "$tag"
    if [ $((i+1)) -lt ${#TAGS[@]} ]; then
        echo "===== $(date +%H:%M:%S) sleeping ${SLEEP_BETWEEN}s ====="
        sleep "$SLEEP_BETWEEN"
    fi
done

echo
echo "===== $(date +%H:%M:%S) all 3 submissions done ====="
