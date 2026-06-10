#!/usr/bin/env bash
# Submit remaining v8m aug variants to TIL with 660s (11 min) sleep between.
# Resumes from mosaic_mixup ep2 (assuming heavy_color + heavy_copy_paste already submitted).
set -euo pipefail

TAGS=(
    v8m-aug-mosaic_mixup-ep2-imgsz1536
    v8m-aug-mosaic_mixup-ep3-imgsz1536
    v8m-aug-light-ep2-imgsz1536
    v8m-aug-light-ep3-imgsz1536
)

SLEEP_BETWEEN=660  # 11 minutes

for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    echo
    echo "===== $(date +%H:%M:%S) submitting [$((i+1))/${#TAGS[@]}] $tag ====="
    til submit cv "$tag"
    if [ $((i+1)) -lt ${#TAGS[@]} ]; then
        echo "===== $(date +%H:%M:%S) sleeping ${SLEEP_BETWEEN}s before next ====="
        sleep "$SLEEP_BETWEEN"
    fi
done

echo
echo "===== $(date +%H:%M:%S) all submissions done ====="
