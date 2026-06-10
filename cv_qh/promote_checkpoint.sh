#!/usr/bin/env bash
# Copy a trained RF-DETR checkpoint into src/weights/best.pt for the Docker build.
#
# Usage:
#   ./promote_checkpoint.sh [path/to/checkpoint.pth]
#
# Default: weights/base/checkpoint_best_ema.pth
set -euo pipefail

CV_QH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-$CV_QH_DIR/weights/base/checkpoint_best_ema.pth}"
DST="$CV_QH_DIR/src/weights/best.pt"

if [[ ! -f "$SRC" ]]; then
    echo "[promote] source checkpoint not found: $SRC" >&2
    exit 1
fi

mkdir -p "$(dirname "$DST")"
cp "$SRC" "$DST"
echo "[promote] copied $SRC ($(du -h "$SRC" | cut -f1)) → $DST"
echo "[promote] next: 'til build cv <tag>' from $CV_QH_DIR"
