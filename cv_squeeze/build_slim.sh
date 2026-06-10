#!/usr/bin/env bash
# Single-step slim build (no TRT engine bake — saves ~15GB).
# Image will use best.pt directly via cv_manager.py's fallback path.
#
# Usage:  ./build_slim.sh [tag]    (default tag = v8m-outline-3ep)
set -euo pipefail

CV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PT="$CV_DIR/src/weights/best.pt"
TAG="${1:-v8m-outline-3ep}"
TEAM="${TEAM_NAME:-nobrainnohack}"
IMAGE="$TEAM-cv:$TAG"

if [[ ! -f "$PT" ]]; then
    echo "[slim-bake] $PT not found" >&2
    exit 1
fi

echo "[slim-bake] building $IMAGE from Dockerfile.slim ..."
(cd "$CV_DIR" && docker build -t "$IMAGE" -f Dockerfile.slim .)

SIZE=$(docker images "$IMAGE" --format '{{.Size}}')
echo "[slim-bake] done. image size: $SIZE"
echo "[slim-bake] next: 'til test cv $TAG', then 'til submit cv $TAG'"
