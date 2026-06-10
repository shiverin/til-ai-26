#!/usr/bin/env bash
# Build the v8m-ft-ep1-tuned CV image from cv_squeeze/ with the TensorRT
# engine baked in. Two-stage bake: build image, run TRT export inside it
# with --gpus all (volume-mount the host weights dir so the .engine drops
# back to the host), then rebuild — the second build copies the .engine
# in via `COPY src ./src`. Idempotent: skips the slow export if the
# engine is already newer than best.pt.
#
# Unlike the teammate's cv/build.sh, this script uses `docker build`
# directly (NOT `til build cv`, which hardcodes til_folder/cv and would
# build the teammate's tree instead of cv_squeeze/). Image is still
# tagged as `<team>-cv:<tag>` so `til submit cv <tag>` works.
#
# Engine is exported at imgsz=[864, 1536] (1536-wide rect, 16:9 — matches
# the sweep winner in finetune/runs/sweeps/v8m_ft_ep1_sweep.json).
#
# Usage:  ./build.sh [tag]         (default tag = v8m-ft-ep1-tuned)
set -euo pipefail

CV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$CV_DIR/src/weights"
PT="$WEIGHTS_DIR/best.pt"
ENGINE="$WEIGHTS_DIR/best.engine"
TAG="${1:-v8m-ft-ep1-tuned}"
TEAM="${TEAM_NAME:-nobrainnohack}"
IMAGE="$TEAM-cv:$TAG"

if [[ ! -f "$PT" ]]; then
    echo "[bake] $PT not found" >&2
    exit 1
fi

if [[ -f "$ENGINE" && "$ENGINE" -nt "$PT" ]]; then
    echo "[bake] $ENGINE is newer than best.pt — skipping engine build"
else
    echo "[bake] step 1/3 — initial docker build $IMAGE (no engine yet) ..."
    (cd "$CV_DIR" && docker build -t "$IMAGE" -f Dockerfile .)

    echo "[bake] step 2/3 — TensorRT export inside $IMAGE (3-8 min) ..."
    docker run --rm --gpus all \
        -v "$WEIGHTS_DIR:/workspace/src/weights" \
        --entrypoint python "$IMAGE" -c "
from ultralytics import YOLO
YOLO('/workspace/src/weights/best.pt').export(
    format='engine', half=True, dynamic=True, imgsz=[864, 1536], batch=16, opset=18,
)"
    [[ -f "$ENGINE" ]] || { echo "[bake] engine missing after export" >&2; exit 1; }
fi

echo "[bake] step 3/3 — rebuilding $IMAGE with .engine baked in ..."
(cd "$CV_DIR" && docker build -t "$IMAGE" -f Dockerfile .)

echo "[bake] done. engine: $(ls -lh "$ENGINE" | awk '{print $5}')"
echo "[bake] next: 'til test cv $TAG' (boots in ~10s now), then 'til submit cv $TAG'"
