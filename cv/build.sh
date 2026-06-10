#!/usr/bin/env bash
# Build the CV image with the TensorRT engine baked in.
#
# Stock `docker build` has no GPU access, so the engine can't be exported
# from a Dockerfile RUN step on this workbench. Workaround: build the image,
# run the engine export inside it (with GPU + a volume mount to drop the
# engine on the host), then rebuild — the second build copies the engine in
# via `COPY src ./src`. Idempotent: skips the slow export if the engine is
# already newer than best.pt.
#
# Usage:  ./build.sh [tag]         (default tag = latest)
set -euo pipefail

CV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$CV_DIR/src/weights"
PT="$WEIGHTS_DIR/best.pt"
ENGINE="$WEIGHTS_DIR/best.engine"
TAG="xyz"

if [[ ! -f "$PT" ]]; then
    echo "[bake] $PT not found — run finetune/promote.py first" >&2
    exit 1
fi

if [[ -f "$ENGINE" && "$ENGINE" -nt "$PT" ]]; then
    echo "[bake] $ENGINE is newer than best.pt — skipping engine build"
else
    echo "[bake] step 1/3 — initial til build cv $TAG (no engine yet) ..."
    (cd "$CV_DIR" && til build cv "$TAG")

    IMAGE=$(docker images --format '{{.Repository}}:{{.Tag}}' \
        | grep -m1 "cv:$TAG" || true)
    [[ -n "$IMAGE" ]] || { echo "[bake] no cv:$TAG image found" >&2; exit 1; }

    echo "[bake] step 2/3 — TensorRT export inside $IMAGE (3-8 min) ..."
    docker run --rm --gpus all \
        -v "$WEIGHTS_DIR:/workspace/src/weights" \
        --entrypoint python "$IMAGE" -c "
from ultralytics import YOLO
YOLO('/workspace/src/weights/best.pt').export(
    format='engine', half=True, dynamic=True, imgsz=1280, batch=16, opset=18,
)"
    [[ -f "$ENGINE" ]] || { echo "[bake] engine missing after export" >&2; exit 1; }
fi

echo "[bake] step 3/3 — rebuilding image with .engine baked in ..."
(cd "$CV_DIR" && til build cv "$TAG")

echo "[bake] done. engine: $(ls -lh "$ENGINE" | awk '{print $5}')"
echo "[bake] next: 'til test cv $TAG' (boots in ~10s now), then 'til submit cv $TAG'"
