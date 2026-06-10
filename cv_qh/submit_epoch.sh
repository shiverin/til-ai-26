#!/usr/bin/env bash
# Snapshot the current best_ema checkpoint, bake it into the submission image,
# and submit to TIL. Lets you submit per-epoch even while training keeps writing
# new bests into the same file.
#
# Usage:
#   ./submit_epoch.sh <tag-suffix>
# Example:
#   ./submit_epoch.sh ep3        # builds rfdetr-ep3, submits rfdetr-ep3
#
# At the end the script sleeps SUBMIT_COOLDOWN seconds (default 600 = 10 min)
# so chained invocations don't overlap — TIL only grades one submission at a
# time per team. Chain like:
#   ./submit_epoch.sh ep1 && ./submit_epoch.sh ep2 && ./submit_epoch.sh ep3
# and each will wait its cooldown before the next starts.
# Override with: SUBMIT_COOLDOWN=300 ./submit_epoch.sh ep1
set -euo pipefail

COOLDOWN="${SUBMIT_COOLDOWN:-600}"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <tag-suffix>   (e.g. ep3)" >&2
    exit 2
fi
SUFFIX="$1"
TAG="rfdetr-${SUFFIX}"

CV_QH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVE="$CV_QH_DIR/weights/base/checkpoint_best_ema.pth"
SNAP="$CV_QH_DIR/weights/base/snapshot_${SUFFIX}.pth"

if [[ ! -f "$LIVE" ]]; then
    echo "[submit] no checkpoint_best_ema.pth at $LIVE — has any epoch saved yet?" >&2
    exit 1
fi

echo "[submit] step 1/4 — snapshotting $LIVE → $SNAP"
cp "$LIVE" "$SNAP"

echo "[submit] step 2/4 — promoting snapshot to src/weights/best.pt"
"$CV_QH_DIR/promote_checkpoint.sh" "$SNAP"

echo "[submit] step 3/4 — til build cv $TAG"
(cd "$CV_QH_DIR" && til build cv "$TAG")

echo "[submit] step 4/4 — til submit cv $TAG"
(cd "$CV_QH_DIR" && til submit cv "$TAG")

echo "[submit] done. Snapshot kept at $SNAP for re-submit if needed."
echo "[submit] cooldown ${COOLDOWN}s before next submission window opens..."
sleep "$COOLDOWN"
echo "[submit] cooldown done."
