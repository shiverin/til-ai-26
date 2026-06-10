#!/usr/bin/env bash
# Watch finetune/runs/yolo11x_r1_light/weights/ for new epoch checkpoints.
# For each new epochN.pt: copy to src/weights/best.pt, build a tagged docker
# image, submit. 660s cooldown between submits.
set -euo pipefail

WEIGHTS_DIR=/home/jupyter/til-ai-26/cv/finetune/runs/yolo11x_r1_light/weights
STAGE_PT=/home/jupyter/til-ai-26/cv/src/weights/best.pt
SCRIPT_LOG=/home/jupyter/til-ai-26/cv/watch_yolo11x_epochs.log
COOLDOWN=660

submitted=()

is_submitted() {
    local needle="$1"
    for s in "${submitted[@]+"${submitted[@]}"}"; do
        [ "$s" = "$needle" ] && return 0
    done
    return 1
}

echo "[watch] $(date +%H:%M:%S) starting epoch watcher" | tee -a "$SCRIPT_LOG"
while true; do
    if [ -d "$WEIGHTS_DIR" ]; then
        for f in "$WEIGHTS_DIR"/epoch*.pt; do
            [ -e "$f" ] || continue
            name=$(basename "$f" .pt)              # epoch0, epoch1, ...
            ep_idx=${name#epoch}
            tag="yolo11x-r1-ep$((ep_idx + 1))-1536"
            if is_submitted "$tag"; then
                continue
            fi
            # Wait 30s after the file appears to make sure write is finalized.
            sleep 30
            echo "[watch] $(date +%H:%M:%S) new checkpoint $name; tag=$tag" | tee -a "$SCRIPT_LOG"
            cp -f "$f" "$STAGE_PT"
            cd /home/jupyter/til-ai-26/cv
            echo "[watch] $(date +%H:%M:%S) building $tag..." | tee -a "$SCRIPT_LOG"
            til build cv "$tag" >> "$SCRIPT_LOG" 2>&1
            echo "[watch] $(date +%H:%M:%S) submitting $tag..." | tee -a "$SCRIPT_LOG"
            til submit cv "$tag" >> "$SCRIPT_LOG" 2>&1
            submitted+=("$tag")
            echo "[watch] $(date +%H:%M:%S) $tag submitted; cooldown ${COOLDOWN}s" | tee -a "$SCRIPT_LOG"
            sleep "$COOLDOWN"
        done
    fi
    sleep 60
done
