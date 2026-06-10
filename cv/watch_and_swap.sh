#!/usr/bin/env bash
# Wait until the in-flight submit_no_retrain.sh finishes submitting variant 2
# (bilateral), kill it before it fires variant 3 (TTA+bilateral, expected DOA),
# then submit the replacement NL-means variant after the 11-min cooldown.
set -euo pipefail

LOG=/home/jupyter/til-ai-26/cv/submit_no_retrain.log
SCRIPT_PID=267959
COOLDOWN=660

echo "[watch] $(date +%H:%M:%S) waiting for variant 2 (bilateral) to enter cooldown..."
# Variant 1 already produced one "sleeping" line. Wait for the second.
until [ "$(grep -c 'sleeping 660s' "$LOG" 2>/dev/null || echo 0)" -ge 2 ]; do
    sleep 20
done

echo "[watch] $(date +%H:%M:%S) variant 2 submitted. Killing script PID=$SCRIPT_PID"
kill "$SCRIPT_PID" 2>/dev/null || echo "[watch] script already gone"
# Also kill any child sleep process the script may have spawned
pkill -P "$SCRIPT_PID" 2>/dev/null || true

echo "[watch] $(date +%H:%M:%S) cooldown ${COOLDOWN}s before NL-means submit..."
sleep "$COOLDOWN"

echo "[watch] $(date +%H:%M:%S) submitting v8m-ep1-nlmeans-1536"
cd /home/jupyter/til-ai-26/cv
til submit cv v8m-ep1-nlmeans-1536

echo "[watch] $(date +%H:%M:%S) NL-means submission complete"
