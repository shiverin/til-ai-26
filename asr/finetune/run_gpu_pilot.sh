#!/bin/bash
# Pilot Gate-B run — NEEDS GPU. Run this when the card is free.
#   bash /home/jupyter/til-ai-26/asr/finetune/run_gpu_pilot.sh
# Optional: BATCH_SIZE=4 bash run_gpu_pilot.sh   (lower if GPU memory is tight)
set -e
FT=/home/jupyter/til-ai-26/asr/finetune
MODELS=/home/jupyter/til-ai-26/asr/models
DATA=/home/jupyter/novice/asr
BATCH_SIZE=${BATCH_SIZE:-8}

echo "===== 1/3  decoder fine-tune (encoder frozen) ====="
docker rm -f asr-pilot-train 2>/dev/null || true
docker run --rm --name asr-pilot-train --gpus all --shm-size=8g \
  -e BATCH_SIZE="$BATCH_SIZE" \
  -v "$DATA":/data:ro -v "$FT":/work -v "$MODELS":/models:ro \
  asr-tts python /work/train_decoder.py

echo "===== 2/3  convert checkpoint -> .nemo (CPU) ====="
docker run --rm \
  -e CUDA_VISIBLE_DEVICES="" \
  -e CKPT=/work/output/pilot_decoder.ckpt \
  -e OUT=/work/output/pilot.nemo \
  -e ARCH_MODEL=/models/parakeet_finetuned.nemo \
  -v "$FT":/work -v "$MODELS":/models:ro \
  asr-tts python /work/ckpt_to_nemo.py

echo "===== 3/3  GATE B — R1 vs pilot candidate ====="
docker run --rm --gpus all \
  -v "$DATA":/data:ro -v "$FT":/work -v "$MODELS":/models:ro \
  asr-tts python /work/gate_b.py

echo
echo "Done. If Gate B says SHIP: copy output/pilot.nemo over"
echo "asr/models/parakeet_finetuned.nemo, then til build + til submit."
echo "If DO NOT SHIP: R1 stays live, nothing changes."
