#!/bin/sh
# Round 3 auto-pipeline: waits for TTS, then augment -> manifest -> train ->
# convert -> Gate B. Runs inside a docker-supervised orchestrator container so
# it survives shell teardown / idle-shutdown.
#
# Submit is NOT auto — happens on the host once Gate B's verdict is in.
set -e

FT=/home/jupyter/til-ai-26/asr/finetune
MODELS=/home/jupyter/til-ai-26/asr/models
DATA=/home/jupyter/novice/asr

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# A. wait for TTS to finish
log "waiting for asr-tts-r3 ..."
docker wait asr-tts-r3 >/dev/null
TTS_EXIT=$(docker inspect asr-tts-r3 --format='{{.State.ExitCode}}')
log "TTS done, exit=$TTS_EXIT, clips=$(ls "$FT/synth/clips" 2>/dev/null | wc -l)"
[ "$TTS_EXIT" = "0" ] || { log "TTS failed"; exit 1; }

# B. augment (skips existing 2270 augmentations)
log "augment ..."
docker rm -f asr-r3-aug 2>/dev/null || true
docker run --rm --name asr-r3-aug -e CUDA_VISIBLE_DEVICES="" \
  -v "$FT":/work asr-tts python /work/augment.py >/dev/null

# C. build pilot manifest
log "build manifest ..."
docker rm -f asr-r3-bld 2>/dev/null || true
docker run --rm --name asr-r3-bld -e CUDA_VISIBLE_DEVICES="" \
  -v "$FT":/work asr-tts python /work/build_pilot_manifest.py

# D. GPU pipeline (train from current 0.984 model + convert + Gate B)
log "GPU pipeline (BATCH_SIZE=4) ..."
docker rm -f asr-r3-pipeline 2>/dev/null || true
docker run --name asr-r3-pipeline --gpus all --shm-size=8g \
  -e BATCH_SIZE=4 \
  -v "$DATA":/data:ro \
  -v "$FT":/work \
  -v "$MODELS":/models:ro \
  asr-tts bash -c '
    set -e
    rm -f /work/output/pilot_decoder.ckpt
    echo "===== 1/3 train ====="
    python /work/train_decoder.py
    echo "===== 2/3 convert ====="
    CUDA_VISIBLE_DEVICES="" \
      CKPT=/work/output/pilot_decoder.ckpt \
      OUT=/work/output/pilot.nemo \
      ARCH_MODEL=/models/parakeet_finetuned.nemo \
      python /work/ckpt_to_nemo.py
    echo "===== 3/3 Gate B ====="
    python /work/gate_b.py
    echo "===== PIPELINE DONE ====="
  '

log "all stages complete. verdict:"
docker logs asr-r3-pipeline 2>&1 | grep -E 'VERDICT|R1 \(current\)|pilot candidate' | tail -3
