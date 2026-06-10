#!/bin/sh
# Auto-pipeline: waits for the TTS container, then runs augment -> manifest
# -> train -> convert -> Gate B. Driven from a docker-in-docker orchestrator
# so the Docker daemon supervises it (survives shell teardown / idle-shutdown).
#
# Submit is NOT automated — happens on the host once Gate B's verdict is in.
set -e

FT=/home/jupyter/til-ai-26/asr/finetune
MODELS=/home/jupyter/til-ai-26/asr/models
DATA=/home/jupyter/novice/asr

log()  { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- A. wait for TTS to finish ----
log "waiting for asr-tts-gen ..."
docker wait asr-tts-gen >/dev/null
TTS_EXIT=$(docker inspect asr-tts-gen --format='{{.State.ExitCode}}')
log "TTS done, exit=$TTS_EXIT, clips=$(ls "$FT/synth/clips" 2>/dev/null | wc -l)"
[ "$TTS_EXIT" = "0" ] || { log "TTS failed"; exit 1; }

# ---- B. augment ----
log "augment ..."
docker rm -f asr-scaleup-aug 2>/dev/null || true
docker run --rm --name asr-scaleup-aug -e CUDA_VISIBLE_DEVICES="" \
  -v "$FT":/work asr-tts python /work/augment.py >/dev/null

# ---- C. build pilot manifest ----
log "build manifest ..."
docker rm -f asr-scaleup-bld 2>/dev/null || true
docker run --rm --name asr-scaleup-bld -e CUDA_VISIBLE_DEVICES="" \
  -v "$FT":/work asr-tts python /work/build_pilot_manifest.py

# ---- D. GPU pipeline (train + convert + gate B) ----
log "GPU pipeline ..."
docker rm -f asr-scaleup-pipeline 2>/dev/null || true
docker run --name asr-scaleup-pipeline --gpus all --shm-size=8g \
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
    echo "===== 3/3 gate B ====="
    python /work/gate_b.py
    echo "===== PIPELINE DONE ====="
  '

log "all stages complete — verdict in: docker logs asr-scaleup-pipeline | grep VERDICT"
