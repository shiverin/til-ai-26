"""Generate synthetic TTS audio for the pilot sentences with Kokoro-82M.

CPU-only. Each sentence is rendered with several voices (mixed accent and
gender) so the decoder fine-tune does not overfit to one speaker. Output is
16 kHz mono WAV (resampled from Kokoro's native 24 kHz) plus a NeMo-style
JSONL manifest.
"""

import json
import os

import librosa
import numpy as np
import soundfile as sf
from kokoro import KPipeline

FT = "/work"
SENTENCES = f"{FT}/output/pilot_sentences.json"
CLIP_DIR = f"{FT}/synth/clips"
MANIFEST = f"{FT}/output/synth_manifest.json"

# (voice, lang_code) — mixed accent and gender for speaker variety.
VOICES = [
    ("af_heart", "a"),       # American female
    ("am_michael", "a"),     # American male
    ("bf_emma", "b"),        # British female
    ("bm_george", "b"),      # British male
    ("af_bella", "a"),       # second American female (different timbre)
]
KOKORO_SR = 24000
TARGET_SR = 16000


def main() -> None:
    os.makedirs(CLIP_DIR, exist_ok=True)
    sentences = json.load(open(SENTENCES))
    pipelines = {lc: KPipeline(lang_code=lc) for _, lc in VOICES}

    manifest = []
    skipped = 0
    for idx, entry in enumerate(sentences):
        text = entry["text"]
        for voice, lc in VOICES:
            path = f"{CLIP_DIR}/{idx:04d}_{voice}.wav"
            # Skip clips that already exist (Round 2 work that we don't want
            # to redo). make_sentences is deterministic, so the same (idx,
            # voice) corresponds to the same sentence text.
            if os.path.exists(path):
                info = sf.info(path)
                manifest.append({
                    "audio_filepath": path,
                    "duration": info.frames / info.samplerate,
                    "text": text,
                    "word": entry["word"],
                    "voice": voice,
                    "source": entry["source"],
                })
                skipped += 1
                continue
            chunks = []
            for _, _, audio in pipelines[lc](text, voice=voice):
                a = audio.cpu().numpy() if hasattr(audio, "cpu") \
                    else np.asarray(audio)
                chunks.append(a.astype(np.float32))
            if not chunks:
                continue
            wav = librosa.resample(
                np.concatenate(chunks), orig_sr=KOKORO_SR, target_sr=TARGET_SR
            )
            sf.write(path, wav, TARGET_SR)
            manifest.append({
                "audio_filepath": path,
                "duration": len(wav) / TARGET_SR,
                "text": text,
                "word": entry["word"],
                "voice": voice,
                "source": entry["source"],
            })
        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/{len(sentences)} sentences "
                  f"({skipped} skipped)", flush=True)

    with open(MANIFEST, "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"DONE: {len(manifest)} clips -> {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
