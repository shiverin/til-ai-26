"""Add domain noise to the clean TTS clips so they resemble the real audio.

Real competition audio is noisy; clean TTS is unnaturally pristine and the
encoder treats it as out-of-domain. This applies SNR-based additive noise,
mild speed/tempo perturbation, and random gain. Runs after Gate A — the
pronunciation check uses the *clean* clips; the fine-tune uses these noised
ones. CPU-only.

Input:  output/synth_manifest.json  + synth/clips/*.wav
Output: output/synth_aug_manifest.json + synth/clips_aug/*.wav
"""

import json
import os

import numpy as np
import soundfile as sf
from audiomentations import AddGaussianSNR, Compose, Gain, TimeStretch

FT = "/work"
SYNTH_MANIFEST = f"{FT}/output/synth_manifest.json"
AUG_DIR = f"{FT}/synth/clips_aug"
OUT_MANIFEST = f"{FT}/output/synth_aug_manifest.json"
SR = 16000
SEED = 42

augment = Compose([
    AddGaussianSNR(min_snr_db=5.0, max_snr_db=25.0, p=0.9),
    TimeStretch(min_rate=0.9, max_rate=1.1,
                leave_length_unchanged=False, p=0.5),
    Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.3),
])


def main() -> None:
    os.makedirs(AUG_DIR, exist_ok=True)
    np.random.seed(SEED)
    rows = [json.loads(line) for line in open(SYNTH_MANIFEST)]

    out = []
    skipped = 0
    for i, r in enumerate(rows):
        name = os.path.basename(r["audio_filepath"])
        path = f"{AUG_DIR}/{name}"
        if os.path.exists(path):
            # Already augmented in a previous round — keep that augmentation.
            info = sf.info(path)
            out.append({
                "audio_filepath": path,
                "duration": info.frames / info.samplerate,
                "text": r["text"],
                "word": r["word"],
                "source": r["source"],
            })
            skipped += 1
            continue
        wav, _ = sf.read(r["audio_filepath"], dtype="float32")
        noised = augment(samples=wav, sample_rate=SR)
        sf.write(path, noised, SR)
        out.append({
            "audio_filepath": path,
            "duration": len(noised) / SR,
            "text": r["text"],
            "word": r["word"],
            "source": r["source"],
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} augmented "
                  f"({skipped} skipped)", flush=True)

    with open(OUT_MANIFEST, "w") as f:
        for m in out:
            f.write(json.dumps(m) + "\n")
    print(f"DONE: {len(out)} augmented clips -> {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
