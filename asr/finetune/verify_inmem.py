"""Verify in-memory audio decoding gives identical WER vs temp-file paths.

Runs on CPU (CUDA_VISIBLE_DEVICES="") over a small subset so it does not
contend for the shared GPU. Greedy decoding is deterministic, so a CPU match
guarantees a GPU match. The point is to confirm two things:
  1. NeMo transcribe() accepts a list of numpy arrays (not just file paths).
  2. Decoding the WAV bytes in memory yields the same transcripts.
"""

import io
import json
import re

import jiwer
import nemo.collections.asr as nemo_asr
import soundfile as sf

MODEL = "/models/parakeet_finetuned.nemo"
MANIFEST = "/work/output/val_manifest.json"
BATCH = 8
SUBSET = 24


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())


def decode(path: str):
    """Decode WAV bytes in memory, mono float32 — mirrors the manager path."""
    raw = open(path, "rb").read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def texts(outs):
    return [norm(getattr(o, "text", o)) for o in outs]


def main() -> None:
    rows = [json.loads(line) for line in open(MANIFEST)][:SUBSET]
    paths = [r["audio_filepath"] for r in rows]
    refs = [norm(r["text"]) for r in rows]

    model = nemo_asr.models.ASRModel.restore_from(MODEL).eval()  # CPU

    out_path = model.transcribe(paths, batch_size=BATCH)
    wer_path = jiwer.wer(refs, texts(out_path))

    arrays, srs = [], set()
    for p in paths:
        audio, sr = decode(p)
        srs.add(sr)
        arrays.append(audio)
    print("sample rates seen:", srs)

    out_mem = model.transcribe(arrays, batch_size=BATCH)
    wer_mem = jiwer.wer(refs, texts(out_mem))

    mismatches = sum(
        1 for a, b in zip(texts(out_path), texts(out_mem)) if a != b
    )
    print("=" * 50)
    print(f"PATHS    WER={wer_path:.4f}  ({SUBSET} clips)")
    print(f"IN-MEM   WER={wer_mem:.4f}  ({SUBSET} clips)")
    print(f"transcript mismatches: {mismatches}/{SUBSET}")
    print("=" * 50)


if __name__ == "__main__":
    main()
