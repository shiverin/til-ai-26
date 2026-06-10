"""Verify FP16-autocast inference vs FP32 — WER must hold, want a speedup.

Pure model.half() destroys WER on the Conformer encoder (verified: WER 1.7).
Autocast (mixed precision) keeps weights/sensitive ops in FP32 and only casts
matmuls to FP16, matching the 16-mixed training recipe. This checks that the
autocast path keeps WER while still hitting the FP16 tensor cores.

Transcribes the 200-clip val set in both modes after a warmup pass so timings
are not skewed by CUDA-graph capture / cuDNN autotune.
"""

import json
import re
import time

import jiwer
import nemo.collections.asr as nemo_asr
import torch

MODEL = "/models/parakeet_finetuned.nemo"
MANIFEST = "/work/output/val_manifest.json"
BATCH = 16


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())


def transcribe(model, paths, autocast: bool):
    if autocast:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outs = model.transcribe(paths, batch_size=BATCH)
    else:
        outs = model.transcribe(paths, batch_size=BATCH)
    return [norm(getattr(o, "text", o)) for o in outs]


def main() -> None:
    rows = [json.loads(line) for line in open(MANIFEST)]
    paths = [r["audio_filepath"] for r in rows]
    refs = [norm(r["text"]) for r in rows]

    model = nemo_asr.models.ASRModel.restore_from(MODEL).eval().cuda()

    transcribe(model, paths[:8], autocast=False)  # warmup
    t0 = time.time()
    hyps32 = transcribe(model, paths, autocast=False)
    dt32 = time.time() - t0
    wer32 = jiwer.wer(refs, hyps32)

    transcribe(model, paths[:8], autocast=True)  # warmup the fp16 graph
    t0 = time.time()
    hyps16 = transcribe(model, paths, autocast=True)
    dt16 = time.time() - t0
    wer16 = jiwer.wer(refs, hyps16)

    print("=" * 50)
    print(f"FP32       WER={wer32:.4f}  acc={1 - wer32:.4f}  time={dt32:.1f}s")
    print(f"FP16-amp   WER={wer16:.4f}  acc={1 - wer16:.4f}  time={dt16:.1f}s")
    print(f"WER delta (FP16amp - FP32) = {wer16 - wer32:+.4f}")
    print(f"speedup = {dt32 / dt16:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()
