"""Validation gate: current live model vs the new candidate fine-tune.

Transcribes the held-out validation set with both the currently-deployed
model (Round 1, mounted at /models) and the new candidate (Round 2, in the
output dir), computes competition-style WER, and reports which wins. The
candidate ships only if it beats the current model.
"""

import json
import re

import nemo.collections.asr as nemo_asr

OUT_DIR = "/work/output"
VAL_MANIFEST = f"{OUT_DIR}/val_manifest.json"
CURRENT = "/models/parakeet_finetuned.nemo"          # Round 1 — currently live
CANDIDATE = f"{OUT_DIR}/parakeet_finetuned.nemo"     # Round 2 — new


def normalize(text: str) -> str:
    """Competition-style normalization: lowercase, drop punctuation."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (wa != wb),
            ))
        prev = cur
    return prev[-1]


def corpus_wer(hyps: list[str], refs: list[str]) -> float:
    """Aggregate word error rate over the corpus."""
    total_dist = total_len = 0
    for hyp, ref in zip(hyps, refs):
        ref_words = ref.split()
        total_dist += _edit_distance(hyp.split(), ref_words)
        total_len += len(ref_words)
    return total_dist / max(total_len, 1)


def transcribe(model, paths: list[str]) -> list[str]:
    outputs = model.transcribe(paths, batch_size=16)
    return [normalize(getattr(o, "text", o) or "") for o in outputs]


def main() -> None:
    rows = [json.loads(line) for line in open(VAL_MANIFEST)]
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize(r["text"]) for r in rows]

    cur = nemo_asr.models.ASRModel.restore_from(CURRENT).eval().cuda()
    cur_wer = corpus_wer(transcribe(cur, paths), refs)
    del cur

    cand = nemo_asr.models.ASRModel.restore_from(CANDIDATE).eval().cuda()
    cand_wer = corpus_wer(transcribe(cand, paths), refs)

    print(f"current (R1):   WER={cur_wer:.4f}  acc={max(0, 1 - cur_wer):.4f}")
    print(f"candidate (R2): WER={cand_wer:.4f}  acc={max(0, 1 - cand_wer):.4f}")
    print("WINNER:", "candidate" if cand_wer < cur_wer else "current")


if __name__ == "__main__":
    main()
