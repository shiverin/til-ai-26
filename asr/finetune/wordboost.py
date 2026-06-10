"""Validates a vocabulary-based word-boosting post-corrector on the val set.

Builds a "world vocabulary" from the training transcripts, transcribes the
val set with the current model, then compares WER with and without the
corrector. The corrector replaces an out-of-vocabulary output word with a
frequent world word that is one edit away (e.g. 'wex' -> 'vex').
"""

import collections
import json
import re

import nemo.collections.asr as nemo_asr

OUT_DIR = "/work/output"
TRAIN_MANIFEST = f"{OUT_DIR}/train_manifest.json"
VAL_MANIFEST = f"{OUT_DIR}/val_manifest.json"
MODEL = "/models/parakeet_finetuned.nemo"

KNOWN_MIN = 2     # an output word with >= this count is left alone
TARGET_MIN = 5    # only correct toward world words at least this frequent
_ALPHA = "abcdefghijklmnopqrstuvwxyz'"


def normalize(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def edits1(word: str) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    out = set()
    for a, b in splits:
        if b:
            out.add(a + b[1:])                                  # delete
        if len(b) > 1:
            out.add(a + b[1] + b[0] + b[2:])                    # transpose
        for c in _ALPHA:
            if b:
                out.add(a + c + b[1:])                          # replace
            out.add(a + c + b)                                  # insert
    return out


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def corpus_wer(hyps: list[str], refs: list[str]) -> float:
    dist = length = 0
    for hyp, ref in zip(hyps, refs):
        rw = ref.split()
        dist += edit_distance(hyp.split(), rw)
        length += len(rw)
    return dist / max(length, 1)


def main() -> None:
    vocab = collections.Counter()
    for line in open(TRAIN_MANIFEST):
        vocab.update(normalize(json.loads(line)["text"]).split())

    rows = [json.loads(line) for line in open(VAL_MANIFEST)]
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize(r["text"]) for r in rows]

    model = nemo_asr.models.ASRModel.restore_from(MODEL).eval().cuda()
    outputs = model.transcribe(paths, batch_size=16)
    raw = [normalize(getattr(o, "text", o) or "") for o in outputs]

    corrections = []

    def correct(text: str) -> str:
        words = text.split()
        for i, w in enumerate(words):
            if vocab.get(w, 0) >= KNOWN_MIN:
                continue
            cands = [(vocab[e], e) for e in edits1(w)
                     if vocab.get(e, 0) >= TARGET_MIN]
            if cands:
                best = max(cands)[1]
                if best != w:
                    words[i] = best
                    corrections.append((w, best))
        return " ".join(words)

    corrected = [correct(t) for t in raw]

    raw_wer = corpus_wer(raw, refs)
    cor_wer = corpus_wer(corrected, refs)
    print(f"raw:        WER={raw_wer:.4f}  acc={max(0, 1 - raw_wer):.4f}")
    print(f"corrected:  WER={cor_wer:.4f}  acc={max(0, 1 - cor_wer):.4f}")
    print(f"corrections applied: {len(corrections)}")
    for w, b in collections.Counter(corrections).most_common(30):
        print(f"  {w!r} -> {b!r}")
    print("VERDICT:", "boosting helps" if cor_wer < raw_wer
          else "no gain — do not ship")


if __name__ == "__main__":
    main()
