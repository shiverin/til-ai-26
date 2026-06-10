"""Extract candidate fictional vocabulary from the training transcripts.

Fictional / world words are proper-noun-like: they appear Capitalized in the
*middle* of a sentence (not merely at a sentence start, where any word is
capitalized). We rank candidates by rarity — the words the ASR model has had
the least exposure to are the ones it mishears most.

Pure-CPU, stdlib only. Output: a ranked JSON list at output/fictional_words.json.
"""

import collections
import json

FT = "/home/jupyter/til-ai-26/asr/finetune"
TRAIN_MANIFEST = f"{FT}/output/train_manifest.json"
OUT = f"{FT}/output/fictional_words.json"

# Words legitimately capitalized mid-sentence in English — not fictional vocab.
_STOPCAPS = {"I", "Im", "Ive", "Ill", "Id", "A"}


def sentences(text):
    """Rough sentence split on terminal punctuation."""
    out, cur = [], []
    for tok in text.split():
        cur.append(tok)
        if tok[-1:] in ".!?":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def clean(tok):
    """Strip surrounding punctuation, keep internal apostrophes/hyphens."""
    return tok.strip(".,!?;:\"'()[]")


def main() -> None:
    rows = [json.loads(line) for line in open(TRAIN_MANIFEST)]

    total = collections.Counter()   # lowercased word -> total occurrences
    capmid = collections.Counter()  # lowercased word -> mid-sentence-cap count
    raw_form = {}                   # lowercased word -> a representative form

    for r in rows:
        for words in sentences(r["text"]):
            for i, tok in enumerate(words):
                w = clean(tok)
                if not w or not w[0].isalpha():
                    continue
                lw = w.lower()
                total[lw] += 1
                if i > 0 and w[0].isupper() and w not in _STOPCAPS:
                    capmid[lw] += 1
                    raw_form.setdefault(lw, w)

    # A fictional candidate: capitalized mid-sentence at least twice, and
    # capitalized in the majority of its occurrences (so it is a name, not an
    # ordinary word that happened to start a few sentences).
    candidates = []
    for lw, capc in capmid.items():
        tot = total[lw]
        if capc >= 2 and capc / tot >= 0.5:
            candidates.append({
                "word": raw_form[lw],
                "total": tot,
                "cap_mid": capc,
            })

    candidates.sort(key=lambda c: (c["total"], c["word"]))  # rarest first

    with open(OUT, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"{len(candidates)} fictional-word candidates -> {OUT}\n")
    print(f"{'word':22s}{'total':>7s}{'cap_mid':>9s}")
    for c in candidates:
        print(f"{c['word']:22s}{c['total']:7d}{c['cap_mid']:9d}")


if __name__ == "__main__":
    main()
