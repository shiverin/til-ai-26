"""Error analysis of the current fine-tuned model on the validation set.

Transcribes the val set with the live model, then breaks the errors down:
overall WER, the worst samples, the most common substitution / deletion /
insertion patterns, and a digit-vs-words (number formatting) check.
"""

import collections
import difflib
import json
import re

import nemo.collections.asr as nemo_asr

OUT_DIR = "/work/output"
VAL_MANIFEST = f"{OUT_DIR}/val_manifest.json"
MODEL = "/models/parakeet_finetuned.nemo"


def normalize(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


def main() -> None:
    rows = [json.loads(line) for line in open(VAL_MANIFEST)]
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize(r["text"]) for r in rows]

    model = nemo_asr.models.ASRModel.restore_from(MODEL).eval().cuda()
    outputs = model.transcribe(paths, batch_size=16)
    hyps = [normalize(getattr(o, "text", o) or "") for o in outputs]

    subs = collections.Counter()
    dels = collections.Counter()
    inss = collections.Counter()
    per_sample = []
    total_dist = total_len = 0
    digit_mismatch = 0

    for idx, (ref, hyp) in enumerate(zip(refs, hyps)):
        rw, hw = ref.split(), hyp.split()
        dist = edit_distance(rw, hw)
        total_dist += dist
        total_len += len(rw)
        per_sample.append((dist / max(len(rw), 1), idx, ref, hyp))
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, rw, hw).get_opcodes():
            if tag == "replace":
                subs[(" ".join(rw[i1:i2]), " ".join(hw[j1:j2]))] += 1
            elif tag == "delete":
                for w in rw[i1:i2]:
                    dels[w] += 1
            elif tag == "insert":
                for w in hw[j1:j2]:
                    inss[w] += 1
        # Number-formatting check: digits on exactly one side.
        if has_digit(ref) != has_digit(hyp):
            digit_mismatch += 1

    print(f"=== overall: WER={total_dist / max(total_len, 1):.4f} "
          f"over {len(rows)} samples ===\n")

    print("=== 12 worst samples ===")
    for wer, idx, ref, hyp in sorted(per_sample, reverse=True)[:12]:
        print(f"[{idx}] WER={wer:.2f}")
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}\n")

    print("=== top 25 substitutions (ref -> hyp : count) ===")
    for (r, h), c in subs.most_common(25):
        print(f"  {c:3d}  {r!r} -> {h!r}")

    print("\n=== top 15 deletions (missed words) ===")
    for w, c in dels.most_common(15):
        print(f"  {c:3d}  {w!r}")

    print("\n=== top 15 insertions (extra words) ===")
    for w, c in inss.most_common(15):
        print(f"  {c:3d}  {w!r}")

    print(f"\n=== digit/words mismatch: {digit_mismatch}/{len(rows)} "
          f"samples have digits on only one side ===")


if __name__ == "__main__":
    main()
