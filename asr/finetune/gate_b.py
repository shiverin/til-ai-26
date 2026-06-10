"""GATE B — does the pilot decoder fine-tune actually beat R1?

Compares R1 against the pilot candidate on two axes (GPU):
  1. Overall WER on the 200-clip real validation set — must NOT regress.
  2. Target-word recovery on the 222 held-out real hard-word clips — the
     direct test of whether synthetic exposure helped the fictional words.
     "Recovered" counts the joined form and any single-space split.

Ship the candidate only if val WER does not regress AND hard-word recovery
improves. Otherwise R1 stays live.

NEEDS GPU. Env: CANDIDATE=<path to candidate .nemo> (default output/pilot.nemo)
"""

import json
import os
import re

import nemo.collections.asr as nemo_asr

OUT_DIR = "/work/output"
VAL = f"{OUT_DIR}/val_manifest.json"
HARDWORD = f"{OUT_DIR}/hardword_eval_manifest.json"
R1 = "/models/parakeet_finetuned.nemo"
CANDIDATE = os.environ.get("CANDIDATE", f"{OUT_DIR}/pilot.nemo")


def norm(s: str) -> str:
    s = (s or "").lower().replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip()


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
    for h, r in zip(hyps, refs):
        rw = r.split()
        dist += edit_distance(h.split(), rw)
        length += len(rw)
    return dist / max(length, 1)


def split_forms(w: str) -> set[str]:
    return {w[:i] + " " + w[i:] for i in range(1, len(w))}


def recovered(hyp: str, word: str) -> bool:
    nw = norm(word)
    return nw in hyp.split() or f" {nw} " in f" {hyp} " \
        or any(sf in hyp for sf in split_forms(nw))


def transcribe(model, paths: list[str]) -> list[str]:
    outs = model.transcribe(paths, batch_size=16)
    return [norm(getattr(o, "text", o) or "") for o in outs]


def evaluate(model):
    val = [json.loads(line) for line in open(VAL)]
    val_wer = corpus_wer(
        transcribe(model, [r["audio_filepath"] for r in val]),
        [norm(r["text"]) for r in val])

    hard = [json.loads(line) for line in open(HARDWORD)]
    hyps = transcribe(model, [r["audio_filepath"] for r in hard])
    hit = total = 0
    for r, hyp in zip(hard, hyps):
        for w in r["target_words"]:
            total += 1
            hit += recovered(hyp, w)
    return val_wer, hit, total


def main() -> None:
    results = {}
    for name, path in [("R1 (current)", R1), ("pilot candidate", CANDIDATE)]:
        model = nemo_asr.models.ASRModel.restore_from(path).eval().cuda()
        val_wer, hit, total = evaluate(model)
        results[name] = (val_wer, hit, total)
        print(f"{name:18s}  val WER={val_wer:.4f}  acc={1 - val_wer:.4f}  "
              f"hard-word recovery={hit}/{total} "
              f"({100 * hit / max(total, 1):.1f}%)", flush=True)
        del model

    r1_wer, r1_hit, tot = results["R1 (current)"]
    c_wer, c_hit, _ = results["pilot candidate"]
    ship = c_wer <= r1_wer + 1e-4 and c_hit > r1_hit
    print(f"\nVERDICT: {'SHIP — candidate wins' if ship else 'DO NOT SHIP'}")
    print(f"  val WER: candidate {c_wer:.4f} vs R1 {r1_wer:.4f}")
    print(f"  hard-word recovery: candidate {c_hit} vs R1 {r1_hit} / {tot}")


if __name__ == "__main__":
    main()
