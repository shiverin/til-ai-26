"""GATE A — verify Kokoro pronounced the fictional words acceptably.

Two CPU-only checks:
  1. Phonemes — run Kokoro's own G2P on each bare target word and print the
     phoneme string, so the intended pronunciation can be eyeballed.
  2. Recovery — transcribe the synthetic clips with the R1 model on CPU and
     report, per word, how often R1 recovers the word. "Recovered" counts both
     the joined form and any single-space split (e.g. "sky bridge" for
     "Skybridge") — those are correct pronunciations, just tokenization.

A word with low recovery AND sane phonemes is a genuine hard target (good).
A word with low recovery AND wrong phonemes means TTS mispronounced it (bad —
that word must be dropped, or if widespread, the whole approach is dead).
"""

import collections
import json
import re

import nemo.collections.asr as nemo_asr

FT = "/work"
TARGETS = f"{FT}/output/pilot_targets.json"
SYNTH_MANIFEST = f"{FT}/output/synth_manifest.json"
MODEL = "/models/parakeet_finetuned.nemo"
REPORT = f"{FT}/output/pronunciation_report.json"


def norm(s: str) -> str:
    s = (s or "").lower().replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip()


def split_forms(word: str) -> set[str]:
    """All single-space splits of a word: skybridge -> {'sky bridge', ...}."""
    return {word[:i] + " " + word[i:] for i in range(1, len(word))}


def phonemes(words: list[str]) -> dict:
    """Best-effort: Kokoro/misaki G2P for each bare word."""
    try:
        from misaki import en
        g2p = en.G2P()
        out = {}
        for w in words:
            res = g2p(w)
            out[w] = res[0] if isinstance(res, (tuple, list)) else str(res)
        return out
    except Exception as e:  # API drift — secondary signal, don't fail the gate
        print(f"(phoneme G2P unavailable: {e!r})")
        return {}


def main() -> None:
    words = json.load(open(TARGETS))["words"]

    phon = phonemes(words)
    if phon:
        print("=== Kokoro G2P phonemes (eyeball these) ===")
        for w in words:
            print(f"  {w:20s} {phon.get(w, '?')}")
        print()

    rows = [json.loads(line) for line in open(SYNTH_MANIFEST)]
    paths = [r["audio_filepath"] for r in rows]
    print(f"transcribing {len(paths)} synthetic clips with R1 on CPU...",
          flush=True)
    model = nemo_asr.models.ASRModel.restore_from(MODEL).eval()  # CPU
    outs = model.transcribe(paths, batch_size=8)
    hyps = [norm(getattr(o, "text", o) or "") for o in outs]

    per = collections.defaultdict(
        lambda: {"n": 0, "exact": 0, "split": 0, "miss": 0, "examples": []})
    for r, hyp in zip(rows, hyps):
        w = r["word"]
        nw = norm(w)
        d = per[w]
        d["n"] += 1
        if nw in hyp.split() or f" {nw} " in f" {hyp} ":
            d["exact"] += 1
        elif any(sf in hyp for sf in split_forms(nw)):
            d["split"] += 1
        else:
            d["miss"] += 1
            if len(d["examples"]) < 4:
                d["examples"].append({"ref": norm(r["text"]), "hyp": hyp})

    print("=== per-word recovery (R1 on synthetic audio) ===")
    print(f"{'word':18s}{'n':>4s}{'exact':>7s}{'split':>7s}"
          f"{'miss':>6s}{'recov%':>8s}")
    report = {}
    for w in words:
        d = per[w]
        rec = d["exact"] + d["split"]
        pct = 100 * rec / max(d["n"], 1)
        report[w] = {"phonemes": phon.get(w), **d, "recovered_pct": pct}
        print(f"{w:18s}{d['n']:4d}{d['exact']:7d}{d['split']:7d}"
              f"{d['miss']:6d}{pct:7.0f}%")

    tot_n = sum(d["n"] for d in per.values())
    tot_rec = sum(d["exact"] + d["split"] for d in per.values())
    print(f"\noverall recovered: {tot_rec}/{tot_n} "
          f"({100 * tot_rec / max(tot_n, 1):.0f}%)")

    print("\n=== sample misses (word: ref -> hyp) ===")
    for w in words:
        for ex in per[w]["examples"][:2]:
            print(f"  [{w}] {ex['ref']}\n      -> {ex['hyp']}")

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {REPORT}")


if __name__ == "__main__":
    main()
