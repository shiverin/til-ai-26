"""Build sentence contexts for each pilot target word.

Two sources, per the plan:
  - real:      sentences pulled from the training transcripts (correct grammar,
               authentic style) — the primary source.
  - synthetic: radio/military-comms carrier sentences (this dataset's register)
               with the target word slotted in — to add volume and voice
               variety beyond the handful of real contexts.

Pure-CPU, stdlib only. Output: output/pilot_sentences.json.
"""

import json
import random
import re

FT = "/home/jupyter/til-ai-26/asr/finetune"
TARGETS = f"{FT}/output/pilot_targets.json"
TRAIN_MANIFEST = f"{FT}/output/train_manifest.json"
OUT = f"{FT}/output/pilot_sentences.json"

REAL_CAP = 6        # max real sentences kept per word
SYNTH_PER_WORD = 8  # synthetic carriers per word
SEED = 42

# Comms-register carriers. {W} works as a place, unit, ship, org, or faction —
# kept deliberately POS-flexible so one template set covers every target word.
TEMPLATES = [
    "Command, we have movement near {W}.",
    "Confirm the current status of {W}, over.",
    "All units, regroup at {W} and hold.",
    "Intelligence reports {W} has been compromised.",
    "Proceed toward {W} and await further orders.",
    "The briefing mentioned {W} more than once.",
    "Say again — {W} — how copy, over.",
    "We lost contact with {W} at zero three hundred.",
    "Reroute the convoy clear of {W} immediately.",
    "{W} is not responding to repeated hails.",
    "Recon places hostile activity around {W}.",
    "Logistics flagged a supply delay out of {W}.",
]


def sentences(text):
    """Split a transcript into rough sentences on terminal punctuation."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def main() -> None:
    rng = random.Random(SEED)
    words = json.load(open(TARGETS))["words"]
    rows = [json.loads(line) for line in open(TRAIN_MANIFEST)]

    # Index every training sentence once.
    all_sents = []
    for r in rows:
        all_sents.extend(sentences(r["text"]))

    out = []
    for w in words:
        needle = w.lower().rstrip("s")  # also catches plural / possessive
        real = [s for s in all_sents if needle in s.lower()]
        rng.shuffle(real)
        for s in real[:REAL_CAP]:
            out.append({"word": w, "text": s, "source": "real"})

        for tpl in rng.sample(TEMPLATES, SYNTH_PER_WORD):
            out.append({"word": w, "text": tpl.format(W=w), "source": "synth"})

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    n_real = sum(1 for e in out if e["source"] == "real")
    n_synth = sum(1 for e in out if e["source"] == "synth")
    print(f"{len(out)} sentences -> {OUT}  (real={n_real}, synth={n_synth})")
    per = {}
    for e in out:
        per[e["word"]] = per.get(e["word"], 0) + 1
    for w in words:
        print(f"  {w:20s} {per.get(w, 0):3d}")


if __name__ == "__main__":
    main()
