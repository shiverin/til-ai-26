"""Build the pilot training set: real + augmented synthetic data.

  - pilot_train_manifest.json   : real train clips (minus the hard-word clips)
                                  + augmented synthetic clips.
  - hardword_eval_manifest.json : real train clips that contain a target word,
                                  held out of training, tagged with which
                                  target words they contain — used at Gate B
                                  to measure target-word recognition on REAL
                                  audio.
  - val_manifest.json           : untouched, 100% real (the overall-WER gate).

CPU-only, stdlib only.
"""

import json

FT = "/work"
TARGETS = f"{FT}/output/pilot_targets.json"
TRAIN = f"{FT}/output/train_manifest.json"
SYNTH_AUG = f"{FT}/output/synth_aug_manifest.json"
PILOT_TRAIN = f"{FT}/output/pilot_train_manifest.json"
HARDWORD_EVAL = f"{FT}/output/hardword_eval_manifest.json"


def main() -> None:
    words = json.load(open(TARGETS))["words"]
    stems = [(w, w.lower().rstrip("s")) for w in words]

    train = [json.loads(line) for line in open(TRAIN)]
    synth = [json.loads(line) for line in open(SYNTH_AUG)]

    keep, heldout = [], []
    for r in train:
        low = r["text"].lower()
        hits = [w for w, s in stems if s in low]
        if hits:
            heldout.append({**r, "target_words": hits})
        else:
            keep.append(r)

    pilot_train = keep + [
        {"audio_filepath": s["audio_filepath"],
         "duration": s["duration"], "text": s["text"]}
        for s in synth
    ]

    for path, data in [(PILOT_TRAIN, pilot_train),
                       (HARDWORD_EVAL, heldout)]:
        with open(path, "w") as f:
            for e in data:
                f.write(json.dumps(e) + "\n")

    pct = 100 * len(synth) / max(len(pilot_train), 1)
    print(f"real kept        : {len(keep)}")
    print(f"synthetic added  : {len(synth)}")
    print(f"pilot train      : {len(pilot_train)}  (synthetic {pct:.0f}%)")
    print(f"hard-word eval   : {len(heldout)} real clips held out -> "
          f"{HARDWORD_EVAL}")
    covered = {w for e in heldout for w in e["target_words"]}
    missing = [w for w in words if w not in covered]
    if missing:
        print(f"WARNING: {len(missing)} target words have no real clip: "
              f"{missing}")


if __name__ == "__main__":
    main()
