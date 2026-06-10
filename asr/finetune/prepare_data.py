"""Builds NeMo train/val manifests from novice/asr/asr.jsonl."""

import json
import os
import random

import soundfile as sf

DATA_DIR = "/data"
OUT_DIR = "/work/output"
VAL_SIZE = 200
SEED = 42


def main() -> None:
    with open(os.path.join(DATA_DIR, "asr.jsonl")) as f:
        rows = [json.loads(line) for line in f]

    entries = []
    for row in rows:
        path = os.path.join(DATA_DIR, row["audio"])
        duration = sf.info(path).duration
        entries.append({
            "audio_filepath": path,
            "duration": duration,
            "text": row["transcript"],
        })

    random.Random(SEED).shuffle(entries)
    val = entries[:VAL_SIZE]
    train = entries[VAL_SIZE:]

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, data in [("train_manifest.json", train),
                       ("val_manifest.json", val)]:
        with open(os.path.join(OUT_DIR, name), "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    print(f"train={len(train)} val={len(val)} total={len(entries)}")


if __name__ == "__main__":
    main()
