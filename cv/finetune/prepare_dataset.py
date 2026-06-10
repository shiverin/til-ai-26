"""Convert the novice COCO dataset to YOLO format. Run once before training.

Writes finetune/data/{train,val}/{images,labels}/, dataset.yaml, and
val_coco.json (used by sweep_conf.py for COCO-format evaluation).

Usage:
    python prepare_dataset.py [--src /home/jupyter/novice/cv] [--val-frac 0.15]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 18 classes; list index = YOLO class id. COCO category ids are already 0..17.
NAMES = [
    "cargo aircraft", "commercial aircraft", "drone", "fighter jet",
    "fighter plane", "helicopter", "light aircraft", "missile",
    "truck", "car", "tank", "bus", "van",
    "cargo ship", "yacht", "cruise ship", "warship", "sailboat",
]

FINETUNE_DIR = Path(__file__).resolve().parent
OUT_DIR = FINETUNE_DIR / "data"


def coco_to_yolo_line(ann, img_w, img_h, cat_map):
    """Convert one COCO annotation to a YOLO label line
    '<cls> <cx> <cy> <w> <h>', all coords normalized to [0, 1]."""
    cls = cat_map[ann["category_id"]]
    left, top, w, h = ann["bbox"]
    cx = (left + w / 2.0) / img_w
    cy = (top + h / 2.0) / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def make_splits(image_ids, val_frac, seed=42):
    """Deterministically shuffle image ids into {'train': [...], 'val': [...]}."""
    ids = list(image_ids)
    np.random.default_rng(seed).shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return {"val": ids[:n_val], "train": ids[n_val:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/home/jupyter/novice/cv")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    with open(src / "annotations.json") as f:
        coco = json.load(f)

    cat_map = {c["id"]: i for i, c in enumerate(coco["categories"])}
    img_map = {img["id"]: img for img in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    splits = make_splits(img_map.keys(), args.val_frac, args.seed)
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}")

    for split, ids in splits.items():
        img_out = OUT_DIR / split / "images"
        lbl_out = OUT_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_id in tqdm(ids, desc=split):
            meta = img_map[img_id]
            fname = meta["file_name"]
            shutil.copy(src / "images" / fname, img_out / fname)
            lines = [
                coco_to_yolo_line(a, meta["width"], meta["height"], cat_map)
                for a in anns_by_img.get(img_id, [])
            ]
            (lbl_out / f"{Path(fname).stem}.txt").write_text("\n".join(lines))

    (OUT_DIR / "dataset.yaml").write_text(
        f"path: {OUT_DIR}\ntrain: train/images\nval: val/images\n\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n"
    )

    val_ids = set(splits["val"])
    val_coco = {
        "images": [img_map[i] for i in splits["val"]],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in val_ids],
        "categories": coco["categories"],
    }
    (OUT_DIR / "val_coco.json").write_text(json.dumps(val_coco))
    print(f"Dataset ready at {OUT_DIR} (dataset.yaml, val_coco.json written)")


if __name__ == "__main__":
    main()
