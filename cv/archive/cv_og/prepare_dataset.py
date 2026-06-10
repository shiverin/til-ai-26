"""Convert novice COCO dataset -> YOLO format. Run once before training."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC_DIR = Path("/home/jupyter/novice/cv")
OUT_DIR = Path("/home/jupyter/til-ai-26/cv/cv_og/cv_yolo")
VAL_FRAC = 0.15

NAMES = [
    "cargo aircraft", "commercial aircraft", "drone", "fighter jet",
    "fighter plane", "helicopter", "light aircraft", "missile",
    "truck", "car", "tank", "bus", "van",
    "cargo ship", "yacht", "cruise ship", "warship", "sailboat",
]


def main():
    ann_path = SRC_DIR / "annotations.json"
    img_dir = SRC_DIR / "images"
    with open(ann_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: i for i, c in enumerate(coco["categories"])}
    img_map = {img["id"]: img for img in coco["images"]}

    anns_by_img: dict = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    ids = list(img_map.keys())
    np.random.seed(42)
    np.random.shuffle(ids)
    n_val = int(len(ids) * VAL_FRAC)
    splits = {"val": ids[:n_val], "train": ids[n_val:]}
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}")

    for split, split_ids in splits.items():
        img_out = OUT_DIR / split / "images"
        lbl_out = OUT_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in tqdm(split_ids, desc=split):
            meta = img_map[img_id]
            fname = meta["file_name"]
            W, H = meta["width"], meta["height"]
            shutil.copy(img_dir / fname, img_out / fname)

            lines = []
            for ann in anns_by_img.get(img_id, []):
                cls = cat_map[ann["category_id"]]
                l, t, w, h = ann["bbox"]
                cx = (l + w / 2) / W
                cy = (t + h / 2) / H
                nw = w / W
                nh = h / H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            stem = Path(fname).stem
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines))

    yaml_text = f"""path: {OUT_DIR}
train: train/images
val: val/images

nc: {len(NAMES)}
names: {NAMES}
"""
    (OUT_DIR / "dataset.yaml").write_text(yaml_text)
    print(f"Dataset written to {OUT_DIR}")
    print(f"YAML: {OUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
