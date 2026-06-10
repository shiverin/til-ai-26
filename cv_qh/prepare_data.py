"""Prepare TIL Novice CV dataset in RF-DETR's expected per-split COCO layout.

Reads /home/jupyter/novice/cv/annotations.json, deterministically splits image
IDs 90/10 (seed=42), and writes:

    dataset/train/_annotations.coco.json
    dataset/train/<file_name>           # symlink -> SOURCE_IMAGES/<file_name>
    dataset/valid/_annotations.coco.json
    dataset/valid/<file_name>           # symlink -> SOURCE_IMAGES/<file_name>

Idempotent: re-running deletes and recreates the split directories.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SOURCE_JSON = Path("/home/jupyter/novice/cv/annotations.json")
SOURCE_IMAGES = Path("/home/jupyter/novice/cv/images")
DATASET = ROOT / "dataset"


def _split_ids(image_ids: list[int], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(image_ids), dtype=np.int64)  # sort first for determinism
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_frac))
    return ids[n_val:].tolist(), ids[:n_val].tolist()


def _write_split(split: str, image_ids: set[int], coco: dict) -> tuple[int, int]:
    out_dir = DATASET / split
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    images = [img for img in coco["images"] if img["id"] in image_ids]
    anns = [a for a in coco["annotations"] if a["image_id"] in image_ids]

    for img in tqdm(images, desc=f"symlink {split}", unit="img"):
        link = out_dir / img["file_name"]
        target = SOURCE_IMAGES / img["file_name"]
        if not target.exists():
            raise FileNotFoundError(f"Source image missing: {target}")
        link.symlink_to(target)

    out_json = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": images,
        "annotations": anns,
    }
    (out_dir / "_annotations.coco.json").write_text(json.dumps(out_json))
    return len(images), len(anns)


def _print_class_balance(split: str, coco_path: Path, num_classes: int) -> None:
    data = json.loads(coco_path.read_text())
    counts = Counter(a["category_id"] for a in data["annotations"])
    print(f"  [{split}] per-class box counts:")
    for cid in range(num_classes):
        n = counts.get(cid, 0)
        flag = "  <-- ZERO" if n == 0 else ""
        print(f"    {cid:2d}: {n:5d}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not SOURCE_JSON.exists():
        raise FileNotFoundError(f"Missing {SOURCE_JSON}")
    coco = json.loads(SOURCE_JSON.read_text())
    all_ids = [img["id"] for img in coco["images"]]
    train_ids, val_ids = _split_ids(all_ids, args.val_frac, args.seed)
    print(f"Split: train={len(train_ids)}  val={len(val_ids)}  (seed={args.seed})")

    DATASET.mkdir(exist_ok=True)
    n_tr_img, n_tr_box = _write_split("train", set(train_ids), coco)
    n_va_img, n_va_box = _write_split("valid", set(val_ids), coco)
    print(f"Wrote train: {n_tr_img} images, {n_tr_box} boxes")
    print(f"Wrote valid: {n_va_img} images, {n_va_box} boxes")
    _print_class_balance("train", DATASET / "train" / "_annotations.coco.json", len(coco["categories"]))
    _print_class_balance("valid", DATASET / "valid" / "_annotations.coco.json", len(coco["categories"]))


if __name__ == "__main__":
    main()
