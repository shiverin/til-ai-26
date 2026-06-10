"""Precompute GrabCut foreground masks for every training/val image.

For each image, run cv2.grabCut on each GT box and save the union foreground
mask as a single-channel PNG (255 = foreground, 0 = background). Used by
train_noise_aug.py at training time to apply a distance-transform-based
gradient blend at the silhouette — paid once here, free per batch.

Usage:
    python precompute_masks.py --split both          # train + val
    python precompute_masks.py --split val --limit 8  # smoke test on 8 imgs
    python precompute_masks.py --workers 12

Outputs:
    cv_squeeze/finetune/data/masks/train/{img_id}.png
    cv_squeeze/finetune/data/masks/val/{img_id}.png

Resumable: existing mask PNGs are skipped, so re-running picks up where
a previous run was interrupted.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CV_SQUEEZE = Path(__file__).resolve().parent.parent
TRAIN_IMAGES = Path("/home/jupyter/til-ai-26/cv/finetune/data/train/images")
TRAIN_LABELS = Path("/home/jupyter/til-ai-26/cv/finetune/data/train/labels")
VAL_IMAGES = Path("/home/jupyter/til-ai-26/cv/finetune/data/val/images")
VAL_LABELS = Path("/home/jupyter/til-ai-26/cv/finetune/data/val/labels")
MASKS_ROOT = CV_SQUEEZE / "finetune" / "data" / "masks"


def load_boxes(lbl_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """YOLO labels (class cx cy w h, normalized) → Nx4 xyxy in pixel coords."""
    if not lbl_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    out = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        out.append([
            (cx - bw / 2) * img_w,
            (cy - bh / 2) * img_h,
            (cx + bw / 2) * img_w,
            (cy + bh / 2) * img_h,
        ])
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)


PAD_PX = 20  # crop padding around each box before running GrabCut


def grabcut_one(args):
    img_path, lbl_path, out_path, iter_count = args
    if out_path.exists():
        return None  # resume support
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return f"unreadable: {img_path.name}"
    h, w = bgr.shape[:2]
    boxes = load_boxes(lbl_path, w, h)
    if len(boxes) == 0:
        # No labels: empty mask so the trainer can no-op
        cv2.imwrite(str(out_path), np.zeros((h, w), dtype=np.uint8))
        return None
    fg_union = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        rw, rh = x2 - x1, y2 - y1
        if rw < 4 or rh < 4:
            continue
        # Crop region around the box — GrabCut runs O(area * iters), so
        # cropping turns this from ~seconds per image into ~10s of ms.
        cx1 = max(0, x1 - PAD_PX); cy1 = max(0, y1 - PAD_PX)
        cx2 = min(w, x2 + PAD_PX); cy2 = min(h, y2 + PAD_PX)
        crop = bgr[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        # Box coords inside the crop
        bx, by = x1 - cx1, y1 - cy1
        mask = np.zeros((ch, cw), dtype=np.uint8)
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(
                crop, mask, (bx, by, rw, rh),
                bgd, fgd, iter_count, cv2.GC_INIT_WITH_RECT,
            )
        except cv2.error:
            continue
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0,
        ).astype(np.uint8)
        fg_union[cy1:cy2, cx1:cx2] = np.maximum(
            fg_union[cy1:cy2, cx1:cx2], fg,
        )
    cv2.imwrite(str(out_path), fg_union)
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--iter-count", type=int, default=5,
                   help="GrabCut iterations; more = better mask, slower")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--limit", type=int, default=0,
                   help="if > 0, only process first N images (smoke test)")
    return p.parse_args()


def collect_jobs(images_dir, labels_dir, out_dir, iter_count, limit):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem
        jobs.append((
            img_path,
            labels_dir / f"{stem}.txt",
            out_dir / f"{stem}.png",
            iter_count,
        ))
    if limit > 0:
        jobs = jobs[:limit]
    return jobs


def main():
    args = parse_args()
    jobs = []
    if args.split in ("train", "both"):
        jobs.extend(collect_jobs(
            TRAIN_IMAGES, TRAIN_LABELS, MASKS_ROOT / "train",
            args.iter_count, args.limit,
        ))
    if args.split in ("val", "both"):
        jobs.extend(collect_jobs(
            VAL_IMAGES, VAL_LABELS, MASKS_ROOT / "val",
            args.iter_count, args.limit,
        ))
    print(f"[precompute] {len(jobs)} jobs across {args.workers} workers")
    print(f"[precompute] writing to {MASKS_ROOT}/")
    if not jobs:
        return

    errors = []
    with mp.Pool(args.workers) as pool:
        for err in tqdm(
            pool.imap_unordered(grabcut_one, jobs),
            total=len(jobs), desc="grabcut",
        ):
            if err is not None:
                errors.append(err)

    print(f"\n[precompute] done. errors: {len(errors)}")
    for e in errors[:10]:
        print(f"  {e}")


if __name__ == "__main__":
    main()
