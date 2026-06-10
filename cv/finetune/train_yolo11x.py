"""Fine-tune yolo11x from COCO on the TIL Novice CV dataset.

Single-shot trainer for the 0.727 → 0.76 push. R1 light aug (the recipe that
produced the 0.727 winner), save_period=1, submit every checkpoint to TIL.

Usage:
    python finetune/train_yolo11x.py             # 3 epochs default
    python finetune/train_yolo11x.py --epochs 5  # more epochs
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)
os.environ.setdefault("WANDB_MODE", "disabled")

from ultralytics import YOLO  # noqa: E402

FINETUNE_DIR = Path(__file__).resolve().parent
REPO_CV = FINETUNE_DIR.parent
DATASET_YAML = FINETUNE_DIR / "data" / "dataset.yaml"
PRETRAINED = REPO_CV / "pretrained" / "yolo11x.pt"
RUNS_DIR = FINETUNE_DIR / "runs"

# R1 light aug — same recipe that produced the v8m 0.727 winner.
R1_LIGHT = dict(
    hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
    copy_paste=0.3, mixup=0.0, mosaic=1.0,
    degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
    fliplr=0.5, flipud=0.0, erasing=0.0,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=2,
                   help="T4 15GB-safe at imgsz=1280; batch=4 OOMs.")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--lr0", type=float, default=0.01,
                   help="Default ultralytics fine-tune LR; COCO→task transfer.")
    p.add_argument("--name", default="yolo11x_r1_light")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not PRETRAINED.exists():
        raise SystemExit(f"missing {PRETRAINED}")

    print(f"=== {args.name} ===")
    print(f"start={PRETRAINED.name} epochs={args.epochs} batch={args.batch} "
          f"imgsz={args.imgsz} lr0={args.lr0}")
    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        amp=True,
        workers=4,
        save_period=1,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        optimizer="SGD",
        cos_lr=True,
        lr0=args.lr0,
        lrf=0.01,
        warmup_epochs=1,
        seed=0,
        close_mosaic=10,
        **R1_LIGHT,
    )
    weights_dir = RUNS_DIR / args.name / "weights"
    print(f"\nDone. Checkpoints: {weights_dir}")


if __name__ == "__main__":
    main()
