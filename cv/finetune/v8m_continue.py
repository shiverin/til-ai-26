"""Conservative continuation training for the real v8m ep1 checkpoint.

This is intentionally different from ``archive/cv_og/train.py``: it starts
from the proven ep1 checkpoint and uses a small LR plus gentler augmentation so
each saved epoch can be confidence-swept before any Docker image is promoted.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep Ultralytics settings local to the repo.
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

from ultralytics import YOLO  # noqa: E402


FINETUNE_DIR = Path(__file__).resolve().parent
REPO_CV = FINETUNE_DIR.parent
DATASET_YAML = FINETUNE_DIR / "data" / "dataset.yaml"
START_WEIGHTS = REPO_CV / "src" / "weights" / "best_v8m_ep1.pt"
RUNS_DIR = FINETUNE_DIR / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=START_WEIGHTS)
    parser.add_argument("--name", default="v8m_ep1_continue")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "auto"])
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(weights)
    print(f"[v8m-continue] starting from: {weights}")
    model = YOLO(str(weights))
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        amp=True,
        patience=args.patience,
        workers=args.workers,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        resume=False,
        save_period=args.save_period,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        warmup_epochs=0,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.05,
    )
    print(f"[v8m-continue] best model: {results.save_dir}/weights/best.pt")
    print(f"[v8m-continue] mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print("[v8m-continue] next: sweep confidence on best.pt and each saved epoch worth testing")


if __name__ == "__main__":
    main()
