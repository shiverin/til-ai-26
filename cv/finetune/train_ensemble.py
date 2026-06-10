"""Train one nano model with a named aug regime. See spec §3 / aug_regimes.py.

Usage:
    python train_ensemble.py --slot M1
    python train_ensemble.py --slot M3 --epochs 30 --batch 16
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

# Auto-load cv/.env so WANDB_API_KEY is available before wandb import.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Isolate ultralytics settings file (SOLUTION.md §2).
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

from ultralytics import YOLO  # noqa: E402

from aug_regimes import SLOT_MAP, REGIMES, aug_kwargs  # noqa: E402

FINETUNE_DIR = Path(__file__).resolve().parent
REPO_CV = FINETUNE_DIR.parent
DATASET_YAML = FINETUNE_DIR / "data" / "dataset.yaml"
RUNS_DIR = FINETUNE_DIR / "runs"

PRETRAINED = {
    "yolov8n": REPO_CV / "pretrained" / "yolov8n.pt",
    "yolo11n": REPO_CV / "yolo11n.pt",
    "yolo26n": REPO_CV / "yolo26n.pt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--slot", required=True, choices=list(SLOT_MAP),
                   help="M1..M5 — selects backbone and aug regime per spec §2/§3.")
    p.add_argument("--epochs", type=int, default=30,
                   help="Max epochs. Save every epoch; submit-driven epoch picking.")
    p.add_argument("--batch", type=int, default=16,
                   help="Nano models fit comfortably at batch 16 on T4 15GB.")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch on 5%% data for pipeline check.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backbone, regime = SLOT_MAP[args.slot]
    pretrained = PRETRAINED[backbone]
    if not pretrained.exists():
        raise SystemExit(f"missing pretrained weights: {pretrained}")

    aug = aug_kwargs(regime)
    epoch_window = REGIMES[regime]["epoch_window"]

    run_name = f"{args.slot}_{backbone}_{regime}"
    epochs = 1 if args.smoke else args.epochs
    fraction = 0.05 if args.smoke else 1.0

    print(f"=== {run_name} ===")
    print(f"backbone: {pretrained}")
    print(f"regime: {regime}, epoch_window for submission: {epoch_window}")
    print(f"aug kwargs: {aug}")

    model = YOLO(str(pretrained))
    model.train(
        data=str(DATASET_YAML),
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        amp=True,
        workers=4,
        cache="disk",
        seed=0,
        optimizer="SGD",
        cos_lr=True,
        lrf=0.01,
        save_period=1,
        epochs=epochs,
        fraction=fraction,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        **aug,
    )

    # Release VRAM (training keeps the optimizer state around).
    del model
    torch.cuda.empty_cache()
    print(f"\nDone. Checkpoints: {RUNS_DIR / run_name / 'weights'}")
    print(f"Submit checkpoints from epoch_window={epoch_window} (epoch{epoch_window[0]}.pt … epoch{epoch_window[1]}.pt)")


if __name__ == "__main__":
    main()
