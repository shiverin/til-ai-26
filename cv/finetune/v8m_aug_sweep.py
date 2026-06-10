"""v8m_ep1 + 1-2 more epochs with a named aug regime (gentle batch=2 nudge).

Reuses REGIMES from aug_regimes.py (R1-R5). The hypothesis: the original 0.718
came from a specific 1-epoch / batch=2 / mild-aug recipe; OTHER 1-epoch
variants from the same baseline might also work because they explore different
sprite-generalization sweet spots. Submit each saved epoch to TIL.

Usage:
    python v8m_aug_sweep.py --regime R2_heavy_color
    python v8m_aug_sweep.py --regime R3_heavy_copy_paste --epochs 1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Auto-load cv/.env (for WANDB_API_KEY etc.); harmless if missing.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)
os.environ.setdefault("WANDB_MODE", "disabled")

from ultralytics import YOLO  # noqa: E402

from aug_regimes import REGIMES, aug_kwargs  # noqa: E402


def patch_noise_aug():
    """Replace ultralytics' near-zero-prob albumentations with noise-calibrated
    transforms. Use when the eval set is known to be degraded (TIL adds noise
    that's not in train). Copied from finetune/train.py."""
    import albumentations as A
    from ultralytics.data.augment import Albumentations

    transforms = A.Compose([
        A.GaussNoise(std_range=(10 / 255, 50 / 255), p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), p=0.3),
        A.MotionBlur(blur_limit=9, p=0.2),
        A.ImageCompression(quality_range=(50, 95), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
    ])
    orig_init = Albumentations.__init__

    def custom_init(self, p=1.0, **kwargs):
        orig_init(self, p=p, **kwargs)
        self.transform = transforms
        self.contains_spatial = False
        print("[v8m-sweep] noise augmentation active")

    Albumentations.__init__ = custom_init

FINETUNE_DIR = Path(__file__).resolve().parent
REPO_CV = FINETUNE_DIR.parent
DATASET_YAML = FINETUNE_DIR / "data" / "dataset.yaml"
START_WEIGHTS = REPO_CV / "src" / "weights" / "best_v8m_ep1.pt"
RUNS_DIR = FINETUNE_DIR / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True, choices=list(REGIMES))
    p.add_argument("--epochs", type=int, default=2,
                   help="Save every epoch; submit each to find TIL peak.")
    p.add_argument("--batch", type=int, default=2,
                   help="Gentle nudge from the proven 0.718 recipe.")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Matched to v8m's training imgsz.")
    p.add_argument("--lr0", type=float, default=0.001,
                   help="Low LR for fine-tune-from-converged starting point.")
    p.add_argument("--noise", action="store_true",
                   help="Add GaussNoise/blur/compression aug — TIL eval has noise.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not START_WEIGHTS.exists():
        raise SystemExit(f"missing {START_WEIGHTS}")
    aug = aug_kwargs(args.regime)
    suffix = "_noise" if args.noise else ""
    run_name = f"v8m_aug_{args.regime}{suffix}"
    print(f"=== {run_name} ===")
    print(f"aug: {aug}")
    if args.noise:
        patch_noise_aug()

    model = YOLO(str(START_WEIGHTS))
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
        name=run_name,
        exist_ok=True,
        optimizer="SGD",
        cos_lr=True,
        lr0=args.lr0,
        lrf=0.01,
        warmup_epochs=0,
        seed=0,
        close_mosaic=0,  # disable for short fine-tunes — default 10 is a no-op at 2 epochs
        **aug,
    )
    weights_dir = RUNS_DIR / run_name / "weights"
    print(f"\nDone. Checkpoints: {weights_dir}")
    print(f"epoch0.pt = +1 ep, epoch1.pt = +2 ep (relative to best_v8m_ep1)")


if __name__ == "__main__":
    main()
