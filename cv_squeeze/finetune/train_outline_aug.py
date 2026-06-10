"""3-epoch retrain from best.pt with the soft-outline GrabCut augmentation.

Reads precomputed silhouette masks from cv_squeeze/finetune/data/masks/
(produced by precompute_masks.py) and at training time applies a
distance-transform-based gradient blend at each object's outline with
probability NOISE_AUG_P (default 0.5).

Hypothesis: the L-15 frame-edge attention shrinks when the model can no
longer shortcut on the matte-halo cue, forcing it to learn features from
the object body. Detection retention should hold; CAM heat should move
off the silhouette toward the object interior.

Usage:
    python train_outline_aug.py                         # 3-ep default
    NOISE_AUG_P=0.7 python train_outline_aug.py         # heavier aug
    NOISE_AUG_SIGMA_LO=12 NOISE_AUG_SIGMA_HI=22 python train_outline_aug.py

Output: cv_squeeze/finetune/runs/outline_aug_3ep/weights/best.pt
"""
import os
from pathlib import Path

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

from ultralytics import YOLO  # noqa: E402

from train_noise_aug import patch_outline_aug  # noqa: E402

SEED = "/home/jupyter/til-ai-26/cv_squeeze/src/weights/best.pt"
DATA = "/home/jupyter/til-ai-26/cv/finetune/data/dataset.yaml"
PROJECT = str(Path(__file__).resolve().parent / "runs")


def main():
    patch_outline_aug()
    model = YOLO(SEED)
    model.train(
        data=DATA,
        epochs=3,
        imgsz=1280,
        batch=4,
        freeze=0,
        lr0=5e-4,
        lrf=0.01,
        cos_lr=True,
        optimizer="SGD",
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=1,
        patience=10,
        amp=True,
        cache="disk",
        # Same minimal-aug recipe as train_min_aug.py — kill mosaic/mixup/etc
        # so the only "new" signal is our outline softening. Otherwise we
        # can't separate aug effects from training noise.
        mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
        erasing=0.0, scale=0.0, translate=0.0, degrees=0.0,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        project=PROJECT,
        name="outline_aug_3ep",
        exist_ok=True,
        seed=0,
        device=0,
        verbose=True,
    )


if __name__ == "__main__":
    main()
