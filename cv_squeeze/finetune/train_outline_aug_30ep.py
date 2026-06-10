"""30-epoch scale-up training with soft-outline GrabCut aug.

Resumes from the 3-ep outline_aug ckpt (now promoted to
cv_squeeze/src/weights/best.pt). Same recipe as train_outline_aug.py
but longer + checkpoint every 5 epochs so we can roll back if/when val
loss diverges from train loss (overfit signal).

Ultralytics auto-writes results.csv and results.png to the run dir after
each epoch — that's the train+val loss curve to watch.

Usage:
    nohup python train_outline_aug_30ep.py > 30ep.log 2>&1 &

Output: cv_squeeze/finetune/runs/outline_aug_30ep/weights/{best,last,epoch{N}}.pt
        cv_squeeze/finetune/runs/outline_aug_30ep/results.{csv,png}
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
        epochs=30,
        save_period=5,             # checkpoint every 5 epochs
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
        patience=15,               # generous — we want to see the curve
        amp=True,
        cache="disk",
        # Minimal-aug recipe — keep the only "new" signal as the outline blur
        mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
        erasing=0.0, scale=0.0, translate=0.0, degrees=0.0,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        project=PROJECT,
        name="outline_aug_30ep",
        exist_ok=True,
        seed=0,
        device=0,
        verbose=True,
    )


if __name__ == "__main__":
    main()
