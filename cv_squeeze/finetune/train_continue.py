"""Continue v8m-ft-ep1 training for 5 more epochs at low LR.

The 0.718-LB ckpt was saved after only 1 of 30 planned epochs — the cosine
LR was barely past warmup. This script picks up from there with HALF the
original lr0 (5e-4 instead of 1e-3) so we refine rather than reset, and
keeps the EXACT original aug recipe (mosaic=0.5, erasing=0.4) so any LB
movement is attributable to training time alone, not aug changes.

Output: cv_squeeze/finetune/runs/v8m_ft_ep6/weights/best.pt
"""
import os
from pathlib import Path

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

from ultralytics import YOLO  # noqa: E402

# Start from the archive copy (untouched original 0.718-LB weights).
SEED = "/home/jupyter/til-ai-26/cv/archive/v8m_ft_ep1.pt"
DATA = "/home/jupyter/til-ai-26/cv/finetune/data/dataset.yaml"
PROJECT = str(Path(__file__).resolve().parent / "runs")


def main():
    model = YOLO(SEED)
    model.train(
        data=DATA,
        epochs=5,
        imgsz=1280,
        batch=4,                   # OG was batch=2; bumped because GPU free now
        freeze=0,
        lr0=5e-4,                  # half of OG 1e-3 — refine not reset
        lrf=0.01,
        cos_lr=True,
        optimizer="AdamW",         # OG used 'auto' which picks AdamW
        weight_decay=0.0005,
        warmup_epochs=0,           # already past warmup
        patience=15,
        amp=True,
        cache="disk",
        # === EXACT original aug recipe (from v8m_ft_ep1 train_args) ===
        mosaic=0.5, close_mosaic=10,
        erasing=0.4,
        scale=0.5, translate=0.1, degrees=0.0,
        fliplr=0.5, flipud=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mixup=0.0, copy_paste=0.0,
        project=PROJECT,
        name="v8m_ft_ep6",
        exist_ok=True,
        seed=0,
        device=0,
        verbose=True,
    )


if __name__ == "__main__":
    main()
