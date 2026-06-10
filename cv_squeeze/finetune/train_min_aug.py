"""Minimal-aug retrain from best.pt — hypothesis test for the mosaic bug.

Hypothesis: the original v8m-ft-ep1 training used mosaic=0.5, and the cam_4662
diagnostic shows the model is attending to the sky (along horizontal bands
that look like mosaic tile seams) instead of the actual vehicles below.
Disabling mosaic + other tile-edge augs should let the model re-attend to
real objects, narrowing the val-vs-LB gap (0.978 vs 0.718).

Short test: 3 epochs from the existing best.pt seed. ~10-15 min on a free T4.
Goal: cheap signal on whether un-learning the mosaic crutch helps. If val mAP
stays high AND the cam_4662 CAM moves off the sky, run a longer retrain
overnight. If not, mosaic isn't the smoking gun and we need a different angle.

Output: cv_squeeze/finetune/runs/min_aug_3ep/weights/best.pt
"""
import os
from pathlib import Path

# Isolate ultralytics settings to cv_squeeze (don't touch shared workbench file)
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

from ultralytics import YOLO  # noqa: E402

# Canonical v8m-ft-ep1 seed (the 0.718 LB model — md5 e84ff960...)
SEED = "/home/jupyter/til-ai-26/cv_squeeze/src/weights/best.pt"
# Dataset config — teammate's preprocessed YOLO copy (read-only)
DATA = "/home/jupyter/til-ai-26/cv/finetune/data/dataset.yaml"
PROJECT = str(Path(__file__).resolve().parent / "runs")


def main():
    model = YOLO(SEED)
    model.train(
        data=DATA,
        epochs=3,
        imgsz=1280,
        batch=4,                   # T4 has 12.2 GB free w/ teammate's ASR; batch=4 fits
        freeze=0,                  # let backbone re-tune away from mosaic shortcut
        lr0=5e-4,                  # low LR — refine, don't destroy
        lrf=0.01,
        cos_lr=True,
        optimizer="SGD",           # explicit, not 'auto' (which silently switches on epoch count)
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=1,           # short warmup for short run
        patience=10,
        amp=True,                  # FP16 mixed precision
        cache="disk",
        # === MINIMAL AUG: kill anything that creates artificial tile-edge cues ===
        mosaic=0.0,                # THE suspected smoking gun
        close_mosaic=0,            # redundant w/ mosaic=0
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,               # default is 0.4 — gone
        scale=0.0,                 # default is 0.5 — gone (produces composite edges)
        translate=0.0,             # default is 0.1 — gone
        degrees=0.0,
        # === KEEP: pure geometric + color jitter (no tile artifacts) ===
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        # === housekeeping ===
        project=PROJECT,
        name="min_aug_3ep",
        exist_ok=True,
        seed=0,
        device=0,
        verbose=True,
    )


if __name__ == "__main__":
    main()
