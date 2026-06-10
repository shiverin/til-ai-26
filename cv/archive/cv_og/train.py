"""Fine-tune YOLOv8m on the novice CV dataset."""

import albumentations as A
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations

BASE_DIR = "/home/jupyter/til-ai-26/cv/cv_og"
DATASET_YAML = f"{BASE_DIR}/cv_yolo/dataset.yaml"
START_WEIGHTS = f"{BASE_DIR}/src/weights/best.pt"
EPOCHS = 80
IMGSZ = 1280
BATCH = 8
PROJECT = f"{BASE_DIR}/cv_runs"
NAME = "yolov8m_noise"

# Noise budget from eval_thresholds_v2.yaml:
#   global RMSE <= 67, inside-bbox RMSE <= 50, inside-bbox SSIM >= 0.3
_NOISE_TRANSFORMS = A.Compose(
    [
        A.GaussNoise(std_range=(10 / 255, 50 / 255), p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), p=0.3),
        A.MotionBlur(blur_limit=9, p=0.2),
        A.ImageCompression(quality_range=(50, 95), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
    ],
)


def _patch_albumentations():
    """Replace ultralytics' near-zero-prob defaults with noise-calibrated transforms."""
    _orig_init = Albumentations.__init__

    def custom_init(self, p=1.0, **kwargs):
        _orig_init(self, p=p, **kwargs)
        self.transform = _NOISE_TRANSFORMS
        self.contains_spatial = False  # all transforms are pixel-level only
        print("Albumentations: noise augmentation active")

    Albumentations.__init__ = custom_init


def main():
    import os
    _patch_albumentations()

    weights = START_WEIGHTS if os.path.exists(START_WEIGHTS) else "yolov8m.pt"
    print(f"Starting from: {weights}")
    model = YOLO(weights)
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        amp=True,
        patience=20,
        workers=4,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        resume=False,
        save_period=5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == "__main__":
    main()
