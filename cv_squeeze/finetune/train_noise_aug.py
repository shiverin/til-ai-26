"""Soft-outline augmentation for ultralytics YOLO training.

Defines YOLONoiseAugDataset — a YOLODataset that, with probability p,
replaces each image with a softer-silhouette variant using precomputed
GrabCut masks. The softening is a distance-transform-based Gaussian
gradient at the object outline (no body blur, no background change).

Goal: force the model to find features other than the matte-halo edge
shortcut introduced by composite training data (see
docs/adversarial-noise-aug-design.md).

To activate:
    from train_noise_aug import patch_outline_aug
    patch_outline_aug()        # before model.train()

Env vars (all optional):
    NOISE_AUG_P        # default 0.5   — probability per sample
    NOISE_AUG_SIGMA_LO # default 8.0   — lower bound for gradient sigma
    NOISE_AUG_SIGMA_HI # default 18.0  — upper bound for gradient sigma
    NOISE_AUG_KSIZE    # default 41    — local Gaussian blur kernel size

Masks must be precomputed first via precompute_masks.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset

HERE = Path(__file__).resolve().parent
CV_SQUEEZE = HERE.parent
MASKS_ROOT = CV_SQUEEZE / "finetune" / "data" / "masks"


def _gradient_blend(img: np.ndarray, fg_mask: np.ndarray,
                    sigma_px: float, blur_ksize: int) -> np.ndarray:
    """Distance-transform-based silhouette gradient blend.

    Same math as noise_attacker.soften_object_outline_gc but with a
    precomputed `fg_mask` (uint8, 255 = foreground) instead of running
    GrabCut at call time.
    """
    if fg_mask.sum() == 0:
        return img
    dist_in = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(255 - fg_mask, cv2.DIST_L2, 3)
    signed = dist_out - dist_in
    alpha = np.exp(-(signed * signed) / (2.0 * sigma_px * sigma_px))
    alpha_3 = np.clip(alpha, 0.0, 1.0)[..., None].astype(np.float32)
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    out = (img.astype(np.float32) * (1.0 - alpha_3)
           + blurred.astype(np.float32) * alpha_3)
    return out.astype(np.uint8)


class YOLONoiseAugDataset(YOLODataset):
    """YOLODataset that softens object outlines via precomputed masks.

    Override is on `get_image_and_label` (the per-image loader) so the
    aug applies BEFORE mosaic/affine — mosaic then naturally combines
    4 already-softened source images. Val mode is unaffected.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_train = "train" in self.prefix
        self.noise_p = float(os.environ.get("NOISE_AUG_P", 0.5))
        self.sigma_lo = float(os.environ.get("NOISE_AUG_SIGMA_LO", 8.0))
        self.sigma_hi = float(os.environ.get("NOISE_AUG_SIGMA_HI", 18.0))
        self.blur_ksize = int(os.environ.get("NOISE_AUG_KSIZE", 41))
        split = "train" if self.is_train else "val"
        self.masks_dir = MASKS_ROOT / split
        if self.is_train:
            n_masks = sum(1 for _ in self.masks_dir.glob("*.png")) \
                if self.masks_dir.exists() else 0
            print(
                f"[noise-aug] active: p={self.noise_p:.2f}, "
                f"sigma=({self.sigma_lo:.1f}, {self.sigma_hi:.1f}), "
                f"ksize={self.blur_ksize}, masks={n_masks}"
            )

    def get_image_and_label(self, index):
        sample = super().get_image_and_label(index)
        if not self.is_train or np.random.random() > self.noise_p:
            return sample

        img_path = Path(self.im_files[index])
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            return sample
        fg = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if fg is None or fg.sum() == 0:
            return sample

        img = sample["img"]
        h, w = img.shape[:2]
        if fg.shape != (h, w):
            fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)

        sigma = float(np.random.uniform(self.sigma_lo, self.sigma_hi))
        sample["img"] = _gradient_blend(img, fg, sigma, self.blur_ksize)
        return sample


def patch_outline_aug():
    """Monkeypatch ultralytics to use YOLONoiseAugDataset.

    Mirrors weighted_dataset.patch_weighted_dataset(). Call BEFORE
    model.train(). Compose with patch_weighted_dataset() by stacking
    subclasses if both are wanted (the second-applied patch wins
    unless its subclass extends the first).
    """
    import ultralytics.data.build as build
    build.YOLODataset = YOLONoiseAugDataset
    p = os.environ.get("NOISE_AUG_P", "0.5")
    print(f"[finetune] outline-aug dataloader active (p={p})")
