"""Five aug regimes for the nano ensemble. Each model gets one regime; the
diversity in aug is the primary diversity source. See spec §3 for rationale.

`epoch_window` is the range of *checkpoint epochs* worth submitting to TIL
(not a code-side early-stop). Local val mAP is an anti-proxy for TIL — pick
by submission, not by val mAP.
"""
from __future__ import annotations

REGIMES = {
    "R1_light": dict(
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
        copy_paste=0.3, mixup=0.0, mosaic=1.0,
        degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0, erasing=0.0,
        epoch_window=(10, 15),
    ),
    "R2_heavy_color": dict(
        hsv_h=0.05, hsv_s=0.8, hsv_v=0.8,
        copy_paste=0.3, mixup=0.0, mosaic=1.0,
        degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0, erasing=0.0,
        epoch_window=(25, 30),
    ),
    "R3_heavy_copy_paste": dict(
        hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
        copy_paste=0.6, mixup=0.0, mosaic=1.0,
        degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0, erasing=0.0,
        epoch_window=(25, 30),
    ),
    "R4_mosaic_mixup": dict(
        hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
        copy_paste=0.3, mixup=0.15, mosaic=1.0,
        degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0, erasing=0.0,
        epoch_window=(25, 30),
    ),
    "R5_anti_overfit": dict(
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
        copy_paste=0.2, mixup=0.0, mosaic=0.5,
        degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0, erasing=0.0,
        epoch_window=(3, 5),
    ),
}

SLOT_MAP = {
    "M1": ("yolov8n", "R1_light"),
    "M2": ("yolo11n", "R2_heavy_color"),
    "M3": ("yolo11n", "R3_heavy_copy_paste"),
    "M4": ("yolo26n", "R4_mosaic_mixup"),
    "M5": ("yolo26n", "R5_anti_overfit"),
}


def aug_kwargs(regime_name: str) -> dict:
    """Return the kwargs to pass to ultralytics `model.train(**...)`, with
    `epoch_window` stripped (it's documentation, not a kwarg)."""
    cfg = dict(REGIMES[regime_name])
    cfg.pop("epoch_window")
    return cfg
