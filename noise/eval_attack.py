"""Simple held-out attack-effect benchmark: pick a fine-tuned checkpoint
that is NOT in the surrogate ensemble, run it on clean vs noised images,
report the drop in number-of-confident-detections.

We deliberately do NOT compute mAP — the held-out checkpoint may detect
slightly different things from the GT, so a raw detection count is a more
honest proxy for "did the attack hurt this model" without needing label
plumbing.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

NOISE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOISE_DIR))
from src.noise_manager import NoiseManager

VAL_IMG_DIR = Path("/home/jupyter/novice/cv/images")
N_IMAGES = 200
CONF_THRESHOLD = 0.4

# Held-out checkpoint = a fine-tuned model NOT in the surrogate ensemble.
CANDIDATE_HELDOUTS = [
    NOISE_DIR.parent / "cv" / "finetune" / "runs" / "v8m_aug_R1_light" / "weights" / "best.pt",
    NOISE_DIR.parent / "cv" / "finetune" / "runs" / "v8m_aug_R2_heavy_color" / "weights" / "best.pt",
    NOISE_DIR.parent / "cv" / "finetune" / "runs" / "v8m_aug_R3_heavy_copy_paste" / "weights" / "best.pt",
    NOISE_DIR.parent / "cv" / "finetune" / "runs" / "v8m_aug_R4_mosaic_mixup" / "weights" / "best.pt",
]


def pick_heldout() -> Path:
    for p in CANDIDATE_HELDOUTS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "no held-out checkpoint found; expected one of:\n" +
        "\n".join(f"  {p}" for p in CANDIDATE_HELDOUTS))


def count_detections(model, arr: np.ndarray) -> int:
    res = model.predict(arr, conf=CONF_THRESHOLD, verbose=False,
                        device=0 if torch.cuda.is_available() else "cpu")
    return int(res[0].boxes.shape[0]) if res[0].boxes is not None else 0


def main() -> int:
    held_out_path = pick_heldout()
    print(f"held-out detector: {held_out_path}")
    model = YOLO(str(held_out_path))

    imgs = sorted(VAL_IMG_DIR.glob("*.jpg"))[:N_IMAGES]
    print(f"running on {len(imgs)} images")

    mgr = NoiseManager()
    total_clean = 0
    total_noised = 0
    for img_path in tqdm(imgs):
        orig = np.array(Image.open(img_path).convert("RGB"))
        buf = io.BytesIO()
        Image.fromarray(orig).save(buf, format="PNG")
        out_b64 = mgr.noise(buf.getvalue())
        noised = np.array(
            Image.open(io.BytesIO(base64.b64decode(out_b64))).convert("RGB"))
        total_clean += count_detections(model, orig)
        total_noised += count_detections(model, noised)

    drop_pct = 1.0 - (total_noised / max(1, total_clean))
    print(f"clean detections:  {total_clean}")
    print(f"noised detections: {total_noised}")
    print(f"detection drop:    {drop_pct:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
