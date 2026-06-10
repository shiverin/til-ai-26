"""Run EigenCAM on val images to see where the model attends.

EigenCAM is gradient-free (principal component of activations at the target
layer) — works with YOLO's multi-output forward without needing a custom
gradient story.

Pure diagnostic for the val/eval gap: if heatmaps cluster on matte-edge
halos (RESEARCH.md §1c) instead of object bodies, the model overfit to
compositing artifacts that the held-out eval images don't have.

Usage:
    python visualize_cam.py [--n 6] [--imgsz 1280] [--layer -2]
"""
import argparse
import os
import random
from pathlib import Path

# Isolate ultralytics' settings file from the shared workbench (SOLUTION.md §2).
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from pytorch_grad_cam import EigenCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from ultralytics import YOLO  # noqa: E402


FINETUNE_DIR = Path(__file__).resolve().parent
CV_DIR = FINETUNE_DIR.parent
VAL_IMAGES = FINETUNE_DIR / "data" / "val" / "images"
WEIGHTS = CV_DIR / "src" / "weights" / "best.pt"
OUT_DIR = FINETUNE_DIR / "cam_out"


class YOLOWrapper(torch.nn.Module):
    """Returns just the first (detection) tensor of YOLO's tuple output so
    pytorch-grad-cam's BaseCAM doesn't choke trying to introspect it."""

    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6,
                   help="number of val images to visualize")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument(
        "--layer", type=int, default=-2,
        help="index into model.model; -1 = Detect head, -2 = last neck block",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"no weights at {WEIGHTS}")
    if not VAL_IMAGES.exists():
        raise FileNotFoundError(f"no val images at {VAL_IMAGES}")

    print(f"[cam] loading model: {WEIGHTS}")
    yolo = YOLO(str(WEIGHTS))
    inner = yolo.model.eval()
    if torch.cuda.is_available():
        inner = inner.cuda()
    wrapped = YOLOWrapper(inner)

    target_module = list(inner.model)[args.layer]
    print(
        f"[cam] target: model.model[{args.layer}] "
        f"= {type(target_module).__name__}"
    )

    cam = EigenCAM(model=wrapped, target_layers=[target_module])

    random.seed(args.seed)
    paths = random.sample(list(VAL_IMAGES.glob("*.jpg")), args.n)

    for path in paths:
        pil = Image.open(path).convert("RGB")
        pil = ImageOps.exif_transpose(pil)
        pil = pil.resize((args.imgsz, args.imgsz))
        rgb = np.array(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        gs = cam(input_tensor=tensor)[0]   # H×W in [0, 1]
        overlay = show_cam_on_image(rgb, gs, use_rgb=True)
        out_path = OUT_DIR / f"cam_{path.stem}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  {out_path}")

    print(f"\n[cam] done. Open files in {OUT_DIR}/")


if __name__ == "__main__":
    main()
