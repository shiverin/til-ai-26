"""Run EigenCAM on val images to see where the model attends.

Same EigenCAM machinery as teammate's cv/finetune/visualize_cam.py, but
copied here so we can point it at different weights (e.g. v8m-ft-ep1 vs the
minimal-aug retrain output) and write to cv_squeeze/cam_out/.

Usage:
    python visualize_cam.py \
        --weights /home/jupyter/til-ai-26/cv_squeeze/src/weights/best.pt \
        --image 4662 \
        --tag v8m_ft_ep1

    python visualize_cam.py \
        --weights /home/jupyter/til-ai-26/cv_squeeze/finetune/runs/min_aug_3ep/weights/best.pt \
        --image 4662 \
        --tag min_aug_3ep
"""
import argparse
import os
from pathlib import Path

# Isolate ultralytics' settings file from the shared workbench.
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

CV_SQUEEZE = Path(__file__).resolve().parent.parent
VAL_IMAGES = Path("/home/jupyter/til-ai-26/cv/finetune/data/val/images")
OUT_DIR = CV_SQUEEZE / "cam_out"


class YOLOWrapper(torch.nn.Module):
    """Returns just the first (detection) tensor of YOLO's tuple output so
    pytorch-grad-cam's BaseCAM doesn't choke."""

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
    p.add_argument("--weights", required=True,
                   help="path to a YOLO best.pt")
    p.add_argument("--image", default="4662",
                   help="val image stem (e.g. '4662' loads 4662.jpg)")
    p.add_argument("--tag", required=True,
                   help="label baked into the output filename — distinguishes models")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument(
        "--layer", type=int, default=-2,
        help="index into model.model; -1 = Detect head, -2 = last neck block",
    )
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"no weights at {weights}")
    img_path = VAL_IMAGES / f"{args.image}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"no val image at {img_path}")

    print(f"[cam] loading model: {weights}")
    yolo = YOLO(str(weights))
    inner = yolo.model.eval()
    if torch.cuda.is_available():
        inner = inner.cuda()
    wrapped = YOLOWrapper(inner)

    target_module = list(inner.model)[args.layer]
    print(
        f"[cam] target: model.model[{args.layer}] = {type(target_module).__name__}"
    )
    cam = EigenCAM(model=wrapped, target_layers=[target_module])

    pil = Image.open(img_path).convert("RGB")
    pil = ImageOps.exif_transpose(pil)
    pil = pil.resize((args.imgsz, args.imgsz))
    rgb = np.array(pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    gs = cam(input_tensor=tensor)[0]   # H×W in [0, 1]
    overlay = show_cam_on_image(rgb, gs, use_rgb=True)
    out_path = OUT_DIR / f"cam_{args.image}_{args.tag}_L{args.layer}.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[cam] wrote {out_path}")


if __name__ == "__main__":
    main()
