"""EigenCAM on (clean, blur_object, soft_outline_gc) images.

Answers: do the box-targeted edge interventions actually reduce L-15
frame-edge attention, or is the model's internal heatmap unchanged
(in which case the aug isn't fixing the bug at the layer we care about)?

Output: cv/finetune/cam_out/<tag>/cam_{img}_L{layer}_{transform}.jpg
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV_SQUEEZE = HERE.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(CV_SQUEEZE / ".ultralytics"))

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CV_SQUEEZE / "finetune"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pytorch_grad_cam import EigenCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import noise_attacker  # noqa: E402
from test_attacker import load_gt_boxes  # noqa: E402
from visualize_cam import YOLOWrapper  # noqa: E402

DEFAULT_WEIGHTS = CV_SQUEEZE / "src" / "weights" / "best.pt"
VAL_IMAGES = Path("/home/jupyter/til-ai-26/cv/finetune/data/val/images")
CAM_OUT_ROOT = Path("/home/jupyter/til-ai-26/cv/finetune/cam_out")
IMGSZ = 1024  # smaller than 1280 to survive GPU contention w/ teammate

LAYERS = [-2, -5, -8, -11, -15]
IMG_IDS = [4662, 1567, 1852, 4969]
JOBS = [(img_id, layer) for img_id in IMG_IDS for layer in LAYERS]


def scale_boxes(boxes: np.ndarray, orig_w: int, orig_h: int,
                tgt_w: int, tgt_h: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    sx = tgt_w / float(orig_w)
    sy = tgt_h / float(orig_h)
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


def run_cam(cam: EigenCAM, rgb_uint8: np.ndarray) -> np.ndarray:
    rgb_f = rgb_uint8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_f).permute(2, 0, 1).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    gs = cam(input_tensor=tensor)[0]
    return show_cam_on_image(rgb_f, gs, use_rgb=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                   help="path to a YOLO best.pt")
    p.add_argument("--tag", required=True,
                   help="run label; outputs go to cv/finetune/cam_out/<tag>/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    out_dir = CAM_OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cam] loading {weights}")
    yolo = YOLO(str(weights))
    inner = yolo.model.eval()
    if torch.cuda.is_available():
        inner = inner.cuda()
    wrapped = YOLOWrapper(inner)

    for img_id, layer in JOBS:
        img_path = VAL_IMAGES / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]
        rgb_r = cv2.resize(rgb, (IMGSZ, IMGSZ))
        boxes = load_gt_boxes(img_id, orig_w, orig_h)
        boxes_r = scale_boxes(boxes, orig_w, orig_h, IMGSZ, IMGSZ)

        transforms = [
            ("clean", lambda im: im),
            ("blur_object", lambda im: noise_attacker.blur_object_edges(im, boxes_r)),
            ("soft_outline_gc", lambda im: noise_attacker.soften_object_outline_gc(im, boxes_r)),
        ]

        target = list(inner.model)[layer]
        print(f"\n[cam] img={img_id} layer={layer} ({type(target).__name__})")
        cam = EigenCAM(model=wrapped, target_layers=[target])

        for name, fn in transforms:
            perturbed = fn(rgb_r)
            overlay = run_cam(cam, perturbed)
            out = out_dir / f"cam_{img_id}_L{layer}_{name}.jpg"
            cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"  {name:18s} -> {out.name}")


if __name__ == "__main__":
    main()
