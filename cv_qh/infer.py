"""Run RF-DETR inference over a directory of images and emit COCO-results JSON.

Output records: {"image_id": <stem>, "category_id": int, "bbox": [l,t,w,h],
"score": float}. Boxes are zero-indexed integer LTWH pixels, clipped to image
bounds — the wire format cv/cv_server emits.

image_id is the image file stem (e.g. "3235" from "3235.jpg") so downstream
COCO eval can join against an annotations file that uses int ids.

category_id is shifted by -1 from the raw RF-DETR output: the fine-tuned model
emits class IDs in [1, num_classes] (class 0 reserved for "no object" in DETR
postprocess), while the source dataset and class_names.py use 0-indexed IDs in
[0, num_classes - 1].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from rfdetr import RFDETRBase, RFDETRLarge


def _load_model(weights: Path, variant: str, resolution: int | None, device: str | None):
    cls = {"base": RFDETRBase, "large": RFDETRLarge}[variant]
    init_kwargs: dict = {"pretrain_weights": str(weights)}
    if resolution is not None:
        init_kwargs["resolution"] = resolution
    if device is not None:
        init_kwargs["device"] = device
    return cls(**init_kwargs)


def _detections_to_records(image_id: int | str, dets, img_w: int, img_h: int, conf: float) -> list[dict]:
    """Convert one image's supervision.Detections → COCO-results dicts.

    Filter by score >= conf. Clip xyxy boxes to the image, convert to int LTWH,
    drop degenerate (w<=0 or h<=0) boxes.
    """
    out = []
    xyxy = dets.xyxy            # (N, 4) float
    cls_ids = dets.class_id     # (N,) int
    scores = dets.confidence    # (N,) float
    for i in range(len(xyxy)):
        s = float(scores[i])
        if s < conf:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        x1 = max(0.0, min(x1, img_w))
        y1 = max(0.0, min(y1, img_h))
        x2 = max(0.0, min(x2, img_w))
        y2 = max(0.0, min(y2, img_h))
        l, t = int(round(x1)), int(round(y1))
        w, h = int(round(x2 - x1)), int(round(y2 - y1))
        if w <= 0 or h <= 0:
            continue
        if l + w > img_w:
            w = img_w - l
        if t + h > img_h:
            h = img_h - t
        if w <= 0 or h <= 0:
            continue
        # RF-DETR emits 1-indexed class IDs (class 0 = no-object). Source
        # dataset / class_names.py use 0-indexed IDs, so shift back.
        out.append({
            "image_id": image_id,
            "category_id": int(cls_ids[i]) - 1,
            "bbox": [l, t, w, h],
            "score": s,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Directory of .jpg images.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "large"), default="base")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Score floor; default keeps all dets for mAP ranking.")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Override torch device (e.g. 'cpu', 'cuda', 'cuda:0').")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if not args.images.is_dir():
        raise NotADirectoryError(args.images)

    import torch
    device = args.device
    if device is None and not torch.cuda.is_available():
        device = "cpu"
    model = _load_model(args.weights, args.variant, args.resolution, device)

    image_paths = sorted(p for p in args.images.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise FileNotFoundError(f"No images in {args.images}")

    records: list[dict] = []
    for p in tqdm(image_paths, desc="infer", unit="img"):
        with Image.open(p) as im:
            img = im.convert("RGB")
            img_w, img_h = img.size
            dets = model.predict(img, threshold=args.conf)
        stem = p.stem
        image_id: int | str = int(stem) if stem.isdigit() else stem
        records.extend(_detections_to_records(image_id, dets, img_w, img_h, args.conf))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records))
    print(f"Wrote {len(records)} detections across {len(image_paths)} images → {args.out}")


if __name__ == "__main__":
    main()
