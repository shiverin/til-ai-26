"""Sweep the inference confidence threshold to maximize COCO mAP.

The TIL harness (test/test_cv.py) scores every detection with a hard-coded
score=1.0, so AP is decided purely by WHICH boxes are emitted. This script
predicts once at a low conf, then re-thresholds + re-scores to find the
mAP-maximising conf. Set the serving CVManager conf to the printed value.

Usage:
    python sweep_conf.py --weights runs/stage2/weights/best.pt
    python sweep_conf.py --weights src/weights/best_v8m_ep1.pt \
        --confs 0.30,0.35,0.40,0.45,0.50,0.55 --output sweep.csv
"""
import argparse
import csv
import os
from pathlib import Path

# Isolate ultralytics' settings file from the shared workbench (SOLUTION.md §2).
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from tqdm import tqdm  # noqa: E402
from ultralytics import YOLO  # noqa: E402

FINETUNE_DIR = Path(__file__).resolve().parent
DATA_DIR = FINETUNE_DIR / "data"


def build_coco_results(detections):
    """Build a COCO results list. Mirrors the harness: every detection gets
    score=1.0 regardless of model confidence. `detections` is a list of
    (image_id, category_id, [x, y, w, h]) tuples."""
    return [
        {"image_id": int(img_id), "category_id": int(cat_id),
         "bbox": [float(v) for v in bbox], "score": 1.0}
        for img_id, cat_id, bbox in detections
    ]


def pick_best_conf(conf_to_map):
    """Return the conf with the highest mAP from a {conf: mAP} dict."""
    return max(conf_to_map, key=conf_to_map.get)


def parse_confs(value):
    """Parse a comma-separated confidence list into sorted floats."""
    confs = sorted({round(float(v.strip()), 4) for v in value.split(",") if v.strip()})
    if not confs:
        raise argparse.ArgumentTypeError("at least one confidence is required")
    return confs


def _evaluate(coco_gt, results):
    """Run COCOeval on a results list; return mAP@.5:.95 (stats[0])."""
    if not results:
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--confs",
        type=parse_confs,
        default=parse_confs("0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60"),
        help="Comma-separated confidence thresholds to evaluate.",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    args = parser.parse_args()

    coco_gt = COCO(str(DATA_DIR / "val_coco.json"))
    fname_to_id = {
        Path(img["file_name"]).name: img["id"]
        for img in coco_gt.dataset["images"]
    }
    img_dir = DATA_DIR / "val" / "images"
    model = YOLO(args.weights)

    # One inference pass at low conf; keep every box with its model confidence.
    raw = []  # (image_id, category_id, [x,y,w,h], model_conf)
    for fname, img_id in tqdm(fname_to_id.items(), desc="predict"):
        res = model.predict(str(img_dir / fname), imgsz=args.imgsz,
                            conf=0.001, iou=args.iou, verbose=False)[0]
        if res.boxes is None:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
            raw.append((img_id, int(c),
                        [x1, y1, x2 - x1, y2 - y1], float(p)))

    # Re-threshold + re-score (score=1.0) at each conf.
    conf_to_map = {}
    for thr in args.confs:
        kept = [(i, c, b) for (i, c, b, p) in raw if p >= thr]
        conf_to_map[float(thr)] = _evaluate(coco_gt, build_coco_results(kept))

    print("\nconf  mAP@.5:.95")
    for thr, m in sorted(conf_to_map.items()):
        print(f"{thr:.2f}  {m:.4f}")
    best = pick_best_conf(conf_to_map)
    print(f"\nBest conf = {best} (mAP {conf_to_map[best]:.4f})")
    print("Set DEFAULT_CONF in cv/src/cv_manager.py to this value.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["conf", "map_5095"])
            for thr, m in sorted(conf_to_map.items()):
                writer.writerow([f"{thr:.4f}", f"{m:.6f}"])
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
