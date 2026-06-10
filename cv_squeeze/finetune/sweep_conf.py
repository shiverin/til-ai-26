"""Sweep the inference confidence threshold to maximize COCO mAP.

The TIL harness (test/test_cv.py) scores every detection with a hard-coded
score=1.0, so AP is decided purely by WHICH boxes are emitted. This script
predicts once at a low conf, then re-thresholds + re-scores to find the
mAP-maximising conf. Set the serving CVManager conf to the printed value.

Usage:
    python sweep_conf.py --weights runs/stage2/weights/best.pt
"""
import argparse
import json
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
import torch  # noqa: E402
from torchvision.ops import batched_nms  # noqa: E402

FINETUNE_DIR = Path(__file__).resolve().parent
DATA_DIR = FINETUNE_DIR / "data"


def rethreshold_and_renms(candidates, conf_thr, iou_thr):
    """Re-evaluate one (conf_thr, iou_thr) cell over cached candidates.

    `candidates` is a list of dicts with keys 'xyxy' (4 floats), 'conf' (float),
    'cls' (int). Filters by conf >= conf_thr, then runs class-wise NMS at
    iou_thr. Returns the surviving candidate dicts (subset of the input, in
    descending conf order — matches what ultralytics returns post-NMS).
    """
    kept = [c for c in candidates if c["conf"] >= conf_thr]
    if not kept:
        return []
    boxes = torch.tensor([c["xyxy"] for c in kept], dtype=torch.float32)
    scores = torch.tensor([c["conf"] for c in kept], dtype=torch.float32)
    cls = torch.tensor([c["cls"] for c in kept], dtype=torch.int64)
    keep_idx = batched_nms(boxes, scores, cls, iou_thr).tolist()
    return [kept[i] for i in keep_idx]


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


CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
IOU_GRID = [0.45, 0.55, 0.65, 0.70, 0.75]
IMGSZ_GRID = [1280, 1536]
RECT_GRID = [False, True]


def _eval_with_per_class(coco_gt, results):
    """Run COCOeval; return (mAP@.5:.95, per-class AP dict)."""
    if not results:
        return 0.0, {c["id"]: 0.0 for c in coco_gt.dataset["categories"]}
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    mAP = float(ev.stats[0])
    # ev.eval["precision"] shape: [T, R, K, A, M] -- T thresholds, K classes.
    # Average over T (IoU), R (recall), A=0 (all), M=-1 (maxDet=100) -> per-class AP.
    pr = ev.eval["precision"]
    per_cls = {}
    for k, cat in enumerate(coco_gt.dataset["categories"]):
        vals = pr[:, :, k, 0, -1]
        vals = vals[vals > -1]
        per_cls[cat["id"]] = float(vals.mean()) if vals.size else 0.0
    return mAP, per_cls


def _cache_candidates(model, fname_to_id, img_dir, imgsz, rect):
    """One forward pass over val at very loose settings. Returns
    {image_id: [candidate, ...]} where each candidate is the dict shape
    expected by `rethreshold_and_renms`, plus an extra 'image_id' key for
    convenience."""
    cache = {}
    for fname, img_id in tqdm(
        fname_to_id.items(), desc=f"predict imgsz={imgsz} rect={rect}"
    ):
        res = model.predict(
            str(img_dir / fname),
            imgsz=imgsz,
            rect=rect,
            conf=0.001,   # below any CONF_GRID value -- keep low-confidence candidates
            iou=0.95,     # above any IOU_GRID value -- keep overlapping candidates for re-NMS
            max_det=300,  # ultralytics default; avoids truncating before re-NMS
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            cache[img_id] = []
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        cache[img_id] = [
            {"image_id": int(img_id), "xyxy": list(box),
             "conf": float(p), "cls": int(c)}
            for box, p, c in zip(xyxy, conf, cls)
        ]
    return cache


def _candidates_to_results(kept):
    """Convert post-re-NMS candidates to COCO-results form with score=1.0."""
    # Relies on the assertion in main() that ultralytics cls == COCO category_id.
    dets = []
    for c in kept:
        x1, y1, x2, y2 = c["xyxy"]
        dets.append(
            (c["image_id"], c["cls"], [x1, y1, x2 - x1, y2 - y1])
        )
    return build_coco_results(dets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--out", default=str(FINETUNE_DIR / "runs/sweeps/v8m_ft_ep1_sweep.json"),
    )
    args = parser.parse_args()

    coco_gt = COCO(str(DATA_DIR / "val_coco.json"))
    # Assumption: ultralytics' 0-based `cls` index maps 1-to-1 to COCO category_id
    # in this dataset. Verified: val_coco.json uses contiguous 0..17 ids. If a future
    # dataset uses sparse or 1-based ids, build a cls_idx -> cat_id map here and use
    # it inside _candidates_to_results.
    cat_ids = sorted(c["id"] for c in coco_gt.dataset["categories"])
    assert cat_ids == list(range(len(cat_ids))), (
        f"val_coco.json category ids are not contiguous-from-zero ({cat_ids[:5]}...); "
        f"_candidates_to_results assumes ultralytics cls == category_id"
    )
    fname_to_id = {
        Path(img["file_name"]).name: img["id"]
        for img in coco_gt.dataset["images"]
    }
    img_dir = DATA_DIR / "val" / "images"
    model = YOLO(args.weights)

    grid = []  # list of {"imgsz","rect","conf","iou","mAP","per_class_AP"}
    for imgsz in IMGSZ_GRID:
        for rect in RECT_GRID:
            cache = _cache_candidates(model, fname_to_id, img_dir, imgsz, rect)
            for conf_thr in CONF_GRID:
                for iou_thr in IOU_GRID:
                    kept_all = []
                    for cands in cache.values():
                        kept_all.extend(
                            rethreshold_and_renms(cands, conf_thr, iou_thr)
                        )
                    results = _candidates_to_results(kept_all)
                    mAP, per_cls = _eval_with_per_class(coco_gt, results)
                    grid.append({
                        "imgsz": imgsz, "rect": rect,
                        "conf": float(conf_thr), "iou": float(iou_thr),
                        "mAP": mAP, "per_class_AP": per_cls,
                    })
                    print(
                        f"imgsz={imgsz} rect={rect} "
                        f"conf={conf_thr:.2f} iou={iou_thr:.2f} "
                        f"mAP={mAP:.4f}"
                    )

    # Winner: highest mAP; tie-break -> lowest imgsz, then rect=True.
    grid_sorted = sorted(
        grid,
        key=lambda r: (-r["mAP"], r["imgsz"], not r["rect"]),
    )
    winner = grid_sorted[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "weights": str(Path(args.weights).resolve()),
        "val_coco": str((DATA_DIR / "val_coco.json").resolve()),
        "image_ids": sorted(fname_to_id.values()),
        "grid": grid,
        "winner": winner,
    }, indent=2))

    print("\n=== WINNER ===")
    print(json.dumps(winner, indent=2))
    print(f"\nWrote {out_path}")
    print(
        "Paste these into src/cv_manager.py:\n"
        f"  CONF  = {winner['conf']}\n"
        f"  IOU   = {winner['iou']}\n"
        f"  IMGSZ = {winner['imgsz']}\n"
        f"  RECT  = {winner['rect']}"
    )


if __name__ == "__main__":
    main()
