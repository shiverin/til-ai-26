from sweep_conf import build_coco_results, pick_best_conf


def test_build_coco_results_forces_score_one():
    dets = [(5, 3, [1.0, 2.0, 3.0, 4.0]), (5, 7, [0.0, 0.0, 9.0, 9.0])]
    results = build_coco_results(dets)
    assert all(r["score"] == 1.0 for r in results)
    assert results[0] == {
        "image_id": 5, "category_id": 3,
        "bbox": [1.0, 2.0, 3.0, 4.0], "score": 1.0,
    }


def test_pick_best_conf_returns_argmax():
    assert pick_best_conf({0.1: 0.50, 0.2: 0.71, 0.3: 0.64}) == 0.2


import numpy as np

from sweep_conf import rethreshold_and_renms


def _candidate(x1, y1, x2, y2, conf, cls):
    """Helper for building one candidate row used by rethreshold_and_renms."""
    return {"xyxy": [x1, y1, x2, y2], "conf": conf, "cls": cls}


def test_rethreshold_and_renms_drops_below_conf():
    cands = [
        _candidate(0, 0, 10, 10, 0.30, 0),
        _candidate(0, 0, 10, 10, 0.05, 0),
    ]
    kept = rethreshold_and_renms(cands, conf_thr=0.20, iou_thr=0.50)
    assert len(kept) == 1
    assert kept[0]["conf"] == 0.30


def test_rethreshold_and_renms_class_wise_nms_keeps_higher_conf():
    # Two overlapping boxes, same class, IoU > 0.5 -> only the higher-conf survives.
    cands = [
        _candidate(0, 0, 10, 10, 0.80, 1),
        _candidate(1, 1, 11, 11, 0.60, 1),
    ]
    kept = rethreshold_and_renms(cands, conf_thr=0.10, iou_thr=0.30)
    assert len(kept) == 1
    assert kept[0]["conf"] == 0.80


def test_rethreshold_and_renms_does_not_suppress_across_classes():
    # Two overlapping boxes, different classes -> class-wise NMS keeps both.
    cands = [
        _candidate(0, 0, 10, 10, 0.80, 1),
        _candidate(1, 1, 11, 11, 0.60, 2),
    ]
    kept = rethreshold_and_renms(cands, conf_thr=0.10, iou_thr=0.30)
    assert len(kept) == 2


import pytest
from pathlib import Path


@pytest.mark.gpu
def test_renms_matches_ultralytics_predict():
    """The whole sweep relies on: cached low-conf low-iou predict + in-memory
    re-NMS at (C, I) == fresh model.predict(conf=C, iou=I). Pin it on one image.
    """
    import os
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(Path(__file__).resolve().parent.parent / ".ultralytics"),
    )
    from ultralytics import YOLO

    weights = Path("/home/jupyter/til-ai-26/cv/src/weights/best.pt")
    if not weights.exists():
        pytest.skip(f"no weights at {weights}")
    val_dir = Path(
        "/home/jupyter/til-ai-26/cv/finetune/data/val/images"
    )
    images = sorted(val_dir.glob("*.jpg"))[:1]
    if not images:
        pytest.skip(f"no val images at {val_dir}")

    model = YOLO(str(weights))

    # Cache pass: low conf, loose NMS.
    res_loose = model.predict(
        str(images[0]), imgsz=1280, conf=0.001, iou=0.95, verbose=False,
    )[0]
    xyxy = res_loose.boxes.xyxy.cpu().numpy()
    conf = res_loose.boxes.conf.cpu().numpy()
    cls = res_loose.boxes.cls.cpu().numpy()
    cands = [
        {"xyxy": list(box), "conf": float(p), "cls": int(c)}
        for box, p, c in zip(xyxy, conf, cls)
    ]

    # Spot-check three (C, I) cells.
    for C, I in [(0.25, 0.45), (0.10, 0.65), (0.40, 0.70)]:
        ref = model.predict(
            str(images[0]), imgsz=1280, conf=C, iou=I, verbose=False,
        )[0]
        ref_n = 0 if ref.boxes is None else len(ref.boxes)
        ours = rethreshold_and_renms(cands, conf_thr=C, iou_thr=I)
        assert len(ours) == ref_n, (
            f"(C={C}, I={I}): re-NMS kept {len(ours)} vs predict kept {ref_n}"
        )
