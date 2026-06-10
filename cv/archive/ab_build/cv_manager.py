"""CV inference manager.

Inference config restored to the proven 0.718 setup. The ep1-mapfix experiment
(conf=0.01, iou=0.7, +score) scored 0.664 — TIL penalises false positives
directly (it does not score-rank), so conf is a real precision/recall knob and
its optimum is >= 0.25. _CONF is the single variable to tune from here.
"""

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

_WEIGHTS = Path(__file__).parent / "best.pt"
_CONF = 0.40          # conf-tuning test (vs 0.25 -> 0.718)
_IOU = 0.45
_IMGSZ = 1280         # patched at build time
_RECT = False         # patched at build time


class CVManager:
    def __init__(self):
        self.model = YOLO(str(_WEIGHTS))
        self.model.predict(
            np.zeros((_IMGSZ, _IMGSZ, 3), dtype=np.uint8),
            imgsz=_IMGSZ, conf=_CONF, iou=_IOU, rect=_RECT, half=True, verbose=False,
        )

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        img = Image.open(io.BytesIO(image)).convert("RGB")
        results = self.model.predict(
            img, imgsz=_IMGSZ, conf=_CONF, iou=_IOU, rect=_RECT, half=True, verbose=False,
        )
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": int(box.cls[0]),
                })
        return detections
