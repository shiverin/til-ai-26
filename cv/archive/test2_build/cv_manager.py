"""2-model test ensemble — M1 + M2, K=2 vote-merge.

One-off test to check if the nano direction is viable before waiting for
M3-M5 to finish training. Both models must agree (K=2) for a box to emit.
"""

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from ensemble import EnsembleMerger

_BASE = Path(__file__).parent
_WEIGHTS = [_BASE / "m1.pt", _BASE / "m2.pt"]

_CONF = 0.40          # patched at build time
_IMGSZ = 1280         # patched at build time
_IOU = 0.45
_RECT = False         # patched at build time
_K = 2                # patched at build time


class CVManager:
    def __init__(self):
        self.models = [YOLO(str(w)) for w in _WEIGHTS]
        dummy = np.zeros((_IMGSZ, _IMGSZ, 3), dtype=np.uint8)
        for model in self.models:
            model.predict(dummy, imgsz=_IMGSZ, conf=_CONF, iou=_IOU,
                          rect=_RECT, half=True, verbose=False)
        self.merger = EnsembleMerger(k=_K, iou_threshold=0.5)

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        img = Image.open(io.BytesIO(image)).convert("RGB")
        per_model: list[list[dict]] = []
        for model in self.models:
            results = model.predict(
                img, imgsz=_IMGSZ, conf=_CONF, iou=_IOU,
                rect=_RECT, half=True, verbose=False,
            )
            preds: list[dict] = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    preds.append({
                        "xyxy": (x1, y1, x2, y2),
                        "category_id": int(box.cls[0]),
                    })
            per_model.append(preds)

        merged = self.merger.merge(per_model)
        return [
            {"bbox": [m["xyxy"][0], m["xyxy"][1],
                      m["xyxy"][2] - m["xyxy"][0],
                      m["xyxy"][3] - m["xyxy"][1]],
             "category_id": m["category_id"]}
            for m in merged
        ]
