"""Ensemble CV inference manager — loads 5 nano models, vote-merges via K=3.

Weights are flat at /workspace/m{1..5}.pt. Per-model conf comes from
/workspace/ensemble_conf.json. _IMGSZ, _RECT, _K are patched at build time.
"""

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from ensemble import EnsembleMerger

_BASE = Path(__file__).parent
_WEIGHTS = [_BASE / f"m{i}.pt" for i in range(1, 6)]
_CONF_PATH = _BASE / "ensemble_conf.json"

_IMGSZ = 1280         # patched at build time
_IOU = 0.45
_RECT = False         # patched at build time
_K = 3                # patched at build time


class CVManager:
    def __init__(self):
        with _CONF_PATH.open() as f:
            cfg = json.load(f)
        self.confs = [float(cfg[f"m{i}"]) for i in range(1, 6)]
        self.models = [YOLO(str(w)) for w in _WEIGHTS]
        # Warmup each model
        dummy = np.zeros((_IMGSZ, _IMGSZ, 3), dtype=np.uint8)
        for model, conf in zip(self.models, self.confs):
            model.predict(dummy, imgsz=_IMGSZ, conf=conf, iou=_IOU,
                          rect=_RECT, half=True, verbose=False)
        self.merger = EnsembleMerger(k=_K, iou_threshold=0.5)

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        img = Image.open(io.BytesIO(image)).convert("RGB")
        per_model: list[list[dict]] = []
        for model, conf in zip(self.models, self.confs):
            results = model.predict(
                img, imgsz=_IMGSZ, conf=conf, iou=_IOU,
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
