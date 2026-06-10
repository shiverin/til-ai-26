"""Manages the CV model."""

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

_WEIGHTS = Path(__file__).parent / "weights" / "best.pt"
_CONF = 0.25
_IOU = 0.45
_IMGSZ = 1280


class CVManager:

    def __init__(self):
        import torch
        self.half = torch.cuda.is_available()
        self.device = 0 if torch.cuda.is_available() else "cpu"
        if not _WEIGHTS.exists():
            print(f"WARNING: {_WEIGHTS} not found — returning empty predictions")
            self.model = None
            return
        self.model = YOLO(str(_WEIGHTS))
        self.model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            imgsz=640,
            verbose=False,
        )

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        img = Image.open(io.BytesIO(image)).convert("RGB")
        results = self.model.predict(
            img,
            imgsz=_IMGSZ,
            conf=_CONF,
            iou=_IOU,
            half=self.half,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # LTWH
                    "category_id": int(box.cls[0]),
                })
        return detections
