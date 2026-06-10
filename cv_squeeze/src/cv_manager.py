"""Manages the CV model: loads the detector and runs batched inference.

Prefers a baked-in TensorRT engine (best.engine) over the PyTorch checkpoint
(best.pt). FP16 throughout. The (conf, iou, imgsz, rect) constants are
sweep-derived — see finetune/runs/sweeps/v8m_ft_ep1_sweep.json. Re-run the
sweep and update them after any retrain.
"""
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.bbox_utils import xyxy_to_ltwh

# Isolate ultralytics' settings file from the shared workbench (SOLUTION.md §2).
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# conf: sweep winner was 0.6 (val mAP=0.9795 on a flat plateau) but on the LB
# scored 0.704 vs 0.718 baseline at conf=0.25 — the val plateau was dishonest.
# Reverted to 0.25 (the original v8m-ft-ep1 image's value). Keep the rest of
# the sweep winner (imgsz/rect/iou) — TRT+rect cost zero score and bought
# +0.010 speed.
# See finetune/runs/sweeps/v8m_ft_ep1_sweep.json for the original sweep.
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 1536
DEFAULT_RECT = True

# Module-level import so tests can monkeypatch.
from ultralytics import YOLO  # noqa: E402


class CVManager:
    def __init__(self, model, conf=DEFAULT_CONF, iou=DEFAULT_IOU,
                 imgsz=DEFAULT_IMGSZ, rect=DEFAULT_RECT):
        import torch
        self.model = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.rect = rect
        self.half = torch.cuda.is_available()
        self.device = 0 if torch.cuda.is_available() else "cpu"
        # ultralytics' `rect=` kwarg in predict() is dataloader-only and is
        # silently ignored for single-image predict — a scalar imgsz with
        # rect=True still letterboxes to (imgsz, imgsz) square. The only way
        # to force the rect [h, w] shape (needed to match our TRT engine's
        # imgsz=[864, 1536] profile) is to pass imgsz as a 2-element list.
        if rect:
            h = round(imgsz * 9 / 16 / 32) * 32  # 16:9 source -> 864 for 1536
            self._predict_imgsz = [h, imgsz]
        else:
            self._predict_imgsz = imgsz

    @classmethod
    def from_weights(cls, weights_dir=WEIGHTS_DIR, conf=DEFAULT_CONF,
                     iou=DEFAULT_IOU, imgsz=DEFAULT_IMGSZ,
                     rect=DEFAULT_RECT):
        """Prefer best.engine (TRT, baked by build.sh) over best.pt."""
        weights_dir = Path(weights_dir)
        engine = weights_dir / "best.engine"
        pt = weights_dir / "best.pt"
        if engine.exists():
            chosen = engine
        elif pt.exists():
            chosen = pt
        else:
            raise FileNotFoundError(
                f"no best.engine or best.pt in {weights_dir}"
            )
        print(f"[cv] loading {chosen}")
        mgr = cls(YOLO(str(chosen)), conf=conf, iou=iou,
                  imgsz=imgsz, rect=rect)
        mgr._warmup()
        return mgr

    def _warmup(self):
        """One dummy predict to pay CUDA/TRT autotune cost up front.
        Dummy is 1080x1920 (the harness's actual frame shape)."""
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.model.predict(
            dummy, imgsz=self._predict_imgsz,
            conf=self.conf, iou=self.iou,
            half=self.half, device=self.device, verbose=False,
        )

    @staticmethod
    def _decode(image: bytes) -> np.ndarray:
        return np.array(Image.open(io.BytesIO(image)).convert("RGB"))

    def cv_batch(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        """Detect on a batch of JPEG byte blobs. One prediction list per
        input image, in input order; a blob that fails to decode yields []."""
        if not images:
            return []
        decoded, ok_idx = [], []
        for i, blob in enumerate(images):
            try:
                decoded.append(self._decode(blob))
                ok_idx.append(i)
            except Exception as e:  # noqa: BLE001
                print(f"[cv] decode failed for image {i}: {e}")
        results_map: dict[int, list] = {}
        if decoded:
            results = self.model.predict(
                decoded,
                imgsz=self._predict_imgsz,
                conf=self.conf, iou=self.iou,
                half=self.half, device=self.device, verbose=False,
            )
            for idx, res in zip(ok_idx, results):
                results_map[idx] = self._format(res)
        return [results_map.get(i, []) for i in range(len(images))]

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        return self.cv_batch([image])[0]

    @staticmethod
    def _format(res) -> list[dict[str, Any]]:
        if res.boxes is None or len(res.boxes) == 0:
            return []
        h, w = res.orig_shape
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        return [
            {"bbox": xyxy_to_ltwh(x1, y1, x2, y2, w, h), "category_id": int(c)}
            for (x1, y1, x2, y2), c in zip(xyxy, cls)
        ]
