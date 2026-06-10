"""Manages the CV model: loads the detector and runs batched inference.

Serves best.pt directly (FP32 PyTorch on GPU). Diagnostic baseline — keeps
the same precision the sweep_conf.py grid was measured on, so the submission
score should land near the sweep peak if nothing else is wrong. ~30–50 ms/img
on a T4, fast enough for the harness; not as fast as TRT/ONNX but bulletproof.
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

# Inference knobs are env-var driven so one image can be re-tagged into many
# submission variants without code changes. Defaults match the 0.727 winner.
DEFAULT_CONF = float(os.environ.get("CV_CONF", "0.40"))
DEFAULT_IMGSZ = int(os.environ.get("CV_IMGSZ", "1536"))
DEFAULT_IOU = float(os.environ.get("CV_IOU", "0.45"))
TTA = os.environ.get("CV_TTA", "0") not in ("0", "", "false", "False")
DENOISE = os.environ.get("CV_DENOISE", "none").lower()  # none|bilateral|nlmeans|jpeg<Q>|gauss<K>
SAHI = os.environ.get("CV_SAHI", "0") not in ("0", "", "false", "False")
SAHI_SLICE = int(os.environ.get("CV_SAHI_SLICE", "640"))
SAHI_OVERLAP = float(os.environ.get("CV_SAHI_OVERLAP", "0.2"))


class CVManager:
    def __init__(self, model, conf=DEFAULT_CONF, iou=DEFAULT_IOU, imgsz=DEFAULT_IMGSZ,
                 tta=TTA, denoise=DENOISE, sahi_model=None):
        """`model` is a loaded ultralytics model. Use `CVManager.from_weights()`
        for the normal path; pass a model directly for testing."""
        import torch

        self.model = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tta = tta
        self.denoise = denoise
        self.sahi_model = sahi_model
        self.half = torch.cuda.is_available()
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"[cv] conf={conf} iou={iou} imgsz={imgsz} tta={tta} "
              f"denoise={denoise} sahi={'on' if sahi_model else 'off'}")

    @classmethod
    def from_weights(cls, weights_dir=WEIGHTS_DIR, conf=DEFAULT_CONF,
                     iou=DEFAULT_IOU, imgsz=DEFAULT_IMGSZ):
        """Load best.pt directly. No export, no engine build."""
        from ultralytics import YOLO

        weights_dir = Path(weights_dir)
        pt = weights_dir / "best.pt"
        if not pt.exists():
            raise FileNotFoundError(f"no best.pt in {weights_dir}")
        print(f"[cv] loading {pt}")
        sahi_model = None
        if SAHI:
            from sahi import AutoDetectionModel
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(pt),
                confidence_threshold=conf,
                device="cuda:0",
            )
            print(f"[cv] SAHI enabled: slice={SAHI_SLICE} overlap={SAHI_OVERLAP}")
        mgr = cls(YOLO(str(pt)), conf=conf, iou=iou, imgsz=imgsz,
                  sahi_model=sahi_model)
        mgr._warmup()
        return mgr

    def _warmup(self):
        """One dummy prediction to pay CUDA/TRT autotune cost up front."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.imgsz, conf=self.conf,
                           iou=self.iou, device=self.device, verbose=False,
                           augment=self.tta)

    def _decode(self, image: bytes) -> np.ndarray:
        arr = np.array(Image.open(io.BytesIO(image)).convert("RGB"))
        if self.denoise == "bilateral":
            import cv2
            arr = cv2.bilateralFilter(arr, d=5, sigmaColor=35, sigmaSpace=35)
        elif self.denoise == "nlmeans":
            import cv2
            arr = cv2.fastNlMeansDenoisingColored(arr, None, 5, 5, 7, 21)
        elif self.denoise.startswith("jpeg"):
            # "jpeg85" -> quality=85. Probe whether matching eval JPEG quant
            # level normalizes any compression-noise distribution shift.
            q = int(self.denoise[4:]) if len(self.denoise) > 4 else 85
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=q)
            buf.seek(0)
            arr = np.array(Image.open(buf).convert("RGB"))
        elif self.denoise.startswith("gauss"):
            # "gauss5" -> 5x5 Gaussian blur with sigma auto from kernel.
            # Target: additive Gaussian noise from the Noise attacker.
            import cv2
            k = int(self.denoise[5:]) if len(self.denoise) > 5 else 5
            if k % 2 == 0:
                k += 1
            arr = cv2.GaussianBlur(arr, (k, k), 0)
        return arr

    def cv_batch(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        """Detect on a batch of JPEG byte blobs. Returns one prediction list
        per input image, in input order; a blob that fails to decode yields []."""
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
            if self.sahi_model is not None:
                from sahi.predict import get_sliced_prediction
                for idx, img in zip(ok_idx, decoded):
                    res = get_sliced_prediction(
                        img, self.sahi_model,
                        slice_height=SAHI_SLICE, slice_width=SAHI_SLICE,
                        overlap_height_ratio=SAHI_OVERLAP,
                        overlap_width_ratio=SAHI_OVERLAP,
                        verbose=0,
                    )
                    results_map[idx] = self._format_sahi(res, img.shape[:2])
            else:
                results = self.model.predict(
                    decoded, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                    half=self.half, device=self.device, verbose=False,
                    augment=self.tta,
                )
                for idx, res in zip(ok_idx, results):
                    results_map[idx] = self._format(res)
        return [results_map.get(i, []) for i in range(len(images))]

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Single-image convenience wrapper."""
        return self.cv_batch([image])[0]

    @staticmethod
    def _format(res) -> list[dict[str, Any]]:
        """Convert an ultralytics result to the cv/README.md output contract."""
        if res.boxes is None or len(res.boxes) == 0:
            return []
        h, w = res.orig_shape
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        return [
            {"bbox": xyxy_to_ltwh(x1, y1, x2, y2, w, h), "category_id": int(c)}
            for (x1, y1, x2, y2), c in zip(xyxy, cls)
        ]

    @staticmethod
    def _format_sahi(res, hw) -> list[dict[str, Any]]:
        """Convert a sahi PredictionResult to the cv/README.md output contract."""
        h, w = hw
        out = []
        for p in res.object_prediction_list:
            x1, y1, x2, y2 = p.bbox.to_xyxy()
            out.append({
                "bbox": xyxy_to_ltwh(x1, y1, x2, y2, w, h),
                "category_id": int(p.category.id),
            })
        return out
