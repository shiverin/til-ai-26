"""Loads the RF-DETR detector and serves batched inference for the TIL CV task.

Wraps `rfdetr.RFDETRBase` (or `RFDETRLarge` via env var) and converts its
`supervision.Detections` output into the TIL wire format: a list of dicts
`{bbox: [left, top, width, height], category_id: int}` per image, with bbox
coords as zero-indexed integer pixels clipped to image bounds.

Runtime knobs (env vars, all optional — set at container start, no rebuild
needed):
    CV_QH_VARIANT     "base" | "large" (default "base")
    CV_QH_CONF        score floor passed to model.predict (default 0.40 —
                      matches the YOLO baseline's TIL-tuned conf; raise if
                      false positives hurt under the score=1.0 harness, lower
                      if you want to give an mAP-style harness more recall)
    CV_QH_IMGSZ       input resolution at model construction (default unset →
                      RF-DETR's variant default, 560 for base / 728 for large).
                      Must satisfy RF-DETR's patch-stride constraint (multiple
                      of patch_size * num_windows = 64 for base). Common valid
                      values: 432, 560, 728, 1008, 1232.
"""
import io
import os
from pathlib import Path
from typing import Any

from PIL import Image

from src.bbox_utils import xyxy_to_ltwh_int

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

DEFAULT_CONF = 0.40
VARIANT = os.environ.get("CV_QH_VARIANT", "base")
CONF = float(os.environ.get("CV_QH_CONF", DEFAULT_CONF))
_IMGSZ_RAW = os.environ.get("CV_QH_IMGSZ", "").strip()
IMGSZ = int(_IMGSZ_RAW) if _IMGSZ_RAW else None


class CVManager:
    """Holds a loaded RF-DETR model and exposes a batch-inference contract."""

    def __init__(self, model, conf: float = DEFAULT_CONF):
        self.model = model
        self.conf = conf

    @classmethod
    def from_weights(cls, weights_dir: Path = WEIGHTS_DIR, conf: float = CONF,
                     variant: str = VARIANT, imgsz: int | None = IMGSZ) -> "CVManager":
        """Load best.pt from `weights_dir` and warm the model up."""
        from rfdetr import RFDETRBase, RFDETRLarge

        weights = Path(weights_dir) / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(f"no best.pt in {weights_dir}")
        init_kwargs: dict = {"pretrain_weights": str(weights)}
        if imgsz is not None:
            init_kwargs["resolution"] = imgsz
        print(f"[cv] loading RF-DETR-{variant} weights from {weights} "
              f"(conf={conf}, imgsz={imgsz or 'default'})")
        model_cls = {"base": RFDETRBase, "large": RFDETRLarge}[variant]
        model = model_cls(**init_kwargs)
        mgr = cls(model, conf=conf)
        mgr._warmup()
        return mgr

    def _warmup(self) -> None:
        """One dummy inference to pay CUDA/CuDNN autotune up front."""
        dummy = Image.new("RGB", (1280, 720), (0, 0, 0))
        self.model.predict(dummy, threshold=self.conf)

    @staticmethod
    def _decode(image: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image)).convert("RGB")

    def cv_batch(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        """Detect on a batch of JPEG byte blobs.

        Returns one prediction list per input image, in input order. A blob
        that fails to decode yields an empty list.
        """
        out: list[list[dict[str, Any]]] = []
        for i, blob in enumerate(images):
            try:
                img = self._decode(blob)
            except Exception as e:  # noqa: BLE001
                print(f"[cv] decode failed for image {i}: {e}")
                out.append([])
                continue
            w, h = img.size
            dets = self.model.predict(img, threshold=self.conf)
            out.append(self._format(dets, w, h))
        return out

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Single-image convenience wrapper."""
        return self.cv_batch([image])[0]

    @staticmethod
    def _format(dets, img_w: int, img_h: int) -> list[dict[str, Any]]:
        """Convert RF-DETR Detections → TIL output contract.

        RF-DETR emits 1-indexed class IDs (DETR class 0 = "no object"); the
        source dataset and TIL grader use 0-indexed IDs. Shift by -1 to align.
        Degenerate boxes (clipped to zero width/height) are dropped.
        """
        if len(dets) == 0:
            return []
        xyxy = dets.xyxy
        cls_ids = dets.class_id
        records = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            bbox = xyxy_to_ltwh_int(x1, y1, x2, y2, img_w, img_h)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            records.append({
                "bbox": bbox,
                "category_id": int(cls_ids[i]) - 1,
            })
        return records
