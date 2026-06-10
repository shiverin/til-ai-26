import io

import numpy as np
from PIL import Image

from src.cv_manager import CVManager


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy, cls):
        self.xyxy = _FakeTensor(xyxy)
        self.cls = _FakeTensor(cls)

    def __len__(self):
        return len(self.cls.numpy())


class _FakeRes:
    """One detection carrying a distinct category id (used to check order)."""
    def __init__(self, cat_id):
        self.orig_shape = (1080, 1920)
        self.boxes = _FakeBoxes([[100.0, 50.0, 300.0, 250.0]], [cat_id])


class _EmptyRes:
    def __init__(self):
        self.orig_shape = (1080, 1920)
        self.boxes = None


class _FakeModel:
    """Returns one _FakeRes per input image, in order."""
    def __init__(self, cat_ids):
        self._cat_ids = cat_ids

    def predict(self, images, **kwargs):
        return [_FakeRes(c) for c in self._cat_ids[:len(images)]]


def _jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (123, 50, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_format_converts_xyxy_to_ltwh():
    out = CVManager._format(_FakeRes(cat_id=4))
    assert out == [{"bbox": [100.0, 50.0, 200.0, 200.0], "category_id": 4}]


def test_format_handles_empty_boxes():
    assert CVManager._format(_EmptyRes()) == []


def test_cv_batch_empty_input():
    mgr = CVManager(model=_FakeModel([]))
    assert mgr.cv_batch([]) == []


def test_cv_batch_preserves_order_and_tolerates_decode_failure():
    # Index 1 is not a valid image -> decode fails -> that slot must be [].
    blobs = [_jpeg_bytes(), b"not-an-image", _jpeg_bytes()]
    mgr = CVManager(model=_FakeModel(cat_ids=[7, 9]), conf=0.25)
    out = mgr.cv_batch(blobs)
    assert len(out) == 3
    assert out[0] == [{"bbox": [100.0, 50.0, 200.0, 200.0], "category_id": 7}]
    assert out[1] == []
    assert out[2] == [{"bbox": [100.0, 50.0, 200.0, 200.0], "category_id": 9}]


def test_default_constants():
    """Pins the inference constants. conf reverted from sweep winner 0.6 to
    0.25 after LB feedback (0.704 vs 0.718) — val plateau was dishonest;
    see cv_manager.py for the full story."""
    from src import cv_manager
    assert cv_manager.DEFAULT_CONF == 0.25
    assert cv_manager.DEFAULT_IOU == 0.45
    assert cv_manager.DEFAULT_IMGSZ == 1536
    assert cv_manager.DEFAULT_RECT is True


def test_from_weights_prefers_engine_over_pt(tmp_path, monkeypatch):
    """When both best.engine and best.pt exist, from_weights loads .engine."""
    from src import cv_manager
    (tmp_path / "best.pt").write_bytes(b"pt-bytes")
    (tmp_path / "best.engine").write_bytes(b"engine-bytes")
    loaded = {}

    class _Stub:
        def __init__(self, path):
            loaded["path"] = path
        def predict(self, *a, **kw):
            return []

    monkeypatch.setattr(cv_manager, "YOLO", _Stub)
    cv_manager.CVManager.from_weights(weights_dir=tmp_path)
    assert loaded["path"].endswith("best.engine")


def test_from_weights_falls_back_to_pt(tmp_path, monkeypatch):
    """When best.engine is absent, from_weights falls back to best.pt."""
    from src import cv_manager
    (tmp_path / "best.pt").write_bytes(b"pt-bytes")
    loaded = {}

    class _Stub:
        def __init__(self, path):
            loaded["path"] = path
        def predict(self, *a, **kw):
            return []

    monkeypatch.setattr(cv_manager, "YOLO", _Stub)
    cv_manager.CVManager.from_weights(weights_dir=tmp_path)
    assert loaded["path"].endswith("best.pt")
