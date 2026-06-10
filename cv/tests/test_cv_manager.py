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
    mgr = CVManager(model=_FakeModel([]), conf=0.25)
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
