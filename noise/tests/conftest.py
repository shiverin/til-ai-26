"""Shared fixtures for noise tests."""
from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def synth_image_np() -> np.ndarray:
    """640x480 RGB image: gray background with a bright red square in the
    middle. Used to verify that object-aware components localize on the
    square."""
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    img[180:300, 260:380] = [220, 40, 40]  # bright red square
    return img


@pytest.fixture
def synth_image_bytes(synth_image_np) -> bytes:
    """The synthetic image encoded as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(synth_image_np).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synth_image_tensor(synth_image_np) -> torch.Tensor:
    """`[1, 3, H, W]` float tensor in [0, 1], on CPU. Tests that need GPU
    move it explicitly."""
    arr = synth_image_np.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


@pytest.fixture
def square_bbox():
    """Pixel coords of the bright square: (y1, y2, x1, x2)."""
    return (180, 300, 260, 380)
