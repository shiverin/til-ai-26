"""End-to-end tests for NoiseManager."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from src.noise_manager import NoiseManager


def _mock_pipeline_returns_input(monkeypatch):
    """Patch the manager so attack returns input unchanged (CPU, no GPU)."""
    monkeypatch.setattr(
        "src.noise_manager.NoiseManager._build_pipeline",
        lambda self: None,
    )
    monkeypatch.setattr(
        "src.noise_manager.NoiseManager._attack",
        lambda self, arr: arr,
    )


def test_round_trip_returns_png_base64(monkeypatch, synth_image_bytes,
                                       synth_image_np):
    _mock_pipeline_returns_input(monkeypatch)
    mgr = NoiseManager()
    out_b64 = mgr.noise(synth_image_bytes)
    assert isinstance(out_b64, str)
    raw = base64.b64decode(out_b64)
    img = Image.open(io.BytesIO(raw))
    assert img.format == "PNG"
    arr = np.array(img.convert("RGB"))
    assert arr.shape == synth_image_np.shape


def test_decode_failure_returns_original_image(monkeypatch):
    _mock_pipeline_returns_input(monkeypatch)
    mgr = NoiseManager()
    bad = b"not an image"
    out_b64 = mgr.noise(bad)
    assert base64.b64decode(out_b64) == bad
