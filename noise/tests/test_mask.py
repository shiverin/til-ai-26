"""Tests for ObjectnessMask."""
from __future__ import annotations

import pytest
import torch

from src.mask import ObjectnessMask
from src.surrogates import SurrogateEnsemble

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs cuda")
def test_mask_shape_and_range(synth_image_tensor):
    ens = SurrogateEnsemble(models=["v8m"])
    mm = ObjectnessMask(ens.v8m)
    x = synth_image_tensor.cuda()
    mask = mm.mask(x)
    assert mask.shape == (x.shape[-2], x.shape[-1])
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs cuda")
def test_mask_localizes_on_object(synth_image_tensor, square_bbox):
    """Mean mask value inside the bright square should be > outside."""
    ens = SurrogateEnsemble(models=["v8m"])
    mm = ObjectnessMask(ens.v8m)
    x = synth_image_tensor.cuda()
    mask = mm.mask(x)
    y1, y2, x1, x2 = square_bbox
    inside = mask[y1:y2, x1:x2].mean().item()
    outside_sum = mask.sum().item() - mask[y1:y2, x1:x2].sum().item()
    outside_area = mask.numel() - (y2 - y1) * (x2 - x1)
    outside = outside_sum / outside_area
    assert inside > outside, f"inside={inside:.3f}, outside={outside:.3f}"
