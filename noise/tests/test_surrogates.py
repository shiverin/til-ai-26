"""Tests for SurrogateEnsemble."""
from __future__ import annotations

import pytest
import torch

from src.surrogates import SurrogateEnsemble

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs cuda")
def test_v8m_only_loss_is_finite_with_grad(synth_image_tensor):
    """A single-model ensemble (v8m only) returns a finite scalar that
    backpropagates to the input."""
    ens = SurrogateEnsemble(models=["v8m"])
    x = synth_image_tensor.cuda().requires_grad_()
    loss = ens.loss(x)
    assert torch.is_tensor(loss) and loss.dim() == 0
    assert torch.isfinite(loss).all()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs() > 0).any()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs cuda")
def test_full_ensemble_loads_all_four(synth_image_tensor):
    ens = SurrogateEnsemble()
    assert set(ens.models.keys()) == {"v8m", "yolo11n", "yolo26n", "rtdetr"}
    x = synth_image_tensor.cuda().requires_grad_()
    loss = ens.loss(x)
    loss.backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs() > 0).any()
