"""Unit tests for the FGSM helpers in adv_train.py.

These cover the pure-function surface (eps schedule, perturbation math).
The DetectionModel.loss monkeypatch is covered by the smoke-run integration
test (`python finetune/train.py --adv-train --smoke`), not here."""
import torch

from adv_train import current_eps, fgsm_perturb


def test_current_eps_warmup_endpoints():
    # warmup=0 means "use target immediately"
    assert current_eps(0, 0, 0.05) == 0.05
    assert current_eps(99, 0, 0.05) == 0.05


def test_current_eps_linear_ramp():
    # warmup=4 epochs from 0 to 0.04
    assert current_eps(0, 4, 0.04) == 0.0
    assert current_eps(1, 4, 0.04) == 0.01
    assert current_eps(2, 4, 0.04) == 0.02
    assert current_eps(4, 4, 0.04) == 0.04
    # After warmup, stays at target.
    assert current_eps(10, 4, 0.04) == 0.04


def test_fgsm_perturb_bounded_by_eps_and_clipped():
    """|x' - x|_inf <= eps and x' is in [0, 1]."""
    img = torch.rand(2, 3, 16, 16, requires_grad=True)
    loss = (img**2).sum()
    eps = 0.1

    img_adv = fgsm_perturb(loss, img, eps)

    diff = (img_adv - img.detach()).abs().max().item()
    assert diff <= eps + 1e-6, f"max abs diff {diff} exceeded eps {eps}"
    assert img_adv.min().item() >= 0.0
    assert img_adv.max().item() <= 1.0


def test_fgsm_perturb_actually_perturbs():
    """The output differs from the input — it's not a no-op."""
    img = torch.rand(1, 3, 8, 8, requires_grad=True)
    loss = (img**2).sum()
    img_adv = fgsm_perturb(loss, img, eps=0.05)
    assert not torch.allclose(img_adv, img.detach())


def test_fgsm_perturb_returns_detached_tensor():
    """The perturbed image must be detached from the autograd graph so the
    next forward can require_grad on it cleanly without graph entanglement."""
    img = torch.rand(1, 3, 8, 8, requires_grad=True)
    loss = (img**2).sum()
    img_adv = fgsm_perturb(loss, img, eps=0.05)
    assert not img_adv.requires_grad


def test_fgsm_perturb_zero_eps_is_no_op():
    """eps=0 returns the input unchanged (after detach+clamp)."""
    img = torch.rand(1, 3, 8, 8, requires_grad=True).clamp(0, 1)
    loss = (img**2).sum()
    img_adv = fgsm_perturb(loss, img, eps=0.0)
    assert torch.allclose(img_adv, img.detach())
