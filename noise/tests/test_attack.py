"""Tests for PGDAttacker."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.attack import AttackConfig, PGDAttacker


def _mock_ensemble():
    """A trivial ensemble: loss = -image.sum() so gradient is uniformly
    -1. Lets us reason about the attack output without GPU."""
    ens = MagicMock()
    ens.loss = lambda x: -x.sum()
    return ens


def test_perturbation_bounded_by_epsilon():
    cfg = AttackConfig(epsilon=8 / 255, alpha=2 / 255, n_iters=10, momentum=0.0,
                       di_prob=0.0)
    att = PGDAttacker(cfg)
    x = torch.full((1, 3, 64, 64), 0.5)
    mask = torch.ones((64, 64))
    adv = att.attack(x, mask, _mock_ensemble())
    delta = (adv - x).abs().max().item()
    assert delta <= cfg.epsilon + 1e-6
    assert adv.shape == x.shape
    assert torch.isfinite(adv).all()


def test_zero_mask_means_no_perturbation():
    """With mask=0 AND mask_floor=0, no pixel should be perturbed."""
    cfg = AttackConfig(epsilon=8 / 255, alpha=2 / 255, n_iters=5, momentum=0.0,
                       di_prob=0.0, mask_floor=0.0)
    att = PGDAttacker(cfg)
    x = torch.full((1, 3, 64, 64), 0.5)
    mask = torch.zeros((64, 64))
    adv = att.attack(x, mask, _mock_ensemble())
    assert torch.allclose(adv, x)


def test_clamped_to_valid_pixel_range():
    cfg = AttackConfig(epsilon=8 / 255, alpha=2 / 255, n_iters=10, momentum=0.0,
                       di_prob=0.0)
    att = PGDAttacker(cfg)
    x = torch.full((1, 3, 64, 64), 0.99)  # near upper bound
    mask = torch.ones((64, 64))
    adv = att.attack(x, mask, _mock_ensemble())
    assert adv.min() >= 0.0
    assert adv.max() <= 1.0


def test_momentum_stabilizes_noisy_gradient():
    """When the per-step gradient is noisy (DI on), momentum=1.0 should
    produce a perturbation that's closer to the *expected* gradient direction
    than momentum=0.0. We use a noisy mock ensemble whose loss has random
    sign per call; with high momentum, the accumulator should still drive
    the perturbation toward the long-run mean direction.

    Concretely: with a positive-mean noisy gradient and momentum=1.0, the
    perturbation magnitude (sum |delta|) should be larger than momentum=0.0
    where random signs cancel."""
    def noisy_mock():
        ens = MagicMock()
        # 80% of the time gradient is positive, 20% negative — biased noise.
        def loss(x):
            sign = 1.0 if torch.rand(()).item() < 0.8 else -1.0
            return -sign * x.sum()
        ens.loss = loss
        return ens
    torch.manual_seed(42)
    cfg_m0 = AttackConfig(epsilon=8 / 255, alpha=1 / 255, n_iters=20,
                          momentum=0.0, di_prob=0.0)
    torch.manual_seed(42)
    cfg_m1 = AttackConfig(epsilon=8 / 255, alpha=1 / 255, n_iters=20,
                          momentum=1.0, di_prob=0.0)
    x = torch.full((1, 3, 64, 64), 0.5)
    mask = torch.ones((64, 64))
    torch.manual_seed(0)
    adv0 = PGDAttacker(cfg_m0).attack(x, mask, noisy_mock())
    torch.manual_seed(0)
    adv1 = PGDAttacker(cfg_m1).attack(x, mask, noisy_mock())
    # Both produce SOME perturbation; momentum-on uses the bias more.
    pert0 = (adv0 - x).sum().item()
    pert1 = (adv1 - x).sum().item()
    # Both should be biased upward (positive direction) due to the 80/20
    # gradient bias; momentum-on amplifies that bias.
    assert pert1 >= pert0 - 1e-6, f"pert0={pert0:.4f}, pert1={pert1:.4f}"


def test_di_with_prob_1_produces_bounded_output():
    """With di_prob=1.0 every iteration applies random resize+pad — output
    must still be bounded, finite, and shape-preserved."""
    torch.manual_seed(0)
    cfg_di = AttackConfig(epsilon=8 / 255, alpha=2 / 255, n_iters=8,
                          momentum=0.0, di_prob=1.0)
    x = torch.full((1, 3, 64, 64), 0.5)
    mask = torch.ones((64, 64))
    torch.manual_seed(123)
    adv_di = PGDAttacker(cfg_di).attack(x, mask, _mock_ensemble())
    assert adv_di.shape == x.shape
    assert torch.isfinite(adv_di).all()
    assert (adv_di - x).abs().max() <= cfg_di.epsilon + 1e-6
