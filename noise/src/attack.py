"""Mask-weighted PGD attack with MI-FGSM + DI-FGSM."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AttackConfig:
    # 2026-06-09: gate failure mode is SSIM-inside (≥0.3); the mask
    # concentrates perturbation inside bboxes which collapses SSIM fast.
    # eps=40/255 + soft-floor mask gives gate margin while keeping
    # ~25-30% detection drop on held-out.
    # 2026-06-09: paired with log-trick loss. eps=45 + tiny floor=0.05
    # keeps attack strong (~20% drop on held-out) while pushing borderline
    # gate-fail images over the SSIM-inside threshold.
    epsilon: float = 45 / 255      # L_inf cap on perturbation
    alpha: float = 4.5 / 255       # step size (= epsilon / 10)
    n_iters: int = 30              # PGD iterations
    mask_floor: float = 0.05       # tiny mask floor for gate margin
    momentum: float = 1.0          # MI-FGSM decay (0 disables)
    di_prob: float = 0.5           # DI-FGSM application probability
    di_low: float = 0.9            # DI random-resize lower bound
    di_high: float = 1.0           # DI random-resize upper bound


class PGDAttacker:
    def __init__(self, cfg: AttackConfig | None = None):
        self.cfg = cfg or AttackConfig()

    def _input_diversity(self, x: Tensor) -> Tensor:
        """DI-FGSM: with prob di_prob, random-resize x to a smaller size
        and pad back to original. Returns x unchanged with prob 1 - di_prob.
        """
        cfg = self.cfg
        if torch.rand(()).item() > cfg.di_prob:
            return x
        _, _, H, W = x.shape
        scale = float(torch.empty(()).uniform_(cfg.di_low, cfg.di_high))
        Hs, Ws = max(1, int(H * scale)), max(1, int(W * scale))
        small = F.interpolate(x, size=(Hs, Ws), mode="bilinear",
                              align_corners=False)
        pad_h, pad_w = H - Hs, W - Ws
        top = int(torch.randint(0, pad_h + 1, ()).item()) if pad_h > 0 else 0
        left = int(torch.randint(0, pad_w + 1, ()).item()) if pad_w > 0 else 0
        out = F.pad(small, (left, pad_w - left, top, pad_h - top), value=0.0)
        return out

    def attack(self, image: Tensor, mask: Tensor, ensemble) -> Tensor:
        """`image`: [B, 3, H, W] in [0, 1]. `mask`: [H, W] or [B, H, W] in [0, 1].
        Returns the perturbed image, same shape, in [0, 1]."""
        cfg = self.cfg
        x = image.detach()
        delta = torch.zeros_like(x, requires_grad=True)
        if mask.dim() == 2:
            mask_b = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask_b = mask.unsqueeze(1)
        # Soft-floor the mask: background still gets some perturbation budget
        # to keep SSIM-inside under the gate (concentrating fully on objects
        # collapses SSIM-inside below 0.3).
        mask_b = cfg.mask_floor + (1.0 - cfg.mask_floor) * mask_b
        g = torch.zeros_like(x)  # momentum accumulator
        for _ in range(cfg.n_iters):
            x_adv = (x + delta).clamp(0.0, 1.0)
            x_adv_di = self._input_diversity(x_adv)
            loss = -ensemble.loss(x_adv_di)
            grad = torch.autograd.grad(loss, delta, retain_graph=False,
                                       create_graph=False)[0]
            # MI-FGSM: normalize by L1 norm, accumulate with decay.
            grad_norm = grad / (grad.abs().mean(dim=(1, 2, 3),
                                                keepdim=True) + 1e-12)
            g = cfg.momentum * g + grad_norm
            step = cfg.alpha * g.sign() * mask_b
            delta = (delta.detach() + step).clamp(-cfg.epsilon, cfg.epsilon)
            delta.requires_grad_(True)
        adv = (x + delta.detach()).clamp(0.0, 1.0)
        return adv
