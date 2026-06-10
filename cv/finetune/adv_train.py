"""Single-step FGSM adversarial training for ultralytics YOLO.

Monkeypatches DetectionModel.loss so that, when enabled, each training batch's
input is perturbed by `ε · sign(∇_x L)` before the actual training loss is
computed. This forces the model off non-robust features (matte halos in our
composite training data, per Grad-CAM) and onto features that survive small
input perturbations.

Reference: Tsipras et al. 2019, "Robustness May Be at Odds with Accuracy."
See cv/SOLUTION.md §11 for the design rationale.

Usage from train.py:

    from adv_train import patch_adversarial, unpatch_adversarial, update_epoch
    patch_adversarial(target_eps=4/255, warmup_epochs=2)
    model.add_callback("on_train_epoch_start", lambda t: update_epoch(t.epoch))
    model.train(...)
    unpatch_adversarial()
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class _AdvState:
    """Module-level state for the active adversarial regime."""
    enabled: bool = False
    target_eps: float = 0.0
    warmup_epochs: int = 0
    current_epoch: int = 0
    _orig_loss: Optional[Callable] = None


_state = _AdvState()


def current_eps(epoch: int, warmup: int, target: float) -> float:
    """Linear warmup from 0 to `target` over the first `warmup` epochs.

    epoch=0          -> 0
    epoch=warmup     -> target
    epoch>warmup     -> target
    warmup<=0        -> target  (no warmup)
    """
    if warmup <= 0:
        return target
    if epoch >= warmup:
        return target
    return target * epoch / warmup


def fgsm_perturb(loss: torch.Tensor, img: torch.Tensor, eps: float) -> torch.Tensor:
    """One FGSM step: x' = clip(x + ε · sign(∇_x L), 0, 1).

    `loss` must be a scalar tensor backed by a graph that flows through `img`,
    and `img.requires_grad` must be True. Returns a detached perturbed image
    bounded by ε from the input (in L∞) and clipped to the valid [0, 1] image
    range."""
    grad = torch.autograd.grad(loss, img, retain_graph=False)[0]
    return (img.detach() + eps * grad.sign()).clamp(0, 1)


def update_epoch(epoch: int) -> None:
    """Set the current training epoch (drives the warmup schedule).
    Call from an ultralytics `on_train_epoch_start` callback."""
    _state.current_epoch = epoch


def patch_adversarial(target_eps: float, warmup_epochs: int = 0) -> None:
    """Activate FGSM adversarial training by wrapping DetectionModel.loss.

    Idempotent: a second call is a no-op. Pair with `unpatch_adversarial()`."""
    if _state.enabled:
        return

    from ultralytics.nn.tasks import DetectionModel

    _state.enabled = True
    _state.target_eps = float(target_eps)
    _state.warmup_epochs = int(warmup_epochs)
    _state.current_epoch = 0
    _state._orig_loss = DetectionModel.loss

    orig = _state._orig_loss

    def adv_loss(self, batch, preds=None):
        eps = current_eps(_state.current_epoch, _state.warmup_epochs,
                          _state.target_eps)
        # Skip the adversarial dance when:
        #  - eps is 0 (warmup not started, or feature off)
        #  - ultralytics passed preds (already did a forward)
        #  - we're in eval mode (validation pass)
        if eps <= 0.0 or preds is not None or not self.training:
            return orig(self, batch, preds)

        # 1. Clean forward — get gradient of loss w.r.t. input image.
        img = batch["img"].detach().clone().requires_grad_(True)
        batch_clean = {**batch, "img": img}
        clean_loss, _ = orig(self, batch_clean, None)
        loss_scalar = clean_loss.sum() if clean_loss.dim() > 0 else clean_loss

        # 2. FGSM perturbation, then forward again — this loss drives the
        #    optimizer step.
        img_adv = fgsm_perturb(loss_scalar, img, eps)
        batch_adv = {**batch, "img": img_adv}
        return orig(self, batch_adv, None)

    DetectionModel.loss = adv_loss
    print(
        f"[adv] FGSM adversarial training enabled: "
        f"eps={target_eps:.4f}, warmup={warmup_epochs} epoch(s)"
    )


def unpatch_adversarial() -> None:
    """Restore the original DetectionModel.loss. Idempotent."""
    if not _state.enabled:
        return
    from ultralytics.nn.tasks import DetectionModel

    DetectionModel.loss = _state._orig_loss
    _state.enabled = False
    print("[adv] FGSM adversarial training disabled; original loss restored")
