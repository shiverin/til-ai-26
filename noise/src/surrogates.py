"""Surrogate detector ensemble for transfer-attack gradient signals."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
STRIDE = 32  # all surrogates need H, W divisible by 32


class SurrogateEnsemble:
    """Loads N detectors on GPU once. `loss(image)` returns the mean
    detection-loss scalar across all models, with gradient flowing to
    `image`.

    Supported model keys: "v8m" (fine-tuned), "yolo11n", "yolo26n", "rtdetr".
    """

    _UL_MAP = {
        "v8m": ("YOLO", "v8m.pt"),
        "yolo11n": ("YOLO", "yolo11n.pt"),
        "yolo26n": ("YOLO", "yolo26n.pt"),
        "rtdetr": ("RTDETR", "rtdetr-l.pt"),
    }

    def __init__(self, models: list[str] | None = None, device: str = "cuda"):
        if models is None:
            models = list(self._UL_MAP.keys())
        self.device = device
        self.models: dict[str, torch.nn.Module] = {}
        for key in models:
            if key not in self._UL_MAP:
                raise ValueError(f"unknown model key {key!r}")
            cls_name, filename = self._UL_MAP[key]
            from ultralytics import YOLO, RTDETR
            cls = {"YOLO": YOLO, "RTDETR": RTDETR}[cls_name]
            mdl = cls(str(WEIGHTS_DIR / filename)).model
            mdl.to(device).eval()
            for p in mdl.parameters():
                p.requires_grad_(False)
            self.models[key] = mdl
        # Convenience handle for the v8m head, used by ObjectnessMask.
        self.v8m = self.models.get("v8m")

    def loss(self, image: Tensor) -> Tensor:
        """Mean of per-model detection-confidence proxies. Returns 0-D
        tensor with grad wrt `image`. Image is `[B, 3, H, W]` in [0, 1].

        Image is right/bottom padded to a multiple of STRIDE so every
        surrogate's FPN scales align."""
        image = self._pad_to_stride(image)
        total = image.new_zeros(())
        for mdl in self.models.values():
            total = total + self._untargeted_loss(mdl, image)
        return total / max(1, len(self.models))

    @staticmethod
    def _pad_to_stride(image: Tensor) -> Tensor:
        H, W = image.shape[-2:]
        pad_h = (STRIDE - H % STRIDE) % STRIDE
        pad_w = (STRIDE - W % STRIDE) % STRIDE
        if pad_h == 0 and pad_w == 0:
            return image
        return F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    @staticmethod
    def _untargeted_loss(mdl: torch.nn.Module, image: Tensor) -> Tensor:
        """Log-amplified confidence proxy: -log(1 - max_conf) per anchor,
        summed over top-K anchors. Gradient ∝ 1/(1-p) explodes near
        high-confidence predictions, so each PGD step preferentially crushes
        the most confident detections instead of nibbling at borderline
        ones. Standard adversarial-detection loss."""
        TOPK = 20
        EPS = 1e-6
        out = mdl(image)
        if isinstance(out, tuple):
            decoded = out[0]
        elif isinstance(out, dict) and "pred_logits" in out:
            logits = out["pred_logits"]  # [B, N, nc]
            conf = logits.sigmoid().max(dim=-1).values
            scored = -torch.log(1.0 - conf.clamp(EPS, 1.0 - EPS))
            return scored.flatten().topk(min(TOPK, scored.numel())).values.sum()
        else:
            decoded = out
        if not isinstance(decoded, torch.Tensor) or decoded.dim() != 3:
            return image.new_zeros(())
        cls = decoded[:, 4:, :]  # [B, nc, A], already in [0, 1] (eval mode)
        max_per_anchor = cls.max(dim=1).values  # [B, A]
        scored = -torch.log(1.0 - max_per_anchor.clamp(EPS, 1.0 - EPS))
        return scored.flatten().topk(min(TOPK, scored.numel())).values.sum()
