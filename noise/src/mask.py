"""Soft objectness mask from the YOLOv8m surrogate's pre-NMS head."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

STRIDE = 32  # v8m needs H, W divisible by 32


class ObjectnessMask:
    """Reads pre-NMS confidence from a YOLO model, upsamples to image
    resolution, normalizes per image to [0, 1]. No detection or NMS step.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @torch.no_grad()
    def mask(self, image: Tensor) -> Tensor:
        """`image`: [B, 3, H, W] in [0, 1] on the model's device.
        Returns [H, W] (B=1) or [B, H, W] (B>1) in [0, 1].
        """
        H, W = image.shape[-2:]
        pad_h = (STRIDE - H % STRIDE) % STRIDE
        pad_w = (STRIDE - W % STRIDE) % STRIDE
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
        out = self.model(image)
        if isinstance(out, tuple):
            out = out[1]
        if not isinstance(out, (list, tuple)):
            out = [out]
        scale_maps = []
        for t in out:
            if not isinstance(t, torch.Tensor) or t.dim() != 4:
                continue
            if t.shape[1] <= 4:
                conf = t.sigmoid().max(dim=1, keepdim=True).values
            else:
                conf = t[:, 4:, ...].sigmoid().max(dim=1, keepdim=True).values
            conf = F.interpolate(conf, size=(H, W), mode="bilinear",
                                 align_corners=False)
            scale_maps.append(conf)
        if not scale_maps:
            return image.new_zeros((H, W))
        merged = torch.stack(scale_maps, dim=0).max(dim=0).values  # [B,1,Hp,Wp]
        merged = merged.squeeze(1)  # [B, Hp, Wp]
        # Crop padding back to original H, W.
        merged = merged[..., :H, :W]
        flat = merged.view(merged.shape[0], -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        norm = (flat - lo) / (hi - lo + 1e-8)
        norm = norm.view_as(merged)
        if norm.shape[0] == 1:
            return norm[0]
        return norm
