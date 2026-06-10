"""Manages the adversarial-noise model."""
from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image


class NoiseManager:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Load the surrogate ensemble + mask + attacker once."""
        from src.surrogates import SurrogateEnsemble
        from src.mask import ObjectnessMask
        from src.attack import AttackConfig, PGDAttacker
        self.ensemble = SurrogateEnsemble(device=self.device)
        self.mask = ObjectnessMask(self.ensemble.v8m)
        self.attacker = PGDAttacker(AttackConfig())

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr.astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return t

    def _from_tensor(self, t: torch.Tensor) -> np.ndarray:
        arr = (t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0)
               .cpu().numpy() * 255).astype(np.uint8)
        return arr

    def _attack(self, arr: np.ndarray) -> np.ndarray:
        """Run mask + PGD on a single HWC uint8 image. Returns the
        perturbed HWC uint8 image, same shape."""
        x = self._to_tensor(arr)
        mask = self.mask.mask(x)
        adv = self.attacker.attack(x, mask, self.ensemble)
        return self._from_tensor(adv)

    def noise(self, image: bytes) -> str:
        """Returns base64-encoded PNG of the noised image, or base64 of
        the original bytes on any failure."""
        try:
            img = Image.open(io.BytesIO(image)).convert("RGB")
            arr = np.array(img)
            try:
                noised = self._attack(arr)
            except torch.cuda.OutOfMemoryError:
                print("[noise] OOM — returning original")
                return base64.b64encode(image).decode("ascii")
            buf = io.BytesIO()
            Image.fromarray(noised).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            print(f"[noise] error: {e} — returning original")
            return base64.b64encode(image).decode("ascii")
