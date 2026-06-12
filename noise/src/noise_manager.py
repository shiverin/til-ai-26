"""Manages the adversarial-noise model."""
from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
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

    def noise_batch(self, images_bytes: list[bytes]) -> list[str]:
        """Process a list of images as a single GPU batch.

        Images may have different sizes; all are zero-padded to the maximum
        H and W in the batch, attacked together, then cropped back.
        Falls back to sequential per-image processing on OOM or other errors.
        """
        if not images_bytes:
            return []
        try:
            arrays: list[np.ndarray] = []
            orig_sizes: list[tuple[int, int]] = []
            for img_bytes in images_bytes:
                arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                arrays.append(arr)
                orig_sizes.append((arr.shape[0], arr.shape[1]))

            max_H = max(s[0] for s in orig_sizes)
            max_W = max(s[1] for s in orig_sizes)

            # Pad each array to (max_H, max_W) before stacking into a batch.
            tensors = []
            for arr, (H, W) in zip(arrays, orig_sizes):
                if H < max_H or W < max_W:
                    padded = np.zeros((max_H, max_W, 3), dtype=np.uint8)
                    padded[:H, :W] = arr
                    arr = padded
                tensors.append(self._to_tensor(arr))  # [1, 3, max_H, max_W]

            x_batch = torch.cat(tensors, dim=0)               # [B, 3, max_H, max_W]
            mask_batch = self.mask.mask(x_batch)               # [B, max_H, max_W]
            adv_batch = self.attacker.attack(x_batch, mask_batch, self.ensemble)

            results = []
            for i, (orig_H, orig_W) in enumerate(orig_sizes):
                adv = adv_batch[i:i + 1, :, :orig_H, :orig_W]
                buf = io.BytesIO()
                Image.fromarray(self._from_tensor(adv)).save(buf, format="PNG")
                results.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            return results

        except torch.cuda.OutOfMemoryError:
            print("[noise] OOM in batch — falling back to sequential")
            return [self.noise(b) for b in images_bytes]
        except Exception as e:
            print(f"[noise] batch error: {e} — falling back to sequential")
            return [self.noise(b) for b in images_bytes]
