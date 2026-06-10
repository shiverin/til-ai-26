"""Converts a training checkpoint (best.ckpt) into a deployable .nemo file.

Runs as a fresh process after training — no trainer, no dataloaders — so it
cannot hit the post-fit dataloader-shutdown hang.
"""

import os

import nemo.collections.asr as nemo_asr
import torch

# Paths are overridable via env vars so the same script can convert any epoch's
# checkpoint (e.g. CKPT=best_epoch0.ckpt OUT=epoch0.nemo).
ARCH_MODEL = os.environ.get("ARCH_MODEL", "/models/parakeet_finetuned.nemo")
CKPT = os.environ.get("CKPT", "/work/output/best.ckpt")
OUT = os.environ.get("OUT", "/work/output/parakeet_finetuned.nemo")


def main() -> None:
    model = nemo_asr.models.ASRModel.restore_from(ARCH_MODEL)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.save_to(OUT)
    print("SAVED:", OUT)


if __name__ == "__main__":
    main()
