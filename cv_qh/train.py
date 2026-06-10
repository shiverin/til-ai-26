"""Fine-tune RF-DETR (base or large) on the TIL Novice CV dataset.

Examples:
    python train.py --variant base --epochs 30
    python train.py --variant large --epochs 30 --batch 4
    python train.py --variant base --epochs 1 --smoke   # 1 epoch sanity run

Per-epoch checkpoints are saved in weights/<variant>/ by RF-DETR. Pick winners
by TIL submission, NOT by local val mAP (anti-proxy — see project memory).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr import RFDETRBase, RFDETRLarge

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
WEIGHTS_ROOT = ROOT / "weights"

VARIANTS = {
    "base": RFDETRBase,
    "large": RFDETRLarge,
}

DEFAULT_BATCH = {"base": 8, "large": 4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=None,
                        help="Per-step batch (overrides per-variant default).")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps for an effective larger batch.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=None,
                        help="Input resolution; None = RF-DETR variant default.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch, batch=2, for a sanity check.")
    args = parser.parse_args()

    if not (DATASET / "train" / "_annotations.coco.json").exists():
        raise FileNotFoundError(
            "dataset/train/_annotations.coco.json missing; run prepare_data.py first."
        )

    output_dir = WEIGHTS_ROOT / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = 1 if args.smoke else args.epochs
    batch = 2 if args.smoke else (args.batch or DEFAULT_BATCH[args.variant])

    print(f"Training RF-DETR-{args.variant} for {epochs} epochs, batch={batch}, "
          f"grad_accum={args.grad_accum}, lr={args.lr}, resolution={args.resolution}")
    print(f"Dataset: {DATASET}")
    print(f"Output:  {output_dir}")
    print("REMINDER: local val mAP is an anti-proxy for TIL — submit checkpoints, "
          "don't pick by val.")

    model_cls = VARIANTS[args.variant]
    init_kwargs: dict = {}
    if args.resolution is not None:
        init_kwargs["resolution"] = args.resolution
    model = model_cls(**init_kwargs)

    train_kwargs: dict = dict(
        dataset_dir=str(DATASET),
        epochs=epochs,
        batch_size=batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=str(output_dir),
        num_workers=args.num_workers,
        checkpoint_interval=1,  # save every epoch so we can submit early ones
    )
    model.train(**train_kwargs)
    print(f"Done. Checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
