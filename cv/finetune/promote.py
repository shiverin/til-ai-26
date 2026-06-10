"""Pull a W&B artifact's best.pt into cv/src/weights/ for the next image build.

The training run uploads `best.pt` as the `cv-best` artifact in
`<project>/cv-best:<version>` form. Tag the winning version `production` in
the W&B UI, then run this script to deploy that version locally.

Usage:
    python promote.py                    # pulls cv-best:production (default)
    python promote.py --alias latest     # most recent run
    python promote.py --alias v3         # a specific version
"""
import argparse
import os
import shutil
from pathlib import Path

# Auto-load cv/.env if present (WANDB_API_KEY, WANDB_PROJECT, ...).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

FINETUNE_DIR = Path(__file__).resolve().parent
SRC_WEIGHTS = FINETUNE_DIR.parent / "src" / "weights"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "til26-cv"),
    )
    parser.add_argument("--artifact", default="cv-best")
    parser.add_argument(
        "--alias", default="production",
        help="W&B artifact alias to pull (production | latest | vN)",
    )
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not set; cannot pull from W&B.")

    import wandb

    ref = f"{args.project}/{args.artifact}:{args.alias}"
    print(f"[promote] pulling {ref} ...")
    art = wandb.Api().artifact(ref)
    download_dir = Path(art.download())
    src = download_dir / "best.pt"
    if not src.exists():
        raise SystemExit(f"artifact has no best.pt: {download_dir}")

    SRC_WEIGHTS.mkdir(parents=True, exist_ok=True)
    dst = SRC_WEIGHTS / "best.pt"
    shutil.copy(src, dst)
    print(f"[promote] wrote {dst}")
    if art.metadata:
        print(f"[promote] metadata: {art.metadata}")


if __name__ == "__main__":
    main()
