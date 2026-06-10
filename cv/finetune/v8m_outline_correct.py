"""Run the soft-outline training pipeline from the real v8m ep1 checkpoint.

The earlier outline runs in ``cv_squeeze`` were seeded from
``cv_squeeze/src/weights/best.pt``, whose provenance later turned out to be
wrong. This runner reuses the same outline augmentation and masks, but starts
from ``cv/archive/v8m_ft_ep1.pt`` / ``src/weights/best_v8m_ep1.pt``.

Stages:
    3ep   Seed from the real v8m ep1 checkpoint.
    30ep  Seed from the corrected 3ep best checkpoint.
    all   Run 3ep, then 30ep.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

from ultralytics import YOLO  # noqa: E402


FINETUNE_DIR = Path(__file__).resolve().parent
REPO_CV = FINETUNE_DIR.parent
CV_SQUEEZE_FINETUNE = REPO_CV.parent / "cv_squeeze" / "finetune"
REAL_V8M_EP1 = REPO_CV / "archive" / "v8m_ft_ep1.pt"
REAL_V8M_EP1_SHA256 = "d10400fb8287364b8ba849cbf2a940213d038cb63ecb620fdf4b6140d81e8d58"
DATASET_YAML = FINETUNE_DIR / "data" / "dataset.yaml"
RUNS_DIR = FINETUNE_DIR / "runs"
RUN_3EP = "outline_correct_v8m_ep1_3ep"
RUN_30EP = "outline_correct_v8m_ep1_30ep"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_outline_aug() -> None:
    """Activate the existing cv_squeeze outline augmentation."""
    if str(CV_SQUEEZE_FINETUNE) not in sys.path:
        sys.path.insert(0, str(CV_SQUEEZE_FINETUNE))
    from train_noise_aug import patch_outline_aug as _patch_outline_aug

    _patch_outline_aug()


def train_outline(seed: Path, name: str, epochs: int, save_period: int,
                  patience: int) -> Path:
    if not seed.exists():
        raise FileNotFoundError(seed)
    patch_outline_aug()
    model = YOLO(str(seed))
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        save_period=save_period,
        imgsz=1280,
        batch=4,
        freeze=0,
        lr0=5e-4,
        lrf=0.01,
        cos_lr=True,
        optimizer="SGD",
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=1,
        patience=patience,
        amp=True,
        cache="disk",
        # Match the old outline pipeline: minimal base aug so the new signal is
        # outline softening, not mosaic/mixup/copy-paste noise.
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        scale=0.0,
        translate=0.0,
        degrees=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        seed=0,
        device=0,
        workers=8,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[outline-correct] best: {best}")
    print(
        "[outline-correct] mAP50-95:",
        results.results_dict.get("metrics/mAP50-95(B)", "N/A"),
    )
    return best


def run_3ep(name: str = RUN_3EP) -> Path:
    actual = sha256_file(REAL_V8M_EP1)
    if actual != REAL_V8M_EP1_SHA256:
        raise SystemExit(
            f"real v8m ep1 hash mismatch: got {actual}, expected {REAL_V8M_EP1_SHA256}"
        )
    return train_outline(REAL_V8M_EP1, name, epochs=3, save_period=-1, patience=10)


def run_30ep(seed: Path | None = None) -> Path:
    if seed is None:
        seed = RUNS_DIR / RUN_3EP / "weights" / "best.pt"
    return train_outline(seed, RUN_30EP, epochs=30, save_period=5, patience=15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["3ep", "30ep", "all"], default="all")
    parser.add_argument(
        "--name",
        default=RUN_3EP,
        help="Run name for --stage 3ep. Use a fresh name to avoid clobbering old runs.",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        help="Only for --stage 30ep: override the corrected 3ep best checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "3ep":
        run_3ep(args.name)
    elif args.stage == "30ep":
        run_30ep(args.seed)
    else:
        best_3ep = run_3ep(args.name)
        run_30ep(best_3ep)


if __name__ == "__main__":
    main()
