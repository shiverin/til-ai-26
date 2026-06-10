"""Fine-tune YOLO11m on the novice CV dataset (two-stage, scripted).

Stage 1: freeze the backbone, train neck + head.
Stage 2: unfreeze all, lower LR, cosine decay, early-stop on patience.

Usage:
    python train.py                       # default short run
    python train.py --smoke                # 1+1 epoch pipeline check (5% data)
    python train.py --optimizer AdamW --lr0 0.001 --lr0-stage2 0.001
    python train.py --no-weighted          # disable class-balanced sampler
    python train.py --noise-aug            # enable noise/blur augmentation

Monitoring is opt-in. Set WANDB_API_KEY (e.g. in cv/.env or your shell) and
training metrics + best.pt artifact upload to W&B; ultralytics auto-logs to
the active wandb run. Without the env var, training runs offline as normal.
"""
import argparse
import os
from pathlib import Path

import torch

# Auto-load cv/.env (if present) so WANDB_API_KEY etc. land in os.environ
# before anything reads them. Gracefully skipped if python-dotenv isn't
# installed (e.g. minimal envs).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Isolate ultralytics' settings file from the shared workbench (SOLUTION.md §2):
# other venvs' ultralytics versions would otherwise reset the global settings.
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / ".ultralytics"),
)

from ultralytics import YOLO  # noqa: E402
from ultralytics import settings as _ul_settings  # noqa: E402

# Ultralytics writes its settings.json once, on first run, and bakes the
# wandb integration flag from whatever packages were installed at that
# moment. If wandb was added later, this flips the flag so per-epoch
# training metrics actually flow to wandb (not just the system metrics).
if os.environ.get("WANDB_API_KEY") and not _ul_settings.get("wandb"):
    _ul_settings.update({"wandb": True})

from weighted_dataset import patch_weighted_dataset  # noqa: E402

FINETUNE_DIR = Path(__file__).resolve().parent
REPO_CV = FINETUNE_DIR.parent
DATASET_YAML = FINETUNE_DIR / "data" / "dataset.yaml"
PRETRAINED = REPO_CV / "pretrained" / "yolo11m.pt"
RUNS_DIR = FINETUNE_DIR / "runs"

# Augmentation block — composite domain (SOLUTION.md §7).
# mixup + erasing pushed harder after Grad-CAM showed the model overfitting
# to matte-halo cues around composited objects; these intra-image
# perturbations break that shortcut so the model has to learn object
# features instead.
AUG = dict(
    mosaic=1.0, copy_paste=0.3, mixup=0.3, degrees=5.0,
    fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    erasing=0.6,
)


def patch_noise_aug():
    """Replace ultralytics' near-zero-prob albumentations with noise-calibrated
    transforms. Off by default — eval-set degradation is unverified
    (SOLUTION.md §7). Copied from archive/cv_og/train.py."""
    import albumentations as A
    from ultralytics.data.augment import Albumentations

    transforms = A.Compose([
        A.GaussNoise(std_range=(10 / 255, 50 / 255), p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), p=0.3),
        A.MotionBlur(blur_limit=9, p=0.2),
        A.ImageCompression(quality_range=(50, 95), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
    ])
    orig_init = Albumentations.__init__

    def custom_init(self, p=1.0, **kwargs):
        orig_init(self, p=p, **kwargs)
        self.transform = transforms
        self.contains_spatial = False  # all transforms are pixel-level
        print("[finetune] noise augmentation active")

    Albumentations.__init__ = custom_init


def _maybe_wandb_init(args):
    """Start a W&B run if WANDB_API_KEY is set; return None otherwise.

    Project name comes from WANDB_PROJECT (default "til26-cv"); entity comes
    from WANDB_ENTITY (read natively by wandb). Ultralytics auto-logs to
    whatever wandb run is active. The function returns *without* importing
    wandb when the key is absent, so offline runs are unaffected.

    Side effect: removes ultralytics' on_train_end callback so a single
    wandb run survives across both training stages. Without this patch,
    ultralytics calls wandb.run.finish() at the end of each model.train()
    call, which forces Stage 2 to auto-init a fresh run in ultralytics'
    default project (NOT ours). We finish the run ourselves at the end of
    main()."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb

    import ultralytics.utils.callbacks.wb as _ul_wb
    _ul_wb.callbacks.pop("on_train_end", None)

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "til26-cv"),
        config={
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lr0_stage2": args.lr0_stage2,
            "epochs_stage1": args.epochs_stage1,
            "epochs_stage2": args.epochs_stage2,
            "patience": args.patience,
            "freeze": args.freeze,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "weighted_sampler": not args.no_weighted,
            "noise_aug": args.noise_aug,
            "smoke": args.smoke,
        },
        save_code=True,
    )


def _maybe_log_artifact(run, best_pt, val_map):
    """Upload best.pt as a versioned 'cv-best' artifact with the val mAP in
    metadata. No-op if there is no run or no weights file."""
    if run is None or not Path(best_pt).exists():
        return
    import wandb

    art = wandb.Artifact(
        name="cv-best",
        type="model",
        metadata={"val_map_5095": float(val_map) if val_map is not None else None},
    )
    art.add_file(str(best_pt))
    run.log_artifact(art)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW"])
    p.add_argument("--lr0", type=float, default=0.01, help="stage-1 LR")
    p.add_argument("--lr0-stage2", type=float, default=0.005,
                   help="stage-2 initial LR (full finetune)")
    p.add_argument("--epochs-stage1", type=int, default=10)
    p.add_argument("--epochs-stage2", type=int, default=30)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--freeze", type=int, default=10,
                   help="layers frozen in stage 1 (0 = none)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4,
                   help="T4-safe default; Stage 1 alone can take --batch 8")
    p.add_argument("--no-weighted", action="store_true",
                   help="disable the class-balanced sampler")
    p.add_argument("--noise-aug", action="store_true",
                   help="enable noise/blur/compression augmentation")
    p.add_argument("--smoke", action="store_true",
                   help="1+1 epoch run on 5%% of the data to test the pipeline")
    # Adversarial training (Tier B; see SOLUTION.md §11 + adv_train.py)
    p.add_argument("--adv-train", action="store_true",
                   help="enable FGSM adversarial training during Stage 2")
    p.add_argument("--adv-eps", type=float, default=4 / 255,
                   help="FGSM perturbation magnitude in image space "
                        "(default 4/255 ~= 0.0157)")
    p.add_argument("--adv-warmup", type=int, default=2,
                   help="Stage-2 epochs to linearly ramp eps from 0 (default 2)")
    p.add_argument("--run-suffix", default="",
                   help="suffix appended to run dir names "
                        "(e.g. 'adv_only' → runs/stage1_adv_only, stage2_adv_only)")
    return p.parse_args()


def main():
    args = parse_args()
    wandb_run = _maybe_wandb_init(args)
    if not args.no_weighted:
        patch_weighted_dataset()
    if args.noise_aug:
        patch_noise_aug()

    e1, e2 = (1, 1) if args.smoke else (args.epochs_stage1, args.epochs_stage2)
    common = dict(
        data=str(DATASET_YAML), imgsz=args.imgsz, batch=args.batch,
        device=0, amp=True, workers=4, cache="disk", seed=0,
        optimizer=args.optimizer, cos_lr=True, lrf=0.01,
        fraction=0.05 if args.smoke else 1.0,
        project=str(RUNS_DIR), exist_ok=True, **AUG,
    )

    stage1_name = f"stage1_{args.run_suffix}" if args.run_suffix else "stage1"
    stage2_name = f"stage2_{args.run_suffix}" if args.run_suffix else "stage2"

    print(f"=== Stage 1 ({stage1_name}): freeze={args.freeze}, {e1} epoch(s) ===")
    model = YOLO(str(PRETRAINED))
    model.train(name=stage1_name, freeze=args.freeze, epochs=e1,
                lr0=args.lr0, warmup_epochs=3, **common)

    # Release Stage 1's GPU memory before Stage 2 (avoids OOM on T4 15 GB).
    try:
        del model
    finally:
        torch.cuda.empty_cache()

    print(f"=== Stage 2 ({stage2_name}): full finetune, {e2} epoch(s) ===")
    stage1_best = RUNS_DIR / stage1_name / "weights" / "best.pt"
    if not stage1_best.exists():
        last = RUNS_DIR / stage1_name / "weights" / "last.pt"
        print(f"[warn] {stage1_best.name} not found; falling back to {last.name}")
        stage1_best = last
    model = YOLO(str(stage1_best))

    # Adversarial training in Stage 2 only (Stage 1 stayed clean to warm the
    # head). FGSM doubles per-batch compute and adds graph memory, so drop
    # batch to 2 on T4 unless the user explicitly asked for less.
    stage2_common = dict(common)
    if args.adv_train:
        from adv_train import (patch_adversarial, unpatch_adversarial,
                               update_epoch)  # noqa: E402
        patch_adversarial(target_eps=args.adv_eps,
                          warmup_epochs=args.adv_warmup)
        model.add_callback("on_train_epoch_start",
                           lambda t: update_epoch(t.epoch))
        if stage2_common["batch"] > 2:
            print(f"[adv] reducing Stage 2 batch {stage2_common['batch']} -> 2 "
                  f"for FGSM VRAM headroom")
            stage2_common["batch"] = 2

    results = model.train(name=stage2_name, freeze=0, epochs=e2,
                          lr0=args.lr0_stage2, warmup_epochs=3,
                          patience=args.patience, close_mosaic=10,
                          **stage2_common)

    if args.adv_train:
        unpatch_adversarial()

    best = RUNS_DIR / stage2_name / "weights" / "best.pt"
    val_map = results.results_dict.get("metrics/mAP50-95(B)")
    print(f"\nDone. Best weights: {best}")
    print(f"val mAP50-95: {val_map}")
    _maybe_log_artifact(wandb_run, best, val_map)
    if wandb_run is not None:
        wandb_run.finish()
    print("Next: copy best.pt -> cv/src/weights/best.pt, then run sweep_conf.py")
    print("      (or: tag the W&B artifact 'production' + run finetune/promote.py)")


if __name__ == "__main__":
    main()
