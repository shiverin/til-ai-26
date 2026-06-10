"""Full fine-tune continuing from the Round 1 checkpoint.

Starts from the encoder-frozen Round 1 model (.nemo at /models) and
continues training with ALL weights unfrozen, so the FastConformer encoder
can adapt acoustically to the noisy audio. Memory fits the T4's 15 GB via
batch size 1 + gradient accumulation.

Env vars:
  SMOKE=1      -> short pipeline-check run (20 steps)
  MAX_EPOCHS=N -> number of epochs for the full run (default 6)
  BATCH_SIZE=N -> per-step micro-batch size (default 1)
"""

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:  # older NeMo
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

OUT_DIR = "/work/output"
TRAIN_MANIFEST = f"{OUT_DIR}/train_manifest.json"
VAL_MANIFEST = f"{OUT_DIR}/val_manifest.json"
START_MODEL = "/models/parakeet_finetuned.nemo"  # Round 1 checkpoint
SMOKE = os.environ.get("SMOKE") == "1"
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "6"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))


def main() -> None:
    model = nemo_asr.models.ASRModel.restore_from(START_MODEL)

    # Full fine-tune: all weights (encoder included) train.

    model.setup_training_data(OmegaConf.create({
        "manifest_filepath": TRAIN_MANIFEST,
        "sample_rate": 16000,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "max_duration": 32.0,
        "min_duration": 0.1,
    }))
    model.setup_validation_data(OmegaConf.create({
        "manifest_filepath": VAL_MANIFEST,
        "sample_rate": 16000,
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 4,
    }))

    ckpt = ModelCheckpoint(
        dirpath=OUT_DIR, filename="best",
        monitor="val_wer", mode="min", save_top_k=1,
    )
    trainer_kwargs = dict(
        devices=1, accelerator="gpu", precision="16-mixed",
        accumulate_grad_batches=16, logger=False,
        num_sanity_val_steps=0, callbacks=[ckpt],
        enable_checkpointing=True,
    )
    if SMOKE:
        # One tiny epoch so end-of-epoch validation runs and exercises the
        # val_wer checkpoint selection.
        trainer_kwargs.update(
            max_epochs=1, limit_train_batches=24, limit_val_batches=8
        )
    else:
        trainer_kwargs.update(max_epochs=MAX_EPOCHS)
    trainer = pl.Trainer(**trainer_kwargs)

    model.set_trainer(trainer)
    # Lower LR than the decoder-only run — the pretrained encoder is
    # sensitive and a high LR would cause catastrophic forgetting.
    model.setup_optimization(OmegaConf.create({
        "name": "adamw",
        "lr": 2e-5,
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "sched": {"name": "CosineAnnealing",
                  "warmup_steps": 500, "min_lr": 1e-6},
    }))

    # Auto-resume: if a checkpoint from a previous (interrupted) run exists,
    # continue from it instead of restarting. Makes the run resilient to the
    # workbench VM idle-shutdown killing the container.
    resume = f"{OUT_DIR}/best.ckpt"
    resume = resume if os.path.exists(resume) else None
    if resume:
        print(f"RESUMING from {resume}", flush=True)
    trainer.fit(model, ckpt_path=resume)

    best = ckpt.best_model_path
    score = ckpt.best_model_score
    print("BEST CKPT:", best,
          "VAL_WER:", float(score) if score is not None else None,
          flush=True)
    # ModelCheckpoint already wrote best.ckpt during training. The .ckpt is
    # converted to a deployable .nemo separately by ckpt_to_nemo.py (a fresh
    # process). Force-exit so the post-fit dataloader-shutdown hang seen with
    # the full fine-tune cannot stall the run.
    os._exit(0)


if __name__ == "__main__":
    main()
