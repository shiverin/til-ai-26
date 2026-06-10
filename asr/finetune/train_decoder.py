"""Decoder-only fine-tune on real + synthetic data — the pilot's Gate-B run.

Continues from the R1 model with the FastConformer encoder FROZEN: only the
prediction network + joint train. This is the encoder-frozen recipe that
produced R1's 0.980 — the full fine-tune that unfroze the encoder regressed
(0.980 -> 0.941) and is deliberately not repeated.

NEEDS GPU. Run only when the card is free.

Env vars:
  MAX_EPOCHS=N  -> epochs (default 4)
  BATCH_SIZE=N  -> micro-batch (default 8; lower if GPU memory is tight)
  SMOKE=1       -> short pipeline-check run
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
TRAIN_MANIFEST = f"{OUT_DIR}/pilot_train_manifest.json"   # real + synthetic
VAL_MANIFEST = f"{OUT_DIR}/val_manifest.json"             # 100% real
START_MODEL = "/models/parakeet_finetuned.nemo"           # R1 — the 0.980 model
CKPT_NAME = "pilot_decoder"        # distinct name — not the corrupt best.ckpt
SMOKE = os.environ.get("SMOKE") == "1"
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))


def main() -> None:
    model = nemo_asr.models.ASRModel.restore_from(START_MODEL)

    # Encoder frozen — only the decoder (prediction net + joint) adapts. The
    # decoder acts as the transducer's language model, so more exposure to the
    # fictional words raises their prior.
    model.encoder.freeze()

    model.setup_training_data(OmegaConf.create({
        "manifest_filepath": TRAIN_MANIFEST,
        "sample_rate": 16000,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "max_duration": 40.0,
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
        dirpath=OUT_DIR, filename=CKPT_NAME,
        monitor="val_wer", mode="min", save_top_k=1,
    )
    trainer_kwargs = dict(
        devices=1, accelerator="gpu", precision="16-mixed",
        accumulate_grad_batches=4, logger=False,
        num_sanity_val_steps=0, callbacks=[ckpt],
        enable_checkpointing=True,
    )
    if SMOKE:
        trainer_kwargs.update(
            max_epochs=1, limit_train_batches=20, limit_val_batches=8)
    else:
        trainer_kwargs.update(max_epochs=MAX_EPOCHS)
    trainer = pl.Trainer(**trainer_kwargs)

    model.set_trainer(trainer)
    # Decoder-only tolerates a higher LR than the encoder-unfrozen run (which
    # used 2e-5) — the frozen encoder cannot drift.
    model.setup_optimization(OmegaConf.create({
        "name": "adamw",
        "lr": 1e-4,
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "sched": {"name": "CosineAnnealing",
                  "warmup_steps": 500, "min_lr": 1e-6},
    }))

    # Auto-resume from a previous interrupted pilot run, if one exists.
    resume = f"{OUT_DIR}/{CKPT_NAME}.ckpt"
    resume = resume if os.path.exists(resume) else None
    if resume:
        print(f"RESUMING from {resume}", flush=True)
    trainer.fit(model, ckpt_path=resume)

    score = ckpt.best_model_score
    print("BEST CKPT:", ckpt.best_model_path,
          "VAL_WER:", float(score) if score is not None else None,
          flush=True)
    # ckpt_to_nemo.py converts the .ckpt to a deployable .nemo in a fresh
    # process. Force-exit to dodge the post-fit dataloader-shutdown hang.
    os._exit(0)


if __name__ == "__main__":
    main()
