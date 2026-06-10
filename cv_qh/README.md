# cv_qh — RF-DETR Novice CV harness

Experimentation harness to fine-tune RF-DETR on the TIL Novice CV dataset.
See `docs/superpowers/specs/2026-05-23-cv-qh-rfdetr-novice-design.md` for the
full design.

## Setup

```bash
cd cv_qh
uv venv
uv pip install -e ".[dev]"
```

## Run

```bash
# 1. Prepare data (idempotent)
python prepare_data.py

# 2. Train RF-DETR-Base (gate)
python train.py --variant base --epochs 30

# 3. Run inference on val
python infer.py --weights weights/base/checkpoint_best_ema.pth \
                --images dataset/valid \
                --out /tmp/preds.json

# 4. Local eval (anti-proxy! see warning)
python eval_local.py --weights weights/base/checkpoint_best_ema.pth
```

## Warning

Local val mAP@.5:.95 is an **anti-proxy** for the TIL grader's score — it ranks
models the opposite way on average. Submit multiple per-epoch checkpoints to
TIL and pick the winner there, not by `eval_local.py`.

See the `cv-qh-project` memory entry for the full story.
