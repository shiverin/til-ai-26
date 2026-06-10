# CV Solution — Scripted YOLO11m Finetuning Pipeline

**Status**: design, pre-implementation. **Companion doc**: [RESEARCH.md](RESEARCH.md)
(EDA, model landscape, scoring quirks). This file says *what we will build*;
RESEARCH.md says *why*.

**Scope**: a scripted PyTorch finetuning pipeline for the TIL-AI 2026 Novice CV
task (18-class object detection, COCO `mAP@.5:.05:.95`, offline T4 Docker). The
model is **YOLO11m** via `ultralytics`; the pipeline is a set of explicit
scripts — not a one-line `model.train()` — exposing every knob RESEARCH.md
flagged as a lever. Serving is accelerated with **TensorRT FP16**.

This is deliberately a *low-risk* plan: it stays inside the team's working
ultralytics stack (RESEARCH.md §3d), reuses online templates instead of
generating novel code, and fixes the structural problems that caused the AE
codebase to thrash.

---

## 1. Guiding decisions (locked)

| Decision | Choice | Source |
|---|---|---|
| Framework | Ultralytics YOLO, scripted (not a hand-written training loop) | user |
| Model | **YOLO11m** first (`pretrained/yolo11m.pt` already in repo) | user |
| Train budget | Short runs by default; escalate to overnight **only if** the val mAP curve proves it pays | user |
| Resolution | `imgsz=1280` train + val + infer | RESEARCH.md §3a |
| Serving | TensorRT FP16 engine, built in-container | user |
| Tiling (SAHI) | Rejected | RESEARCH.md §3c |

"Scripted from scratch" means `train.py` **orchestrates** the finetune (two
sequential `model.train()` calls, a monkeypatched sampler, explicit
hyperparameters). It does **not** mean re-implementing the autograd loop —
that would discard ultralytics' TAL label assigner and loss for no gain.

---

## 2. Repository layout & environment (anti-thrash)

The AE codebase thrashes because training and serving code interleave, imports
depend on the execution directory, and paths are hardcoded. The same smells
exist here: `cv_manager.py` carries a `try/except ImportError` because the
Dockerfile flattens `src/` into `/workspace/`, and the archived `train.py`
hardcodes `/home/jupyter/...`. The layout below removes the root causes.

```
cv/
  src/                      # SERVING ONLY — the only code baked into the image
    cv_manager.py
    cv_server.py
    bbox_utils.py
    weights/best.pt         # finetuned weights (source of truth in the image)
  finetune/                 # TRAINING ONLY — never enters the image
    prepare_dataset.py      # COCO -> YOLO conversion
    weighted_dataset.py     # class-balanced sampler (copied template)
    train.py                # two-stage scripted driver
    requirements-train.txt  # = -r ../requirements.txt + training extras
    data/                   # generated YOLO dataset      (gitignored)
    runs/                   # ultralytics run outputs      (gitignored)
  Dockerfile                # COPY src ./src  — copies src/ only
  requirements.txt          # serving deps only
  RESEARCH.md  SOLUTION.md
```

**Three rules that kill the thrash:**

1. **Training stays out of the image — structurally.** The Dockerfile copies
   *only* `src/`. `finetune/` is a sibling directory it never references, so
   training code/deps *cannot* leak into the image; it is not a discipline
   issue, it is impossible.

2. **Docker mirrors dev — identical import paths everywhere.** Dockerfile uses
   `WORKDIR /workspace`, `COPY src ./src`, and runs `uvicorn src.cv_server:app`.
   Local dev runs the *same* command from `cv/`. Imports are always
   `from src.bbox_utils import ...` in both places → **delete the
   `try/except ImportError`** in `cv_manager.py`. All script paths derive from
   `Path(__file__).resolve()`; no hardcoded absolute paths. The dataset source
   (`/home/jupyter/novice/cv`) is a single named constant / CLI arg at the top
   of `prepare_dataset.py`. Ultralytics' settings file is also pinned
   per-project via `YOLO_CONFIG_DIR` (a `cv/.ultralytics/` dir, set before any
   ultralytics import) so the shared global settings file on the workbench
   cannot leak state between projects.

3. **Dependencies are a superset/subset pair.** `cv/requirements.txt` holds
   serving deps only (what the image installs — stays slim).
   `finetune/requirements-train.txt` is `-r ../requirements.txt` plus training
   extras (`albumentations`, `tqdm`, `onnx`, `onnxslim`). The **dev `uv` venv
   installs the training file** — the superset — so every dependency is present
   while iterating, and no time is lost debugging missing imports. The image
   still builds from the slim `requirements.txt` only.

`uv` is the mandated package manager — `uv venv && uv pip install -r
finetune/requirements-train.txt` for the dev environment.

---

## 3. Pipeline components & templates

Nearly all code is copied/adapted from a known source — minimal novel code.

| File | Template / source | Notes |
|---|---|---|
| `prepare_dataset.py` | Reuse `archive/cv_og/prepare_dataset.py` | De-hardcode paths; fixed-seed split |
| `weighted_dataset.py` | [y-t-g `YOLOWeightedDataset`](https://y-t-g.github.io/tutorials/yolo-class-balancing/) | ~40-line copy; the only "new" file |
| `train.py` | `archive/cv_og/train.py` + [Ultralytics finetuning guide](https://docs.ultralytics.com/guides/finetuning-guide) | Two-stage driver |
| `cv_manager.py` (edit) | Existing file | Add TensorRT engine build + drop import hack |

---

## 4. Dataset preparation

- Convert `/home/jupyter/novice/cv/annotations.json` (COCO) → YOLO txt labels +
  `dataset.yaml` into `finetune/data/`.
- **Fixed-seed 85/15 train/val split** (reproducibility). RESEARCH.md §1a warns
  the val tail is thin (~20 `cruise ship` instances) — per-class AP there is
  noisy. **Open item**: consider a *stratified* split so every rare class is
  represented in val; start with the simple random split and revisit if tail AP
  is unreadable.
- 173 empty images (3.5%) are kept — the model must learn to emit nothing
  (RESEARCH.md §4d).

---

## 5. Training strategy — two-stage finetune

The domain is *similar* to COCO (car/truck/bus/boat overlap) but
composite-synthetic, so a two-stage finetune fits the
[Ultralytics finetuning guide](https://docs.ultralytics.com/guides/finetuning-guide):

- **Stage 1 — head adaptation.** `freeze=10` (backbone layers 0–9 frozen),
  train neck + detection head only. ~10 epochs, higher LR. The 18-class head
  adapts fast without disturbing pretrained features.
- **Stage 2 — full finetune.** Load Stage-1 `best.pt`, `freeze=0` (all
  trainable), lower LR with cosine decay, `patience`-gated early stop. Refines
  the backbone to the composite domain.

`train.py` runs both stages in sequence (Stage 2 starts from Stage 1's weights).

**Short-by-default, escalate on evidence.** The default is a ~40-epoch total
run. Whether to spend an overnight run is decided by the **`results.csv` val
mAP@.5:.95 curve**: if it is still rising at the `patience` cutoff (not
plateaued), epochs are increased; if it has flattened, it is not. `patience`
itself produces this signal automatically — no separate experiment needed.

**Freeze depth is a cheap sweep.** If Stage 2 disappoints, try `freeze ∈
{0, 5, 10}` and keep the best val mAP — the guide explicitly recommends
treating freeze depth as a tunable.

---

## 6. Hyperparameters

| Param | Stage 1 | Stage 2 | Rationale |
|---|---|---|---|
| `imgsz` | 1280 | 1280 | Dominant accuracy lever; ~100% small objects (RESEARCH.md §3a) |
| `epochs` | 10 | 30 | Short default; escalate on the mAP curve |
| `patience` | — | 15 | Early-stop + exposes the "still improving?" signal |
| `optimizer` | SGD | SGD | **Tunable lever** — see below; SGD is the default, not a silent one |
| `lr0` | 0.01 | 0.005 | Lower in Stage 2 to protect the unfrozen backbone |
| `lrf` | 0.01 | 0.01 | Cosine floor at 1% of `lr0` |
| `cos_lr` | True | True | Smooth decay |
| `momentum` / `weight_decay` | 0.937 / 5e-4 | 0.937 / 5e-4 | ultralytics finetune defaults |
| `warmup_epochs` | 3 | 3 | Protects pretrained weights (guide) |
| `freeze` | 10 | 0 | Two-stage strategy (§5) |
| `batch` | 4 | 4 | Stage-2 full finetune at batch 8 OOMs on T4 15 GB; 4 is the safe two-stage default |
| `amp` | True | True | FP16 mixed precision — required to fit |
| `close_mosaic` | — | 10 | Clean (un-mosaicked) boxes for the final epochs |
| `cache` | disk | disk | Faster epochs; 5k×1080p fits disk |
| `workers` | 4 | 4 | — |
| `seed` | 0 | 0 | Reproducibility |
| `device` | 0 | 0 | Single shared T4 — **coordinate before launching** |

### Optimizer — a visible lever, not a buried default

`optimizer` is exposed as an explicit `train.py` argument, not left to
ultralytics' `optimizer='auto'`. `auto` silently picks AdamW for short
schedules (<10k iterations) and SGD for longer ones — so the optimizer would
quietly change with the epoch count, exactly the kind of indeterminate
behaviour this design avoids (§2).

- **Default: `SGD`** (`lr0=0.01`, `momentum=0.937`, `weight_decay=5e-4`). The
  whole ultralytics recipe — LR, warmup, cosine schedule — is co-tuned for SGD
  on COCO; it generalizes slightly better in the small-data / rare-tail regime
  (§8).
- **Alternative: `AdamW`** (`lr0≈1e-3` — ~10× lower; `lr0=0.01` would diverge).
  Can converge faster in a very short finetune. A one-line A/B, scored on val
  mAP — same status as the `freeze ∈ {0,5,10}` sweep (§5).

Both are first-class `--optimizer` choices; whichever wins on val mAP is kept.

### Resolution: pretrain → finetune "size mismatch"

`yolo11m.pt` is pretrained on COCO at `imgsz=640`; we finetune and serve at
`imgsz=1280`. This is **not** an architectural mismatch — YOLO11 is fully
convolutional and anchor-free, so every weight transfers to any input size;
only the feature-map spatial dimensions change. Finetuning re-calibrates the
head to the new object scale, and the 3-epoch warmup + Stage-1 frozen backbone
ease that transition. Training directly at 1280 is exactly RESEARCH.md §3a's
recommendation — no special handling needed.

The mismatch that *does* hurt is **train vs. inference**: `imgsz` must be
identical across train, val, export, and serving. The TensorRT engine is built
at a fixed `imgsz=1280` (§9), which structurally enforces this.

Two genuine "size" changes, both handled automatically:

- **Class head (80 → 18)**: ultralytics rebuilds the detection head's class
  branch for `nc=18` from `dataset.yaml`; backbone/neck weights load unchanged.
  Nothing to do beyond setting `nc: 18` in `prepare_dataset.py`.
- **Progressive resize** (finetune at 640, then continue at 1280) is available
  but unnecessary — direct 1280 finetuning is standard.

---

## 7. Augmentations

Driven by RESEARCH.md §1c (composite domain, visible matte edges, objects
already pre-rotated):

| Aug | Value | Reason |
|---|---|---|
| `mosaic` | 1.0 (off last 10 ep via `close_mosaic`) | Standard; helps small/sparse scenes |
| `copy_paste` | 0.3 | Mirrors how the data is generated (cut-outs on backgrounds) |
| `mixup` | 0.3 | Pushed up from 0.1 to disrupt the matte-edge cue (see post-submission update below) |
| `erasing` | 0.6 | Pushed up from the 0.4 default for the same reason |
| `degrees` | 5 | Objects are *already* rotated — keep small |
| `fliplr` / `flipud` | 0.5 / 0.0 | Horizontal only; no vertical flips |
| `hsv_h/s/v` | 0.015 / 0.7 / 0.4 | Mild colour jitter (defaults) |
| Noise / blur / JPEG aug | **ON, via `--noise-aug`** (recommended) | See post-submission update below — the matte-edge cue is a shortcut that does **not** transfer to the eval distribution |

**Verified during smoke run**: `copy_paste=0.3` works in detect-only mode in
ultralytics 8.3.40 (flip-based copy-paste, no masks required) — no fallback
needed.

**Post-submission update (Grad-CAM diagnostic).** After the first submission
landed at 0.59 vs a sweep peak of 0.88, EigenCAM
(`finetune/visualize_cam.py`) on the trained model showed attention
clustering on the **matte halo** around composited objects rather than on
the object bodies themselves. RESEARCH.md §1c's earlier guidance ("keep
blur mild so the matte-edge cue survives") is therefore **inverted**: the
matte edge is a shortcut feature that the eval set evidently does not
carry. The augmentation defaults flip accordingly — `--noise-aug` ON,
`mixup=0.3`, `erasing=0.6` — to force the model off the shortcut. Re-run
`visualize_cam.py` after retraining to confirm attention has moved from the
halo onto the object body.

---

## 8. Class imbalance — weighted sampler

12× imbalance; the ship classes + `cargo aircraft` are the rare tail
(RESEARCH.md §1a, §4b). COCO mAP averages per-class AP, so a weak tail costs as
much as a weak head.

- **Mechanism**: `weighted_dataset.py` defines `YOLOWeightedDataset(YOLODataset)`
  — class weights ∝ inverse frequency, per-image weights aggregated across that
  image's labels, `__getitem__` draws training indices by those probabilities
  (val stays sequential). Activated by monkeypatch:
  `import ultralytics.data.build as build; build.YOLODataset =
  YOLOWeightedDataset` — the same monkeypatch pattern the archived `train.py`
  already uses for Albumentations.
- **Aggregation**: `np.mean` default; `np.sum` is a documented toggle (the
  template notes `np.sum` sometimes balances better — dataset-dependent).
- **Caveats (documented, accepted)**: (1) mosaic pulls its other 3 tiles via
  *unweighted* random indices, so balancing is partial, not exact; (2) images
  mixing head and tail classes cannot be perfectly balanced.
- **Decision rule**: keep the sampler **only if per-class val AP on the tail
  improves** — read the per-class AP table first; do not apply it blind
  (RESEARCH.md §4b).

---

## 9. Serving — TensorRT FP16

RESEARCH.md §5: TensorRT FP16 is the top latency lever on a Turing T4
(~2–5× over eager PyTorch). YOLO11m has an NMS head, so a `conf`/`iou`
threshold still applies.

**The version-match trap.** A `.engine` only runs if its TensorRT version
matches the runtime's. Building the engine on the host and `COPY`-ing it in
*re-introduces* mismatch. The engine must be built **inside the container
environment**.

**Approach — build at container startup.** The image bakes `src/weights/best.pt`
(the source of truth, ~40 MB). On first boot, `CVManager.__init__`:

1. If `weights/best.engine` is absent, run
   `YOLO("best.pt").export(format="engine", half=True, dynamic=True,
   imgsz=1280, batch=16)` — built against the container's own TensorRT, so the
   version is correct by construction.
2. Load the `.engine` via `YOLO(...)` (ultralytics loads `.pt`/`.engine`
   through the same API — no hand-written runtime).
3. Run one dummy `predict` to warm up (RESEARCH.md §5 lever 6).

First boot pays a one-time ~1–3 min engine build; every request after is fast.
`.pt` remains the fallback for local dev / if the build fails.

**Manager component spec** (the rewrite of the legacy `cv_manager.py` — replaces
the empty competition template). All weight formats load through `YOLO(...)`:

- `CVManager.__init__(weights_dir, conf, iou, imgsz=1280)` — locate
  `best.engine`; if absent, build it from `best.pt`
  (`export(format="engine", half=True, dynamic=True, imgsz, batch=16)`); load
  it; run one warm-up `predict`.
- `cv_batch(images: list[bytes])` — decode each blob (tolerating per-image
  decode failure), run one batched `predict`, return predictions **in input
  order**.
- `cv(image: bytes)` — single-image wrapper over `cv_batch`.
- `_format(result)` — `xyxy → ltwh` + `category_id`, per `cv/README.md`.
- Imports use the §2 mirrored layout (`from src.bbox_utils import ...`); the
  legacy `try/except ImportError` is removed.

The rewrite happens in the implementation phase, with tests for decode-failure
handling, batch ordering, and the ltwh conversion.

- `requirements.txt` gains `tensorrt` + `onnx` + `onnxslim` (ultralytics'
  engine export goes `.pt → ONNX → .engine` internally).
- The engine is exported with `dynamic=True` (`imgsz=1280` fixed, batch
  dynamic up to 16) so it serves the variable-length `instances` list the
  harness sends, not just a fixed batch of 4.
- **Confidence threshold**: RESEARCH.md §4a is the highest-payoff *serving* fix
  — under the `score=1.0` harness, AP is decided by *which* boxes are emitted,
  so `conf=0.001` floods false positives. The server `conf` must be set to a
  swept, mAP-maximising value. This is serving config, tracked separately from
  this training pipeline, but must not be forgotten.

---

## 10. Parameters & risks to flag

| Item | Action |
|---|---|
| Per-class val AP + confusion matrix | **Read before** any tail-class or augmentation action (RESEARCH.md §4b, §4c) |
| `conf` / `iou` at inference | Sweep on val; set to mAP-max — *not* `conf=0.001` (RESEARCH.md §4a) |
| `copy_paste` in detect mode | Verify it is not a silent no-op (§7) |
| Stratified val split | Revisit if tail-class AP is too noisy to read (§4) |
| Eval-set degradation | Unverified — gates whether noise aug is turned on (§7) |
| Shared T4 | Training fully occupies the GPU — coordinate with teammate before launching |
| Engine build timing | Startup build chosen; if first-request latency is itself budgeted, revisit (build at image-build time needs GPU-enabled `docker build`) |
| YOLO26m | Out of scope for now (YOLO11m first); RESEARCH.md §3b holds it as the next experiment if YOLO11m plateaus |

---

## 11. Adversarial training (Tier B, opt-in via `--adv-train`)

After the anti-halo augmentation retrain narrowed the val/test gap but didn't
close it (0.96 val → 0.68 test), Grad-CAM still showed residual attention on
matte-edge halos. The principled next step is **adversarial training**:
per-batch input perturbation by `ε · sign(∇_x L)` (single-step FGSM, Madry
2018 / Tsipras 2019). Adversarial training is known to develop input-gradient
maps aligned with semantic object features — directly attacking the
non-robust-feature failure mode we observed.

Lives in `cv/finetune/adv_train.py`, opt-in:

```bash
python finetune/train.py --adv-train --adv-eps 0.0157 --adv-warmup 2
```

| Knob | Default | Notes |
|---|---|---|
| `--adv-train` | off | Master switch; off by default for backward compat |
| `--adv-eps` | `4/255 = 0.0157` | Perturbation magnitude in image space |
| `--adv-warmup` | `2` | Stage-2 epochs to ramp ε from 0 (avoids early instability) |

**Stage 1 stays clean** (frozen-backbone head warm-up); **Stage 2** gets the
adversarial dance. Implementation monkeypatches `DetectionModel.loss` to:
clean forward → ∇_x L → perturb → adversarial forward → return adversarial
loss for the optimizer step. ~2× per-batch compute, modest extra VRAM —
batch is auto-dropped to 2 on T4 when `--adv-train` is set.

Validation/acceptance:
- Tests (`cv/tests/test_adv_train.py`) cover the pure helpers
  (`fgsm_perturb`, `current_eps`).
- After training, re-run `finetune/visualize_cam.py`. Attention should move
  onto object bodies (not silhouettes) — that's the load-bearing diagnostic
  that the technique is doing what it's meant to.
- Expected trade-off: val mAP may dip slightly vs the clean-aug run (the
  "robustness–accuracy" tension, Tsipras 2019), but test mAP should improve.

---

## 12. Monitoring & continuous model upload

GCP services other than the workbench itself are off-limits, so monitoring +
artifact storage are split between the workbench's built-in TensorBoard proxy
and W&B (verified reachable from the workbench).

| Concern | Mechanism |
|---|---|
| Live training curves | Ultralytics writes TensorBoard logs to `finetune/runs/*` natively. `tensorboard --logdir cv/finetune/runs --port 6006` is exposed through the workbench's `TENSORBOARD_PROXY_URL` — no SSH tunnel needed. |
| Cross-teammate dashboard, system metrics, phone access | **W&B, opt-in via `WANDB_API_KEY`.** `train.py` calls `_maybe_wandb_init(args)` at start and `_maybe_log_artifact(run, best_pt, val_map)` at end; both no-op when the env var is unset, so offline runs are unaffected. With the key set, ultralytics auto-logs metrics to the active wandb run. |
| Continuous model upload | Each training run uploads `best.pt` as the versioned `cv-best` W&B artifact with `val_map_5095` in its metadata. |
| Promote winner to serving | Tag the chosen version `production` in the W&B UI; `python finetune/promote.py` pulls it to `cv/src/weights/best.pt` for the next image build. |

The W&B integration is two small helpers in `train.py` plus a ~40-line
`promote.py` — both gated behind `WANDB_API_KEY` so the pipeline still works
end-to-end without a W&B account.

---

## 13. Build order

1. `cv/finetune/` scaffold + `requirements-train.txt`; dev `uv` venv.
2. `prepare_dataset.py` — de-hardcoded; generate `finetune/data/`.
3. `weighted_dataset.py` — copy the template.
4. `train.py` — two-stage driver; **short run first**.
5. Read `results.csv` mAP curve + per-class AP → decide sampler keep/drop and
   whether to escalate epochs.
6. Restructure `src/` + `Dockerfile` for the mirrored-path layout; drop the
   import hack.
7. TensorRT startup-build path in `cv_manager.py`; `conf` sweep.

Each step is independently testable; nothing here requires the engine or the
final weights to exist before it can be developed.
