# CV Engineering Research — TIL-AI 2026 Novice Track

**Problem**: Object detection — locate and classify objects across **18 classes**
(aircraft / land vehicles / ships) in 1920×1080 images. Score = **COCO
`mAP@.5:.05:.95`** (`coco_eval.stats[0]`). Offline Docker, single **Tesla T4
(15 GB, Turing)**. Input is a variable-length batch of base64 JPEGs on `POST /cv`;
output is one prediction list per image (`bbox = [x, y, w, h]`, `category_id`).
Current solution: a YOLO detector fine-tuned with `ultralytics` (the `cv_qh`
build runs **YOLO11m**, the `cv_og`/`ab_build` builds run **YOLOv8m**).

> **Research note**: model-landscape figures below were checked against live
> sources in May 2026 (Ultralytics docs, the YOLO26 arXiv paper, the RF-DETR
> repo/blog). COCO mAP numbers are vendor/paper figures on **COCO val** and are
> a *proxy* — this task's data is a narrow composite domain, so the *relative
> ordering* of architectures transfers but absolute numbers do not. The **EDA
> and visual-inspection sections (§1) are first-hand**: computed directly from
> `~/novice/cv/annotations.json` and rendered crops, not from any external
> source. Points that could not be verified are **[Flag]**ged.

---

## 1. Dataset EDA — what we are actually detecting

All numbers below are computed from the **5,000-image** training set in
`~/novice/cv/` (`annotations.json` + `images/`).

### 1a. Counts, sizes, balance

| Property | Value |
|---|---|
| Images | 5,000 — **all exactly 1920×1080** (no resolution variety) |
| Annotations | 18,501 boxes across **18 classes** |
| Objects / image | min 0, **mean 3.70**, median 4, max 12 |
| Empty images | **173** (3.5 %) have zero objects |
| Class imbalance | **12.0× max/min** |

**Class distribution** (descending):

```
fighter jet 2469   tank 2262   truck 2172   commercial aircraft 1651
helicopter 1571    bus 1290    fighter plane 1176   drone 958
light aircraft 870 car 851     missile 805   van 705
cargo ship 363     yacht 304   cargo aircraft 294   sailboat 279
warship 276        cruise ship 205
```

The head is air/land military (`fighter jet`, `tank`, `truck`); the tail is
**all six ship classes plus `cargo aircraft`** — every maritime class has
< 365 examples and `cruise ship` has only 205. With a 90/10 split that is
**~20 `cruise ship` instances in val** — class AP for the tail will be noisy
and is where the model is weakest.

### 1b. Object size — **this is a small-object detection problem**

Bounding-box size as a fraction of the 1920×1080 frame:

| Metric | p1 | p10 | p50 | p90 | p99 |
|---|---|---|---|---|---|
| **area** (% of image) | 0.09 % | 0.16 % | **0.71 %** | 2.83 % | 3.63 % |
| width  | 1.9 % | 3.3 % | 8.4 % | 17.3 % | 20.8 % |
| height | 2.8 % | 4.4 % | 9.3 % | 18.1 % | 20.7 % |

- **55.3 %** of all objects occupy **< 1 % of image area** (COCO-"small"-ish).
- **44.7 %** occupy 1–5 %. **0 %** exceed 25 %. There are *no* large objects.
- Median box is **162×100 px**; the p10 box is only **64×47 px**.
- Median aspect ratio (w/h) ≈ 1.58 (wide-ish — vehicles/aircraft side-on).

**Why this dominates everything**: at the current `cv_qh` inference size
`imgsz=640`, a 1920-wide image is downscaled **3×**. The median 162×100 px
object becomes **~54×33 px**; the p10 object becomes **~21×16 px** — close to
the floor of what an 8/16/32-stride detector head can resolve. **Input
resolution is the single biggest accuracy lever for this dataset** (§3, §4).

### 1c. Visual inspection — composite ("photoshopped") domain

Rendered samples are saved to [research_assets/](research_assets/):
[sample_scenes.jpg](research_assets/sample_scenes.jpg) (6 full scenes with GT
boxes) and [object_crops.jpg](research_assets/object_crops.jpg) (9 zoomed
single-object crops, one per class).

Observations from inspecting the crops directly:

- **The data is synthetic-composite**: real object cut-outs are pasted onto
  unrelated natural backgrounds — forests, ruins, snowfields, mountainsides.
  Tanks float on dirt, airliners and missiles hang in tree canopies. The
  `train.py` augmentation already exploits this (`copy_paste=0.3` — *"copy_paste
  mirrors how the data itself is generated"*).
- **Matting artifacts**: several cut-outs show a visible **white halo / hard
  matte edge** (fighter plane, helicopter, light aircraft). This is a usable
  signal the model can learn — and a reason **heavy blur/recolor augmentation
  should stay mild**, or it erases the edge cue.
- **Pose & scale are unnatural**: aircraft appear at arbitrary roll angles;
  objects ignore scene perspective. Rotation augmentation should stay small
  (`degrees=5` is fine) — the objects themselves are already rotated.
- **Mild sensor noise / JPEG grain** is visible on some crops. The `cv_og`
  build adds Gaussian-noise/blur/compression augmentation — sensible *only* if
  the held-out eval is similarly degraded; **[Flag]** not verified that it is.
- **Confusable class clusters** (visually near-identical at small scale):
  - `fighter jet` vs `fighter plane` (jet vs prop) — and both vs `light aircraft`
  - `commercial aircraft` vs `cargo aircraft` (same airframe, different livery)
  - `truck` vs `van` vs `bus`; `car` vs `van`
  - `cargo ship` vs `warship` vs `cruise ship` vs `yacht` vs `sailboat`

  These pairs are where **classification** error (not localization) will cost
  mAP, and they overlap heavily with the rare-class tail (§1a). Per-class AP on
  the val split should be read before any architecture change.

### 1d. EDA → engineering implications

| Finding | Implication |
|---|---|
| All objects small, none large | Push `imgsz` up; small-object-aware model/training |
| Fixed 1920×1080 input | No multi-scale robustness needed; favours one high-`imgsz` pass (§3a) |
| 12× class imbalance, ship tail | Watch per-class AP; consider class weighting / tail oversampling |
| Confusable class clusters | Classification head matters as much as the box head |
| Composite domain w/ matte edges | `copy_paste` aug helps; don't over-blur away edges |
| 3.5 % empty images | Model must tolerate / emit nothing — and not hallucinate FPs |

---

## 2. Best object-detection models as of 2026

COCO `mAP@.5:.95` on **COCO val**, vendor/paper figures. T4 latency is
TensorRT-FP16 where the source gives it. *Relative* ordering transfers to this
task; absolute mAP does not.

| Model | Params (m-size) | Arch | COCO mAP (size) | T4 latency | Notes |
|---|---|---|---|---|---|
| **YOLO26** (n/s/**m**/l/x) | ~20 M (m) | CNN, anchor-free, **NMS-free**, no DFL | **53.1** (m) / 55.0 (l) / 57.5 (x) | **4.7 ms** (m, TRT) | Released **Jan 2026**. ProgLoss + **STAL (small-target-aware label assignment)**; MuSGD optimizer |
| **YOLO11** (n/s/**m**/l/x) | 20.1 M (m) | CNN, anchor-free, NMS (TopK) | 51.5 (m) / 53.4 (l) / **54.7** (x) | ~4.7 ms (m) | Mature, ultralytics-native, **current `cv_qh` model** |
| YOLO12 (n/s/m/l/x) | ~20 M (m) | Attention-centric CNN-hybrid, NMS | ~52.5 (m) / **55.2** (x) | ~3 % slower than YOLO11-m | Small gain over YOLO11 (~+1 mAP), slightly slower |
| YOLOv8 (n/s/**m**/l/x) | 25.9 M (m) | CNN, anchor-free, NMS | 50.2 (m) / 53.9 (x) | ~5 ms (m) | **Current `cv_og`/`ab_build` model**; superseded by YOLO11 |
| RT-DETR / RTDETRv2 | ~36 M (R34/R50) | DETR (ViT-ish) transformer, NMS-free | 53.1 (R50) / 54.3 (x) | ~5.0 ms (s) | Transformer; needs more data/epochs to converge |
| **RF-DETR** (base/large) | 29 M (base) | DETR on **DINOv2** backbone, NMS-free | **first real-time 60+ mAP** (60.5 @ res 728, 25 FPS T4) | ~4.5–40 ms | SOTA accuracy, Apache-2.0, ICLR 2026; built for fine-tuning |

### Reading the table for *this* task

- **YOLO26m vs YOLO11m**: YOLO26 is the natural in-family upgrade. Two of its
  three headline changes target exactly our data: **STAL** (small-target-aware
  label assignment) and **ProgLoss** are reported to give *"a substantial
  accuracy boost on datasets with small or occluded objects"* — and §1 says
  this dataset is **~100 % small objects**. Being **NMS-free** also removes the
  `iou`/NMS-threshold knob entirely (relevant to §3). **[Flag]**: YOLO26's
  small-object gain is a paper claim on COCO/aerial sets — verify on *our* val
  split before committing.
- **YOLO12** offers ~+1 mAP over YOLO11 at a small speed cost — not worth a
  migration on its own.
- **RF-DETR** is the accuracy ceiling (first real-time 60+ COCO mAP) and its
  DINOv2 backbone is strong on small objects, but it is a **DETR** — slower to
  fine-tune, hungrier for data/epochs, and a different training stack from the
  team's ultralytics pipeline. High-risk for a Novice-track time budget.
- **RT-DETR** has no clear advantage over YOLO26/RF-DETR here and shares DETR's
  slow-convergence cost.

---

## 3. Keep the current YOLO, or switch — analysis

**Where the project stands now** (from `runs/*/results.csv`):

- `cv_qh` **YOLO11m @ imgsz 640**: by **epoch 16**, val **mAP@.5:.95 ≈ 0.890**,
  mAP@.5 ≈ 0.976, P ≈ 0.94, R ≈ 0.95. Strong and still rising.
- `cv_og` **YOLOv8m @ imgsz 1280**: only **2 epochs** logged (mAP ≈ 0.81 @ ep2)
  — **not** a fair comparison; it simply has not trained.

So the live baseline is already a **~0.89 mAP** YOLO11m. The question is what
moves it further, ranked by expected payoff vs risk.

### 3a. Resolution is the dominant lever — and the chosen approach (highest payoff, low risk)

§1b: the median object is 0.71 % of image area; at the current `imgsz=640` a
1920-wide frame is downscaled 3× and objects shrink badly. A higher `imgsz` is
the single biggest accuracy lever for this dataset. **Decision: scale `imgsz`
up in a single pass — do not tile (§3c).** Object *width in pixels* after the
YOLO downscale:

| Percentile | native 1920w | imgsz=1280 (÷1.5) | imgsz=640 (÷3.0) |
|---|---|---|---|
| p1 width  | ~37 px | ~25 px | ~12 px |
| p10 width | 64 px  | ~43 px | ~21 px |
| p50 width | 162 px | ~108 px | ~54 px |

A YOLO head (strides 8/16/32) localizes reliably down to ~16–24 px, so a single
pass at **`imgsz=1280`** already resolves the *entire* size distribution —
even the p1 tail lands at ~25 px. `imgsz=640` is the only genuinely bad choice
(p10 → 21 px, p1 → 12 px = lost). Compute scales ~quadratically with `imgsz`
(§4 covers the T4 budget).

**Recommendation:**

1. Retrain + infer at **`imgsz=1280`** as the new baseline — the first thing to
   do regardless of model choice.
2. Sweep up (1536 → 1920) and stop when val mAP flattens; do not overshoot.
3. Use **rectangular inference** (`rect=True` / a 16:9 size such as 1280×736).
   Padding 1920×1080 to a square wastes ~44 % of the compute on gray bars; rect
   mode reclaims it, buying higher *effective* resolution at equal latency.
4. Fix `imgsz` **before** the TensorRT export — the engine is resolution-specific.

### 3b. Model swap YOLO11m → YOLO26m (moderate payoff, low-moderate risk)

YOLO26 is a drop-in within the ultralytics API (`yolo26n.pt` is **already in
the repo**, so the m-size weights are one download away). Its STAL/ProgLoss
target small objects directly (§2). Being NMS-free also removes a failure mode
(§3d). Risk is low because the training stack is unchanged. **Recommendation:
once §3a is done, fine-tune YOLO26m head-to-head against YOLO11m at the same
`imgsz` and keep whichever wins on val mAP.** Do not swap blind — verify.

### 3c. SAHI / tiled inference — considered and rejected

Sliced inference (split each frame into overlapping ~640 tiles, detect per
tile, merge) is the textbook small-object trick, and published results show
**+~38 % small-object mAP** on comparable data. **It is the wrong tool here**,
for three EDA-driven reasons:

- **Objects are not small enough to need it.** Tiling earns its 6–9× cost only
  when objects stay sub-~20 px *even at native resolution*. §3a shows a single
  `imgsz=1280` pass already puts the whole distribution — including the p1 tail
  (~25 px) — above the detector floor.
- **Scenes are sparse.** Mean 3.7 objects/image (§1a); there is no tile-density
  pressure that a single high-`imgsz` pass cannot handle.
- **The cost is real.** A 1920×1080 frame → ~6–7 overlapping 640-tiles per
  image = **6–7× the forward passes**, plus boundary double-detections to
  reconcile — messier still with an NMS-free model like YOLO26.

**Decision: do not tile.** Keep SAHI only as a last-resort experiment if a
native-resolution pass somehow plateaus with the tiny tail still unresolved —
the §3a pixel table says it will not.

### 3d. Verdict on switching

- **Do not** stay on YOLOv8 (`cv_og`/`ab_build`) — YOLO11/YOLO26 strictly
  supersede it at equal cost.
- **Do** consolidate on one build. `cv_qh` (YOLO11m) is the furthest along and
  is the right base.
- **Do not** jump to RF-DETR/RT-DETR for the Novice track: the DETR training
  cost and stack change are not justified when a tuned YOLO is already at 0.89.
  Hold RF-DETR as a fallback only if YOLO mAP plateaus well short of target.

---

## 4. The scoring quirk and other task-specific issues

### 4a. **Uniform `score=1.0` — the critical, currently-mis-handled issue**

`test/test_cv.py` builds the COCO results list with a **hard-coded
`"score": 1.0`** on *every* returned detection — it never reads a per-box
confidence. COCOeval therefore cannot rank detections by quality; **AP is
decided purely by *which* boxes are emitted.** Every extra low-quality box is
an unmatched **false positive** that drags precision down at every recall
level, collapsing AP.

**`cv/src/cv_manager.py` gets this exactly wrong.** It runs at `conf=0.001`
with a comment claiming *"mAP ranks ALL detections by score, so a high cutoff
caps recall."* That assumption is **false for this harness** — there is no
score to rank by. Running at `conf=0.001` floods the output with garbage boxes
and **destroys precision/AP**. `cv/cv_qh/eval_local.py` already documents the
correct behaviour and exists to sweep the threshold.

**Recommendation (highest-priority, zero-training fix):** run
`eval_local.py`'s confidence sweep on the val split and set the inference
`conf` to the **mAP-maximising** value (the sweep covers 0.05–0.60). Pair it
with a sane `iou` NMS threshold. This is likely the **single largest cheap
win** available and needs no retraining. (A YOLO26 / other NMS-free model
sidesteps the NMS knob but **still** needs a confidence cutoff for the same
reason — the score=1.0 problem is about *box count*, not NMS.)

### 4b. Class imbalance & the ship/cargo-aircraft tail

§1a: 12× imbalance; all maritime classes + `cargo aircraft` are rare. COCO mAP
is a **mean over per-class AP**, so a weak `cruise ship`/`warship`/`sailboat`
AP costs as much as a weak `tank` AP. Levers: (1) read per-class AP on val
first; (2) oversample tail-class images or use `copy_paste` to inject rare
objects; (3) consider class-balanced loss weighting. Do **not** chase imbalance
blindly — measure per-class AP and target only the classes that are actually
low.

### 4c. Confusable classes (§1c)

Localization is likely already good (mAP@.5 ≈ 0.976); residual mA@.5:.95 loss
is partly **box tightness** and partly **classification** between look-alike
pairs. A larger `imgsz` (§3a) helps both. Inspecting the val confusion matrix
(ultralytics emits one) will show whether e.g. `fighter jet`↔`fighter plane`
is leaking AP.

### 4d. Empty images / hallucinated false positives

3.5 % of images have no objects, and §4a means **every** spurious box is pure
AP loss. After the `conf` sweep (4a), spot-check predictions on the 173 empty
images — the model should mostly emit nothing.

---

## 5. Inference optimization for a T4, offline

### Turing (T4) constraints — load-bearing

- **fp16 yes** (good Turing tensor cores), **bf16 no**, **INT8 yes** (DP4A),
  **INT4 no** (Marlin is Ampere+). No Flash-Attention-2 benefit.
- 15 GB VRAM is ample for any m-size detector — VRAM is **not** the constraint.
  **Throughput at high `imgsz`** is.

### Levers

1. **Export to TensorRT FP16.** Ultralytics `model.export(format="engine",
   half=True)`. YOLO26's docs quote **~4.7 ms/image** for YOLO26m on a T4 via
   TRT — the largest single speedup, and it must be built **at image-build
   time** (offline-safe; bake the `.engine` into the Docker image).
2. **Pick `imgsz` deliberately.** It is both the top accuracy lever (§3a) and
   the top latency cost. Sweep `imgsz ∈ {640, 960, 1280}` and read the
   accuracy/latency curve — choose the largest size the per-request time budget
   allows. The TRT engine is `imgsz`-specific, so fix `imgsz` before export.
3. **Batch the whole request.** `cv_manager.cv_batch` already passes the decode
   list to `model.predict` in one call — good. The harness sends `BATCH_SIZE=4`
   per POST; a TRT engine should be built for a batch dimension that covers it
   (dynamic or fixed-4).
4. **INT8** is available on Turing and would cut latency further, but needs a
   calibration set and can cost mAP on small objects — treat as a stretch
   *after* FP16+`imgsz` are settled, and only if the time budget is tight.
5. **Decode path.** `cv_manager` already decodes JPEG→`PIL`→`np.array` and
   batches; that is fine. Skip any temp-file round-trip.
6. **Warm up at startup** — run one dummy `predict` in `CVManager.__init__` so
   the first real request does not pay CUDA/TRT autotune latency.
7. **SAHI tiling** (§3c) multiplies forward passes — only adopt if its mAP gain
   clears the latency budget.

### Expected effect

| Lever | Effect |
|---|---|
| TensorRT FP16 export | large latency cut (m-size YOLO26 ~4.7 ms/img on T4) |
| `imgsz` 640→960/1280 | **+mAP** (top accuracy lever) at ~quadratic latency cost |
| Tuned `conf` (§4a) | **+mAP**, ~zero latency cost — do this first |
| INT8 (stretch) | further latency cut; possible small-object mAP loss |
| SAHI 2×2 (stretch) | +small-object recall; 4–9× latency |

---

## 6. Recommended stack for THIS problem — ranked

### Rank 1 (do immediately, no retraining): fix the confidence threshold

Run `cv_qh/eval_local.py`'s sweep and set the server `conf` to the
mAP-maximising value; stop running `conf=0.001`. Correct the misleading comment
in `cv_manager.py`. **This is the highest payoff-per-effort action on the
board** and is independent of every other decision.

### Rank 2 (recommended core): YOLO11m → YOLO26m, retrained at high `imgsz`

- **Model**: fine-tune **YOLO26m** — STAL + ProgLoss target the small-object
  regime that §1 shows *is* this dataset; NMS-free removes a knob. Run it
  **head-to-head against YOLO11m** at the same settings and keep the winner.
- **Resolution**: train and infer at **`imgsz` 960–1280** (§3a) — the largest
  single accuracy lever. Sweep and pick per the latency budget.
- **Training**: keep `copy_paste` (mirrors the composite domain), keep rotation
  mild, keep blur/recolor mild so matte-edge cues survive (§1c). Address the
  ship/cargo-aircraft tail (§4b) only after reading per-class val AP.
- **Inference**: TensorRT FP16 engine baked into the image at the chosen
  `imgsz`; warm-up at startup; batch the request.
- **Why**: stays inside the team's working ultralytics pipeline (lowest risk),
  fixes the actual data problem (resolution), and adopts a model whose new
  features are aimed at exactly this dataset.

### Rank 3 (rejected — last-resort experiment only): SAHI tiled inference

Considered and rejected (§3c): a single high-`imgsz` pass resolves the whole
size distribution, scenes are sparse, and tiling costs 6–7× latency. Revisit
only if a native-resolution pass implausibly plateaus with the tiny tail still
unresolved.

### Rank 4 (fallback only): RF-DETR

The accuracy ceiling (first real-time 60+ COCO mAP, DINOv2 backbone strong on
small objects). Consider **only** if a tuned YOLO26m plateaus well short of
target — it costs a training-stack change and DETR's slower convergence, a poor
trade for a Novice-track time budget unless the mAP gap is real.

### Things to drop / avoid

- **`conf=0.001` at inference** — actively destroys AP under this harness (§4a).
- **YOLOv8** (`cv_og`, `ab_build`) — superseded by YOLO11/YOLO26; consolidate
  on the `cv_qh` build.
- **`imgsz=640`** for final inference — too small for 0.7 %-area objects.
- **INT4 / bf16** — unsupported on Turing.
- **Heavy noise/blur augmentation** unless the eval set is verified degraded
  (**[Flag]** unverified) — it can erase the composite matte-edge cue.
- **Maintaining three parallel CV builds** (`cv_qh`, `cv_og`, `ab_build`) —
  pick one, delete the divergence.

### Bottom line

The live baseline (fine-tuned **YOLO11m**, ~0.89 mAP) is a sound direction. The
concrete improvements, in order: **(1)** stop inferring at `conf=0.001` — sweep
and set a real threshold (free, biggest cheap win); **(2)** retrain at
**`imgsz=1280`** (sweeping toward 1920) with rectangular inference — the dataset
is ~100 % small objects and resolution is the dominant lever; **(3)** swap to
**YOLO26m** if it wins a head-to-head (its small-object features fit the data);
**(4)** TensorRT-FP16 the final engine for the T4. **Sliced (SAHI) inference is
rejected (§3c)**; RF-DETR is a fallback only.

---

## Sources

Primary sources fetched live (May 2026):

- YOLO26 paper — *"YOLO26: Key Architectural Enhancements and Performance
  Benchmarking for Real-Time Object Detection"* — https://arxiv.org/html/2509.25164v5
- Ultralytics — YOLO11 vs RT-DETR comparison — https://docs.ultralytics.com/compare/yolo11-vs-rtdetr
- Ultralytics — YOLO12 model docs — https://docs.ultralytics.com/models/yolo12
- Ultralytics — YOLO26 / YOLO11 / YOLOv8 comparison blog — https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models
- Ultralytics — SAHI tiled inference guide — https://docs.ultralytics.com/guides/sahi-tiled-inference
- RF-DETR — Roboflow blog & repo — https://blog.roboflow.com/rf-detr/ , https://github.com/roboflow/rf-detr
- Best object detection models 2026 (Roboflow) — https://blog.roboflow.com/best-object-detection-models/
- Maritime small-object detection w/ adaptive tiling (SAHI) — https://arxiv.org/pdf/2511.19728

First-hand analysis (this repo / this session):

- EDA computed from `~/novice/cv/annotations.json` (5,000 images, 18,501 boxes).
- Visual inspection of rendered samples — see
  [research_assets/sample_scenes.jpg](research_assets/sample_scenes.jpg) and
  [research_assets/object_crops.jpg](research_assets/object_crops.jpg).
- Scoring behaviour read from `test/test_cv.py` (`score=1.0`) and
  `cv/cv_qh/eval_local.py`; training results from `cv/cv_qh/runs/*/results.csv`.

> **[Flag]**ged items needing on-hardware / on-data verification before relying
> on a number: (1) YOLO26's small-object gain on *our* composite val split;
> (2) whether the held-out eval images are noise/blur-degraded (drives §1c
> augmentation choices); (3) the mAP-vs-latency curve across `imgsz` and the
> SAHI-tiling payoff — both must be measured on the T4.
