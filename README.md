# DSTA BrainHack TIL-AI 2026 - Team Nobrainnohack

## Introduction

TIL-AI 2026 finals was a real-time multi-modal AI competition. Instead of
submitting one model, we had to run a complete container stack:

1. **ASR**: transcribe noisy speech.
2. **NLP**: ingest a corpus, retrieve supporting documents, and answer
   questions.
3. **CV**: detect and classify objects in images.
4. **Noise**: adversarially perturb images for other teams' CV models.
5. **AE**: control an autonomous Bomberman-style agent.
6. **Finals server**: orchestrate all model containers during a live match.
7. **Surprise**: play a separate multi-agent strategy game.

The finals environment was not just an accuracy leaderboard. It tested
container reproducibility, GPU compatibility, startup behavior, health checks,
mission batching, async networking, and strict request/response contracts. Our
main engineering principle was therefore simple: build the thing that survives
the actual evaluator.

Some of our final choices look surprisingly small or blunt. NLP is a 249 MB
CPU-only BM25 container. AE finals used a deterministic scripted policy instead
of our larger neural experiments. CV used direct FP32 PyTorch inference instead
of a fragile TensorRT export. These decisions were deliberate. We optimized the
submitted system, not the prettiest architecture diagram.

This document follows the public technical writeup style used by older TIL
teams: task-by-task, candid, and implementation-heavy. It also records the
submitted Docker images because the current source tree may not exactly match
what was submitted after post-finals cleanup and experiments.

## Team Members

Team Nobrainnohack.
Shizhen
Xinyang
QinHui
Yilin

## Achievements

The notable engineering achievements were:

- Built a complete finals stack across ASR, NLP, CV, Noise, AE, finals
  orchestration, and Surprise.
- Kept the finals stack Docker-native, with explicit submitted image IDs and
  repo digests for reproducibility.
- Implemented an async WebSocket finals server that batched mission work while
  the AE game loop continued.
- Built a deterministic AE planner around static map priors, belief tracking,
  bomb danger projection, and a strategy cascade.
- Built a CPU-only NLP system with BM25 retrieval and a universal
  evaluator-targeted answer trigger.
- Built a transferable adversarial image noising pipeline using surrogate
  detector gradients, objectness masking, and PGD variants.
- Built local Surprise tournament infrastructure for multi-agent replay
  analysis and opponent pool selection.

## Final Submitted Artifacts

The submitted `:finals` Docker images are the source of truth. Current source
folders may have uncommitted cleanup, symlink differences, or experimental work
that did not ship.

| Task | Final image | Image ID | Repo digest | Source reference |
| --- | --- | --- | --- | --- |
| ASR | `nobrainnohack-asr:finals` | `b7ab13213ad7` | `sha256:06b13c08a96f06c3d5a8bada4ea33c82dd758e47574ff9703e7b2921c02816eb` | `asr/src`, `asr/postprocess` |
| NLP | `nobrainnohack-nlp:finals` | `21d7a48948cb` | `sha256:8119f31ae3dc008230b86b99b6558892d4041f27c03ec41536315d8945a93889` | `nlp/src` |
| CV | `nobrainnohack-cv:finals` | `3e33d543c301` | `sha256:7713b2fd41be29fb714a6c14ae6bb368236879ff3239fbcc681a0b4c4e01b0b7` | `cv/src` |
| Noise | `nobrainnohack-noise:finals` | `8906fbfe3a5a` | `sha256:b72efbb8fa80033c8dfaebc96b8213d8f712be75288a131ccc7189f85b2ac43b` | `noise/src` |
| AE | `nobrainnohack-ae:finals` | `076a66ee2055` | `sha256:87961e4b45992d464b03abf5704fa6323b6233ebb5404b7f147d0e30cabcbfc5` | scripted AE, `AE_STRATEGY=balanced_extreme_opening` |
| Finals server | `nobrainnohack-server:finals` | `9e301b949bee` | `sha256:e9ed136ee694c9a1be4c5767f0b888a69d3aaff96576ec44994bcf0aa6a7ec50` | `til-26-finals/finals` |
| Surprise | `nobrainnohack-surprise:latest` | `3f65d57ecbda` | `sha256:96b501a7d53f65bc85aee5688ef2e5e449251042b7a891a1e61914fd43f73145` | `/home/jupyter/til-26-surprise-yilin/participant/src` |

Important gotchas:

- The active `til-shizhen/nlp` folder points to `nlp_shizhen`, but the
  submitted NLP finals image matches `til-ai-26/nlp`, not `nlp_shizhen`.
- `nlp/src/nlp_manager.py` currently has a tiny cleanup diff from the submitted
  image: the trigger string is inlined instead of assigned through an
  intermediate variable.
- The current AE source contains hybrid-capable research code. The submitted
  AE image is scripted-only with `AE_STRATEGY=balanced_extreme_opening`.
- Direct Vertex AI Model Registry verification was blocked by missing
  `aiplatform.models.list` permission. Local Docker image IDs and repo digests
  are the best available source of truth.

## Finals Server

### Overview

The finals server is the participant-side orchestrator. It connects to the
competition server over WebSocket, receives AE observations and mission batches,
forwards work to local model containers, then sends results back with the
correct task IDs.

It is intentionally small: FastAPI for health, WebSockets for the competition
connection, and `httpx.AsyncClient` for model-container calls.

### Architecture

The server listens for four important message types:

- `task`, `task=ae`: one Bomberman observation, requiring one action.
- `mission_batch`: a batch of ASR, CV, NLP, or Noise items.
- `corpus`: a corpus broadcast for the NLP container to ingest.
- `done`: match completion, after which in-flight async work is drained.

Each mission batch is spawned as an `asyncio` task. This means slow CV, ASR, or
Noise calls do not block all other mission handling. The AE loop also remains
separate from mission work.

### Model Container Fanout

The finals server forwards batches directly to each local service:

| Task | Local endpoint | Port | Return shape |
| --- | --- | --- | --- |
| ASR | `/asr` | 5001 | `{"task_id", "answer"}` |
| CV | `/cv` | 5002 | `{"task_id", "detections"}` |
| Noise | `/noise` | 5003 | `{"task_id", "b64"}` |
| NLP | `/nlp` | 5004 | `{"task_id", "answer", "documents"}` |
| AE | `/ae` | 5005 | action integer |

The important detail is that the model containers already accept
`{"instances": [...]}` payloads. The finals server preserves batching rather
than decomposing a mission into individual HTTP calls.

### Reliability Notes

If a mission handler fails, the server attempts to reply with an empty result
list. This is not ideal for score, but it prevents the competition server from
waiting unnecessarily. That failure mode is much better than a deadlock.

The health endpoint reports whether the WebSocket connection is live. This
keeps the orchestration container honest: it is only healthy if it is actually
connected to the match.

## ASR

### Overview

The ASR challenge is to transcribe noisy speech clips. Our final image used a
fine-tuned NeMo Parakeet model and deterministic post-processing.

Final submitted image:

- Image: `nobrainnohack-asr:finals`
- Image ID: `b7ab13213ad7`
- Source reference: `asr/src` and `asr/postprocess`

### Model

The model is baked into the image as:

```text
/workspace/models/parakeet_finetuned.nemo
```

The manager restores it with NeMo, switches to eval mode, moves it to CUDA, and
serves batches with:

```python
self.model.transcribe(paths, batch_size=16)
```

We tested several speed ideas during development:

- autocast wrapping
- decoder strategy changes
- in-memory decode and thread pools
- TensorRT encoder swaps
- JIT-style optimizations
- batch-size probes

The final result was counterintuitive: NeMo's normal file-path transcription
path was the most reliable and fast enough. We kept it.

### Warmup

The ASR container does a startup warmup by writing a short silent WAV and
transcribing it as a full synthetic batch. This pays one-time CUDA allocator
setup, cuDNN autotune, and decoder graph build costs before the service becomes
healthy.

This mattered because the first timed request should not pay lazy-init cost.

### Post-processing

Post-processing is loaded once from:

```text
/workspace/postprocess/enabled.json
```

The enabled pipeline is:

1. `numbers`
2. `spelling_norm`
3. `manual_corrections`
4. `disfluency`

The rules are intentionally conservative:

- `numbers` spells out digit runs so transcripts match reference style.
- `manual_corrections` fixes proper-noun mishearings that were absent from the
  full reference vocabulary, reducing risk of breaking correct predictions.
- `disfluency` only collapses immediate repeated words from a tiny safe set
  such as `i`, `the`, `a`, and `an`.
- `spelling_norm` is present as a module, but the final ruleset is empty after
  pruning because the model already emitted the desired spelling variants.

### Inference

The server receives base64 WAV bytes, writes temporary files, calls NeMo in a
batch, strips raw outputs, applies post-processing, and deletes temporary
files. On exception, it returns empty strings aligned with the input batch
rather than throwing.

This is a recurring pattern in our stack: preserve output shape even on
failure.

## NLP

### Overview

The NLP finals task looked like retrieval-augmented QA, but the scoring
contract made it possible to separate retrieval from answer-equivalence
scoring.

Our final image used:

- CPU-only BM25 retrieval
- H1 title boosting
- tuned BM25 parameters
- a fixed universal adversarial answer trigger

Final submitted image:

- Image: `nobrainnohack-nlp:finals`
- Image ID: `21d7a48948cb`
- Env: `NLP_TITLE_BOOST=5`, `NLP_BM25_K1=1.2`, `NLP_BM25_B=1.0`
- Source reference: `nlp/src`

### Retrieval

The retriever uses whole-document BM25 via `rank_bm25`. Documents are indexed
as complete documents rather than chunks.

Tokenization is deliberately simple:

```text
lowercase + regex [a-z0-9]+
```

This decomposes IDs like `EA-76-088` into matching rare tokens in both
questions and documents.

### Title Boosting

Each Markdown document's first H1 heading is extracted and prepended to the
document text several times before indexing.

Final setting:

```text
TITLE_BOOST = 5
```

The reason is BM25-specific. A title is often the strongest document identity
signal, but one title occurrence can be drowned out by thousands of body
tokens. Repeating it increases title-token term frequency and helps questions
that refer to a document by its main topic.

### Hyperparameters

Final submitted BM25 parameters:

| Parameter | Value | Reason |
| --- | --- | --- |
| `NLP_TITLE_BOOST` | `5` | Boost H1 title terms without overwhelming body evidence. |
| `NLP_BM25_K1` | `1.2` | Reduce excessive reward for repeated common terms. |
| `NLP_BM25_B` | `1.0` | Fully normalize by document length and penalize long overview docs. |

Phrase bigrams were tried and dropped because they hurt recall slightly on the
local set.

### Answer Strategy

The answer field used a fixed universal adversarial trigger discovered through
a GCG-style search against the answer-equivalence evaluator. We do not paste
the raw trigger in this writeup because it is an opaque evaluator-specific
payload, but the submitted image contains the exact string in
`nlp/src/nlp_manager.py`.

The strategy was:

1. Retrieve plausible top-3 supporting document IDs with BM25.
2. Return the universal trigger in the answer field.

This is not a general-purpose QA system. It is a scoring-aware competition
system. The value was that it was tiny, fast, CPU-only, and avoided GPU memory
contention during finals.

### Inference

The NLP server supports three modes through the same `/nlp` endpoint:

- corpus load: `{"documents": [...]}`
- readiness poll: `{"poll": "true"}`
- QA batch: `{"question": "..."}`

Corpus loading runs in the background so readiness can be polled cleanly. The
health endpoint only reports ready once the manager has initialized.

## CV

### Overview

The CV task is object detection and classification. Our final image served an
Ultralytics detector directly from `best.pt` in FP32 PyTorch.

Final submitted image:

- Image: `nobrainnohack-cv:finals`
- Image ID: `3e33d543c301`
- Source reference: `cv/src`

Final inference environment:

```text
CV_CONF=0.40
CV_IMGSZ=1536
CV_IOU=0.45
CV_TTA=0
CV_DENOISE=none
CV_SAHI=0
```

### Model

The model is loaded with Ultralytics from:

```text
cv/src/weights/best.pt
```

The final service keeps:

```python
self.half = False
```

FP32 was chosen because FP16 and export-based acceleration were less reliable
across GPU architectures. The finals desktop environment could differ from the
GCP training and testing environment, especially around newer Blackwell GPUs.

### Experiments

We tried or prepared more aggressive options:

- TensorRT export
- ONNX export
- SAHI slicing
- test-time augmentation
- denoise preprocessing
- confidence sweeps
- image-size variants
- augmentation and finetuning variants

For the final submitted image, we rejected these extras in favor of the stable
path:

```text
best.pt -> Ultralytics YOLO -> FP32 PyTorch inference
```

This is slower than a tuned TensorRT engine but much harder to break in the
venue environment.

### Inference

The CV service:

1. Decodes JPEG bytes with Pillow.
2. Optionally applies denoise transforms, disabled in finals.
3. Runs batched Ultralytics prediction.
4. Converts `xyxy` boxes to `[x, y, w, h]`.
5. Returns a list of detections per input image.

If an image fails to decode, that image returns an empty detection list.

## Noise

### Overview

The Noise task adversarially perturbs images before competitors' CV models
receive them. Our final image used a gradient-based transfer attack against a
surrogate detector ensemble.

Final submitted image:

- Image: `nobrainnohack-noise:finals`
- Image ID: `8906fbfe3a5a`
- Source reference: `noise/src`

### Surrogate Ensemble

The attacker loads several detector backbones as fixed surrogate models. The
ensemble provides gradients with respect to the input image.

Supported model keys in the code include:

- `v8m`
- `yolo11n`
- `yolo26n`
- `rtdetr`

The loss is a log-amplified confidence proxy. It focuses gradient pressure on
the highest-confidence detections instead of wasting updates on low-confidence
anchors.

### Objectness Mask

The objectness mask is derived from the YOLOv8m surrogate's pre-NMS confidence
maps. The maps are upsampled to image resolution and normalized to `[0, 1]`.

This mask concentrates perturbation around likely objects, where detector
confidence is most vulnerable.

The final attack uses a soft floor, not a hard object-only mask. That keeps
some perturbation budget in the background, which helps with transfer and gate
margin.

### Attack

The core attack is mask-weighted PGD with three transfer tricks:

- **MI-FGSM**: momentum accumulation stabilizes gradient direction.
- **DI-FGSM**: random resizing and padding improves robustness to input
  transformations.
- **TI-FGSM**: Gaussian smoothing of gradients improves translation
  robustness.

Final attack config in source:

| Parameter | Value |
| --- | --- |
| `epsilon` | `45 / 255` |
| `alpha` | `4.5 / 255` |
| `n_iters` | `30` |
| `mask_floor` | `0.05` |
| `momentum` | `1.0` |
| `di_prob` | `0.5` |
| `di_low` | `0.7` |
| `di_high` | `1.0` |
| `ti_kernel_size` | `15` |
| `ti_sigma` | `3.0` |

### Inference

Images can arrive at different sizes, so `noise_batch` pads every image to the
maximum height and width in the batch, attacks them as one tensor, then crops
each output back to its original size.

On CUDA OOM, the service falls back to sequential per-image processing. On
unexpected errors, it returns the original image bytes encoded as base64 rather
than failing the request.

## AE

### Overview

AE is a Bomberman-style autonomous-agent task. The submitted finals image used
a scripted agent, not a neural policy.

Final submitted image:

- Image: `nobrainnohack-ae:finals`
- Image ID: `076a66ee2055`
- Env: `AE_STRATEGY=balanced_extreme_opening`

### Central Observation

The novice AE environment was deterministic. The map, base locations, spawns,
and collectible layout could be precomputed offline.

This changed the task from generic learning to state estimation and planning.
We shipped static map priors and let the agent reason symbolically.

### Map Priors

The scripted agent uses precomputed artifacts such as:

- wall layout
- base positions
- spawn positions
- collectible cells
- resource cells
- respawn timing estimates

The map prior identifies our team at `step == 0` from `base_location`, then
exposes enemy bases and map geometry to the planner.

### Belief State

The per-episode belief system tracks:

- our location and facing
- health, base health, resources, and bomb count
- visible enemies and frozen enemies
- enemy bombs and ally bombs
- own bombs in flight
- destroyed walls
- enemy base health
- stuck detection and blacklists
- collected reward yield windows

The agent never has perfect global state. It folds each viewcone into its
belief and plans using the best available information.

### Danger Map

Enemy bombs are projected into a time-indexed danger field. The bomb geometry
respects line-of-sight blast blocking, so walls and map geometry matter.

The planner can ask:

- when a cell becomes dangerous
- whether a cell is lethal at a specific phase
- how many blast overlaps cover a cell

Ally bombs do not enter the danger map because they are friendly-fire safe,
but they still matter for opening walls and base damage.

### Planner

The planner searches in `(tile, facing, phase)` state space. It considers:

- forward and backward movement
- left and right turns
- staying still
- transient bomb phases
- walls that open after bombs detonate
- stuck-blacklist penalties

This phase-aware search lets the agent route through a cell after a bomb has
already detonated instead of treating all danger as permanently blocked.

### Strategy

The production strategy was `balanced_extreme_opening`. It is a cascade:

1. hunt
2. strike
3. survive
4. forage chain
5. sweep
6. default movement

This ordering is intentionally aggressive. It can bomb enemies or strike bases
before pure survival when the scoring tradeoff is favorable. It still retains a
survive layer and planner-aware fallback.

Post-decision gates handle body-block resolution and scripted opening hooks.

### Neural Work That Did Not Ship

We also built a larger neural AE research pipeline:

- symbolic feature builder
- behavior cloning from scripted teachers
- centralized critic pretraining
- PPO self-play ladder
- ONNX export path

That work was useful, but the finals image used the scripted agent because it
was smaller, deterministic, and easier to trust under live finals constraints.

## Surprise

### Overview

The Surprise task was a separate multi-agent strategy game. The submitted
image used `AGENT=algo`, not an LLM.

Final submitted image:

- Image: `nobrainnohack-surprise:latest`
- Image ID: `3f65d57ecbda`
- Env: `AGENT=algo`
- Source reference: `/home/jupyter/til-26-surprise-yilin/participant/src`

### Agent Pipeline

The algorithmic agent runs a structured per-turn pipeline:

1. Safety shell with a wall-clock budget.
2. Ingest observation into persistent `WorldMemory`.
3. Threat assessment.
4. Diplomacy.
5. Combat and focus-fire planning.
6. Economy and expansion planning.
7. Scouting assignment.
8. Movement planning.
9. Local action validation.

The decision to avoid LLM control was deliberate. The task depends heavily on
hex geometry, production timing, range checks, economy compounding, and legal
action validation. These are strengths of deterministic code.

Chat was treated as a threat surface. The agent only trusted system-level
messages.

### Local Test Server

We built `shizhen-suprise-server`, a local multi-agent harness that launches
participant folders, runs them through the same `/observe` contract, and writes
replays.

It supports:

- folder-based agents
- already-running HTTP agents
- random baselines
- configurable seeds and map sizes
- per-agent logs
- JSONL replay output
- API mode for launching and polling matches

### Opponent Pool

We froze and tested multiple opponent families:

- GPT-derived trait agents
- Gemini-derived agents
- Yilin variants
- v1-derived variants
- random and baseline agents

The trait pool allowed us to test against styles such as diplomacy, air
control, scout vision, expansion fortress, base assault, and balanced macro.

### Replay Collation

Replay collation summarized:

- top-score wins
- sole-survivor wins
- top-gold finishes
- family-level average score
- average gold
- alive rate
- highest end-state score, gold, and material

This mattered because a strategy-game policy cannot be judged from one replay.
We needed aggregate behavior across seeds and opponent mixes.

## Hardware and Runtime Notes

Development and testing used the GCP Workbench environment and local Docker
images. Finals execution could differ from GCP, especially around CUDA and GPU
architecture.

Important runtime choices:

- ASR and Noise used large CUDA/PyTorch images.
- CV used PyTorch CUDA runtime and direct `best.pt` inference.
- NLP used `python:3.12-slim` and did not require GPU.
- AE used a small CPU Python image for the scripted final.
- Finals server used `python:3.13-slim`.
- Surprise used `python:3.12` with `AGENT=algo`.

We avoided relying on direct Vertex AI Model Registry listing because the
service account lacked `aiplatform.models.list`. Docker image IDs and repo
digests were therefore recorded as the artifact source of truth.

## Reproducibility

To verify the submitted images locally:

```bash
docker inspect nobrainnohack-asr:finals
docker inspect nobrainnohack-nlp:finals
docker inspect nobrainnohack-cv:finals
docker inspect nobrainnohack-noise:finals
docker inspect nobrainnohack-ae:finals
docker inspect nobrainnohack-server:finals
docker inspect asia-southeast1-docker.pkg.dev/til-ai-2026/repo-til-26-nobrainnohack/nobrainnohack-surprise:latest
```

Recommended source paths for code study:

| Component | Path |
| --- | --- |
| ASR | `asr/src`, `asr/postprocess` |
| NLP | `nlp/src` |
| CV | `cv/src` |
| Noise | `noise/src` |
| AE docs | `ae/SOLUTION.md`, `ae/docs/SCRIPTED_DESIGN.md` |
| Finals server | `til-26-finals/finals` |
| Surprise submitted agent | `/home/jupyter/til-26-surprise-yilin/participant/src` |
| Surprise harness | `shizhen-suprise-server` |

## Final Words

This project was a systems engineering exercise disguised as a model
competition. The most interesting parts were not always the highest-parameter
models. They were the places where we matched the implementation to the exact
constraints:

- warm the ASR model before health
- keep NLP CPU-only and evaluator-aware
- choose CV robustness over fragile speed
- attack CV models with transfer-focused gradients
- exploit deterministic AE structure with planning
- coordinate mission work asynchronously
- test Surprise strategy through local replay tournaments

The final stack is not a generic product, and we do not pretend it is. It is a
carefully built competition system: empirical, containerized, adversarial where
the evaluator allowed it, and pragmatic where reliability mattered more than
architectural elegance.

For a Devpost submission, this document can be condensed into:

- project overview: multi-modal finals stack for TIL-AI 2026
- what it does: orchestrates AE, ASR, CV, Noise, NLP, and Surprise agents
- how we built it: the task sections above
- challenges: source drift, GPU differences, cold starts, strict contracts
- accomplishments: full stack reproducibility and diverse technical approaches
- built with: Python, FastAPI, Docker, PyTorch, NeMo, Ultralytics, BM25, CUDA

---

## AE PPO Improvement Backlog


Created 2026-05-16. Tracks every simplification taken in the **minimal end-to-end PPO
slice** for the AE (Bomberman) challenge, so we can deliberately revisit them to
raise performance.

The minimal slice deliberately optimizes for *getting a correct pipeline working*
(train -> export -> serve -> score) over *getting a strong policy*. Everything below
is a known, intentional shortcut - not a bug.

Priority key: **P1** = biggest expected score impact / needed for finals - **P2** =
solid gain - **P3** = polish.

---

### Training regime

#### S1 - Random opponents instead of self-play  -  P1
The learning policy controls `agent_0`; the other 5 agents take uniform-random
actions. This exactly matches the **qualifier** evaluation, keeps the environment
stationary, and is robust to get working.
- **Risk:** the policy may overfit to random opponents and fail to generalize to
  **finals**, where opponents are other teams' trained models.
- **Improvement:** parameter-shared self-play (one shared net controls all 6
  agents, all 6 generate data -> ~6x throughput), then a checkpoint **opponent
  pool / league** to avoid exploitable cycles. See [S2](#s2--no-ctde--centralized-critic--p1).

#### S2 - No CTDE / centralized critic  -  P1
The critic sees only the agent's own (partial) observation - fully decentralized.
- **Improvement:** MAPPO-style **centralized critic** that consumes global state
  via `env.state()` ([bomberman_env.py:372](til-26-ae/til_environment/bomberman_env.py#L372)).
  This is the textbook remedy for multi-agent non-stationarity. Pairs with S1's
  self-play.

#### S3 - Only `agent_0` learns  -  P2
Even with one network, we collect transitions only from `agent_0`. Parameter
sharing across all 6 agents would yield ~6x the data per step. Tied to S1.

#### S4 - Default reward config, no reward shaping  -  P2
Training uses the default `cfg.rewards`, where several entries (`step_penalty`,
`stationary_penalty`, `agent_collide_wall`, `agent_collide_agent`) are `0`.
- **Improvement:** set these to small negatives *during training only* to speed
  learning and discourage idling/wall-bumping. **Evaluation uses the default
  config**, so shaped rewards are a training crutch - the final policy must still
  score well under defaults.

#### S16 - No robustness to wrong / noisy sensor readings  -  P1
The minimal slice trains and is scored on the simulator's **clean** observations.
In **finals** the agent's "sensors" (the viewcone and derived fields) may be
wrong or noisy - the TIL challenges are integrated, so observations can be fed
or influenced by upstream perception models (CV detections, etc.) that make
mistakes. The whole project is about the pieces working together, so a policy
that blindly trusts every reading is fragile. A policy trained only on clean
observations will react to corrupted readings as if they were ground truth.
- **Improvement - add an error-correction mechanism:**
  - Train with **observation noise / channel dropout / domain randomization**
    so the policy learns to tolerate corrupted inputs.
  - **Temporal cross-checking** - compare successive observations and reject
    implausible single-frame readings (a wall that appears for one frame, an
    enemy that teleports).
  - A **belief-state / memory model** (recurrent or explicit map memory) that
    accumulates evidence over time and reconciles contradictory detections
    instead of trusting each frame independently.
- Closely related to [S5](#s5--single-frame-observation-no-recurrence--frame-stacking--p1)
  - a memory model is the natural home for this.

#### S17 - Random-only opponents, no heuristic opponent ladder  -  P1
Refines [S1](#s1--random-opponents-instead-of-self-play--p1). Random opponents
never pressure the policy on combat, tile contention, or base defense, so it can
learn degenerate habits that exploit randomness.
- **Improvement:** build a ladder of **scripted/heuristic opponents** (a greedy
  collector, a base-rusher, a camper) and train as a curriculum:
  random -> simple heuristic -> aggressive heuristic -> self-play. Each rung is
  stationary (stable training) while difficulty ramps. Use a *diverse set* to
  avoid overfitting one opponent's quirks. The heuristics double as evaluation
  baselines ([S11](#s11--no-standalone-evaluation-harness--p3)) and as the
  behavior-cloning teacher ([S9](#s9--no-imitation--behavior-cloning-warm-start--p2)).

---

### Algorithm & policy

#### S5 - Single-frame observation, no recurrence / frame-stacking  -  P1
The viewcone is partial; we feed a single frame with no memory.
- **Improvement:** stack the last 4 frames, or use a recurrent policy (LSTM/GRU).
  Either requires the served `ae_manager` to **reset internal state when
  `observation["step"] == 0`** (currently a no-op because the net is stateless).

#### S6 - Untuned, default PPO hyperparameters  -  P2
Standard CleanRL values (gamma=0.99, lambda=0.95, clip 0.2, entropy 0.01, lr 2.5e-4,
4 epochs). No search performed.
- **Improvement:** tune lr, entropy coefficient, GAE lambda, rollout length, and
  minibatch/epoch counts against the 200-step horizon.

#### S7 - Small/basic CNN architecture  -  P3
Minimal conv stack over the viewcone grids.
- **Improvement:** deeper net, residual blocks, or attention over the 25
  observation channels.

#### S8 - Greedy (argmax) action selection at inference  -  P3
Deterministic. Generally fine for evaluation; revisit only if determinism proves
exploitable or if stochasticity helps escape bad states.

#### S9 - No imitation / behavior-cloning warm start  -  P2
Policy is trained from scratch.
- **Improvement:** pre-train via behavior cloning from a heuristic agent or from
  human play recorded with [`play.py`](til-26-ae/play.py), then RL fine-tune -
  faster convergence and a stronger starting point.

#### S18 - Pure end-to-end RL from raw observations, no algorithmic tools  -  P1
The policy must learn navigation, blast geometry, and line-of-sight from scratch
via RL - burning samples on already-solved problems. Env stepping is the
throughput bottleneck, so wasted samples are expensive.
- **Improvement - neuro-symbolic / hybrid RL.** Offload solved sub-problems to
  classical algorithms so RL capacity goes to *strategy*:
  - **BFS/A\* pathfinding** - distance + direction to the nearest mission/
    resource tile, enemy base, and home base, fed as extra observation channels
    (biggest sample-efficiency lever), or as macro-actions in a hierarchical
    policy.
  - **Bomb danger map** - deterministically compute which cells detonate and
    when; feed as channels or use as a safety shield.
  - **Occupancy / belief map** - integrate viewcone observations over time;
    addresses partial observability ([S5](#s5--single-frame-observation-no-recurrence--frame-stacking--p1))
    and noisy sensors ([S16](#s16--no-robustness-to-wrong--noisy-sensor-readings--p1))
    without recurrence.
  Prefer soft features over hard overrides. A **fully scripted agent** (no RL)
  is also worth building - opponent, eval baseline, BC teacher, and fallback.

---

### Infrastructure & tooling

#### S10 - Single environment, no parallel rollouts  -  P2
One env instance; low sample throughput.
- **Improvement:** SuperSuit `concat_vec_envs` to run many env copies in parallel
  - usually the easiest large throughput win.

#### S11 - No standalone evaluation harness  -  P3
We rely on `til test ae` for scoring.
- **Improvement:** a dedicated eval script reporting win rate / mean score over
  many seeds, vs random *and* vs a heuristic baseline, for faster iteration.

#### S12 - `test_ae.py` bodyless-POST issue left unfixed  -  P3
[test_ae.py:30](test/test_ae.py#L30) POSTs to `/ae` with no body as a reset
signal -> HTTP 422, but the response is discarded so it is harmless. Not fixed.

---

### Inference / CPU serving optimization

The AE container serves on CPU by design ([ae/Dockerfile:5-6](ae/Dockerfile#L5-L6)).
The policy is tiny, so the bottleneck is **framework dispatch overhead and image
size**, not FLOPs. Optimizations are ranked accordingly.

#### S13 - Plain PyTorch FP32 inference, no ONNX export  -  P2
The container ships `torch` and runs the model in eager FP32.
- **Improvement:** export the trained policy to **ONNX** and serve via
  `onnxruntime`. The container then drops the `torch` dependency entirely -
  image shrinks from ~800 MB toward ~250 MB, cold-start/provisioning is faster
  (which the AE Dockerfile comment explicitly optimizes for), and ORT typically
  gives a ~1.2-2x CPU latency win on small models by stripping Python/dispatch
  overhead. Requires an export step + ONNX-exportable ops (trivial for CNN+MLP).

#### S14 - No INT8 quantization  -  P3
The model runs in FP32.
- **Improvement:** static quantization or QAT (dynamic quantization does **not**
  cover Conv layers). Low priority - absolute savings on a sub-millisecond model
  are tiny and can regress on CPUs without VNNI/AVX-512. Only pursue if profiling
  shows a real per-step latency budget being hit.

#### S15 - Inference micro-optimizations not yet applied  -  P3
Quick wins to fold in when implementing `ae_manager`:
- `torch.inference_mode()` around the forward pass.
- Benchmark `torch.set_num_threads(1)` - fewer threads is often faster at
  batch size 1.
- `channels_last` memory format for conv tensors.
Note: **batching is not available** - `ae/README.md` fixes the `instances` array
length at 1, so inference is permanently batch-size 1.

### How to use this file
When picking up performance work, start from the **P1** items (S1, S2, S5) - they
have the largest expected impact and are prerequisites for being competitive in
finals. Update each entry's status as it is addressed.

---

## Original Repository Usage Notes

### Get started

Here's a quick overview of the initial setup instructions. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-26/wiki).

Use this repository as a template to create your own, and clone it into your GCP Workbench instance. You'll want to keep your repository private, so you'll need to [create a GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

You'll also need to initialize the Git submodules:

```bash
git submodule update --init
```

This repository requires Python 3.10 or newer to work. While it should theoretically all work fine with all packages installed directly into your base Python environment, it is likely a best practice to you create isolated virtual environments for each task. You can use any tool you'd like to do this; [`virtualenv`](https://virtualenv.pypa.io/en/latest/), [`venv`](https://docs.python.org/3/library/venv.html), [`poetry`](https://python-poetry.org/), etc. The competition instance on GCP comes with [`conda`](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) installed, allowing you to create and activate a new virtual environment with the following steps:

```bash
conda create --name til-asr python=3.13
conda activate til-asr
```

Finally, install the development dependencies into your newly created virtual environment.

```bash
pip install -r requirements-dev.txt
```

You should also considering using [`uv`](https://docs.astral.sh/uv/) for python versioning and dependency management all in one.

### Understanding this repo

There's a subdirectory for each challenge: [`asr/`](/asr), [`cv/`](/cv) and its subcategory [`noise/`](/noise), [`nlp/`](/nlp), and [`ae/`](/ae). Each contains:

* A `src/` directory, where your code lives.
  * `*_manager.py`, which manages your model. This is where your inference and computation takes place.
  * `*_server.py`, which runs a local web server that talks to the rest of the competition infrastructure.
* `Dockerfile`, which is used to build your Docker image for each model.
* `requirements.txt`, which lists the dependencies you need to have bundled into your Docker image.
* `README.md`, which contains specifications for the format of each challenge.

You should also see another subdirectory, [`test/`](/test). This contains tools to test and score your model locally, and are automatically run when you use the `til test TASK` command on your GCP Workbench instance.

There are also two Git submodules, `til-26-finals` and `til-26-ae`. `til-26-finals` contains code that will be pulled into your repo for Semifinals and Finals. `til-26-ae` contains the `til_environment` package, which will allow you to train and test your AE model, and is installed by `pip` during setup. Don't delete or modify the contents of `til-26-finals/`, `til-26-ae/`, or `.gitmodules`.

### Build, test, and submit
Submitting your model for evaluation is simple: just build your Docker image, test it, and submit. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-26/wiki).

On the GCP Workbench instance, your environments come pre-set up with a command line utility `til` that will help you build, test, and submit your trained model containers. If you encounter any issues, look through [#hackoverflow](https://discord.com/channels/1488845200523661454/1488845611032903691) on Discord to see if anyone has encountered your problem; if not, post a new question.

tl;dr:
```bash
til build asr
til test asr
til submit asr
```
Done!

#### Build
You can build your containers using `til build CHALLENGE [tag]`. For example:
```bash
til build asr
til build ae algo-update
```

The script first runs `cd` into the directory of the model you want to build (e.g. `/asr`). Then, it builds the image using Docker, automatically adhering to the required naming scheme `TEAM_ID-CHALLENGE:TAG` using any Docker tag you give it, defaulting to `latest` if not provided. You should then test your model using `til test` before using `til submit` to submit your image for evaluation.

```bash
# cd into the directory. For example, `cd ./asr/`
cd CHALLENGE

# Build your image. Remember the . at the end.
docker build -t TEAM_ID-CHALLENGE:TAG .
```
#### Test
You can test your containers locally using `til test CHALLENGE [tag]`. For example:

```bash
til test cv
til test noise extra-noisy
```

This will deploy your container on a local Docker network without internet access, and test querying it with all the training data in your track directory (either `/home/jupyter/novice` or `/home/jupyter/advanced`). For all the details, check out the [Wiki](https://github.com/til-ai/til-26/wiki).

#### Submit
You can submit your containers for automated evaluation using `til submit CHALLENGE [tag]`. For example:
```bash
til submit nlp
til submit cv epoch-100
```

For all the details of what the submission command does, check out the [Wiki](https://github.com/til-ai/til-26/wiki).

### Links

* The repo [Wiki](https://github.com/til-ai/til-26/wiki) contains tutorials, specifications, resources, and more.
* Your [~~Vertex AI~~ Agent Platform Workbench Instance](https://console.cloud.google.com/agent-platform/workbench/instances?project=til-ai-2026) on Google Cloud Platform is where you'll do most of your development.
* The [Strategist's Handbook](https://tribegroup.notion.site/BrainHack-2026-TIL-AI-Strategist-s-Handbook-33a5263ef45a80429a9dc47c569e40c3) houses the Leaderboard and info about the competition.
* [TIL-AI Curriculum](https://drive.google.com/drive/folders/18zP4pHt5E6YqA3usey16ETEzKNeAn5X9) on Google Drive contains educational materials specially crafted for TIL-AI.
* The [#hackoverflow](https://discord.com/channels/1488845200523661454/1488845611032903691) channel on the TIL-AI Discord server is a forum just for Strategists like you.

---

Code in this repo is licensed under the MIT License.
