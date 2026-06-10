# v8m ep1 Leaderboard Experiments

Baseline image: `nobrainnohack-cv:v8m-ft-ep1-real`.

Build confidence candidates without changing the locked archive:

```bash
python finetune/build_v8m_candidate.py --conf 0.30
python finetune/build_v8m_candidate.py --conf 0.35
python finetune/build_v8m_candidate.py --conf 0.40
python finetune/build_v8m_candidate.py --conf 0.45
python finetune/build_v8m_candidate.py --conf 0.50
python finetune/build_v8m_candidate.py --conf 0.55
```

Run conservative continuation only after confidence candidates are tested:

```bash
python finetune/v8m_continue.py
```

Run the corrected soft-outline pipeline from the real v8m ep1 checkpoint:

```bash
python finetune/v8m_outline_correct.py --stage all
```

For a continuation checkpoint, build with an explicit hash bypass:

```bash
python finetune/build_v8m_candidate.py \
  --weights finetune/runs/v8m_ep1_continue/weights/best.pt \
  --expect-hash '' \
  --conf 0.45 \
  --tag v8m-continue-best-conf045
```

Record every tested/submitted image in `experiments/v8m_ep1_candidates.csv`.
