"""Offline calibration: sweep epsilon, pick the largest value with
>=99% per-image gate pass-rate. Writes results to experiments/calibration.csv
and prints the chosen value."""
from __future__ import annotations

import base64
import csv
import io
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

NOISE_DIR = Path(__file__).resolve().parent
# Add /home/jupyter/til-ai-26/test to sys.path so `noise_eval` is importable
# as a top-level package (test/ has no __init__.py).
sys.path.insert(0, str(NOISE_DIR.parent / "test"))

from src.noise_manager import NoiseManager
from src.attack import AttackConfig
from noise_eval.pipeline import EvalPipeline
from noise_eval.fairness_checker import FairnessChecker

VAL_IMG_DIR = Path("/home/jupyter/novice/cv/images")
ANNOTATIONS = Path("/home/jupyter/novice/cv/annotations.json")
THRESHOLDS = NOISE_DIR.parent / "test" / "noise_eval" / "eval_thresholds_v2.yaml"
OUT_CSV = NOISE_DIR / "experiments" / "calibration.csv"
N_IMAGES = 200
EPS_SWEEP = [4 / 255, 6 / 255, 8 / 255, 10 / 255]


def _load_coco_boxes() -> dict[str, np.ndarray]:
    """Returns {file_name: [N, 4] array of COCO-format [x, y, w, h] boxes}."""
    coco = json.loads(ANNOTATIONS.read_text())
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    by_file: dict[str, list] = defaultdict(list)
    for ann in coco["annotations"]:
        fname = id_to_file.get(ann["image_id"])
        if fname:
            by_file[fname].append(ann["bbox"])
    return {f: np.asarray(v, dtype=np.float32) for f, v in by_file.items()}


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    boxes_by_file = _load_coco_boxes()
    imgs = sorted(VAL_IMG_DIR.glob("*.jpg"))[:N_IMAGES]
    print(f"calibrating on {len(imgs)} images, epsilons={EPS_SWEEP}")

    mgr = NoiseManager()
    pipeline = EvalPipeline()
    checker = FairnessChecker(str(THRESHOLDS))

    rows = []
    for eps in EPS_SWEEP:
        mgr.attacker.cfg = replace(mgr.attacker.cfg, epsilon=eps, alpha=eps / 4)
        passes = 0
        for img_path in tqdm(imgs, desc=f"eps={eps:.4f}"):
            orig = np.array(Image.open(img_path).convert("RGB"))
            buf = io.BytesIO()
            Image.fromarray(orig).save(buf, format="PNG")
            out_b64 = mgr.noise(buf.getvalue())
            noised = np.array(
                Image.open(io.BytesIO(base64.b64decode(out_b64))).convert("RGB")
            )
            H, W = orig.shape[:2]
            boxes = boxes_by_file.get(img_path.name,
                                       np.zeros((0, 4), dtype=np.float32))
            summary = pipeline.evaluate_batched_with_boxes(
                [orig], [noised], [boxes])
            metrics = {k: v for k, v in summary.mean.items()}
            res = checker.evaluate(metrics)
            if res.passed:
                passes += 1
            rows.append({
                "eps": eps,
                "image": img_path.name,
                "passed": int(res.passed),
                **metrics,
            })
        rate = passes / len(imgs)
        print(f"  eps={eps:.4f}  pass_rate={rate:.3f}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV}")

    by_eps = defaultdict(list)
    for r in rows:
        by_eps[r["eps"]].append(r["passed"])
    chosen = None
    for eps in sorted(by_eps.keys(), reverse=True):
        rate = sum(by_eps[eps]) / len(by_eps[eps])
        if rate >= 0.99:
            chosen = eps
            break
    print(f"chosen epsilon: {chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
