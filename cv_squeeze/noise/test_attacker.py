"""Per-transform eyeball + metric test for noise_attacker.

For each val image, run each transform individually (halo / sky_noise /
horizon / blur_edges) and the full composite, then:
  1. Compute SSIM, RMSE vs clean
  2. Run YOLO best.pt on clean and perturbed, match boxes by IoU >= 0.5
  3. Save side-by-side {clean | perturbed} for visual inspection

CPU YOLO to avoid contention with any GPU workload.

Outputs:
  cv_squeeze/noise/test_out/{img}_{transform}.jpg  — side-by-side
  cv_squeeze/noise/test_out/summary.csv            — metrics table
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import cv2  # type: ignore
import numpy as np

# Isolate ultralytics settings
HERE = Path(__file__).resolve().parent
CV_SQUEEZE = HERE.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(CV_SQUEEZE / ".ultralytics"))

# Local import
sys.path.insert(0, str(HERE))
import noise_attacker  # noqa: E402

from skimage.metrics import structural_similarity as ssim  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WEIGHTS = CV_SQUEEZE / "src" / "weights" / "best.pt"
VAL_IMAGES = Path("/home/jupyter/til-ai-26/cv/finetune/data/val/images")
VAL_LABELS = Path("/home/jupyter/til-ai-26/cv/finetune/data/val/labels")
OUT_DIR = HERE / "test_out"


def load_gt_boxes(img_id: int, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO-format labels for img_id, return Nx4 xyxy in pixel coords."""
    lbl = VAL_LABELS / f"{img_id}.txt"
    if not lbl.exists():
        return np.zeros((0, 4), dtype=np.float32)
    out = []
    for line in lbl.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        out.append([x1, y1, x2, y2])
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)

# 4 diagnostic images we have CAMs on
TEST_IDS = [4662, 1567, 1852, 4969]


def _make_transforms(gt_boxes: np.ndarray) -> list[tuple[str, callable]]:
    """Return [(name, fn)] for each transform; fn(img) -> perturbed img.

    Each closure gets its own deterministic rng so cross-transform RNG draws
    can't shift each other's outputs (the bug that bit us at strength gating).
    `gt_boxes` is per-image Nx4 xyxy for the box-restricted transforms.
    """
    return [
        ("halo",        lambda img: noise_attacker.add_fake_halo(
                             img, np.random.default_rng(0))),
        ("sky_noise",   lambda img: noise_attacker.add_sky_noise(
                             img, np.random.default_rng(0))),
        ("horizon",     lambda img: noise_attacker.add_fake_horizon(
                             img, np.random.default_rng(0))),
        ("blur_edges",  lambda img: noise_attacker.blur_hard_edges(
                             img, np.random.default_rng(0))),
        ("blur_object", lambda img: noise_attacker.blur_object_edges(
                             img, gt_boxes)),
        ("soft_outline", lambda img: noise_attacker.soften_object_outline(
                              img, gt_boxes)),
        ("soft_outline_gc", lambda img: noise_attacker.soften_object_outline_gc(
                                 img, gt_boxes)),
        ("composite",   lambda img: noise_attacker.apply(
                             img, strength=1.0, seed=0)),
    ]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def _xyxy(res) -> np.ndarray:
    """Boxes as Nx4 float array; empty array if none."""
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return res.boxes.xyxy.cpu().numpy().astype(np.float32)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every pair (Na, Nb). Returns (Na, Nb) array."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    xa1, ya1, xa2, ya2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    xb1, yb1, xb2, yb2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(xa1, xb1)
    iy1 = np.maximum(ya1, yb1)
    ix2 = np.minimum(xa2, xb2)
    iy2 = np.minimum(ya2, yb2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def match_boxes(clean_xyxy: np.ndarray, noised_xyxy: np.ndarray,
                iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy match clean→noised. Returns (retained, missed, false_pos)."""
    iou = _iou_matrix(clean_xyxy, noised_xyxy)
    nc, nn = iou.shape
    if nc == 0:
        return 0, 0, nn
    matched_n = set()
    retained = 0
    for ci in range(nc):
        order = np.argsort(-iou[ci])
        for ni in order:
            if iou[ci, ni] < iou_thr:
                break
            if ni in matched_n:
                continue
            matched_n.add(int(ni))
            retained += 1
            break
    missed = nc - retained
    false_pos = nn - len(matched_n)
    return retained, missed, false_pos


def side_by_side(clean: np.ndarray, noised: np.ndarray) -> np.ndarray:
    """Stack two same-shape RGB images horizontally with a 4-px separator."""
    h, w, _ = clean.shape
    sep = np.full((h, 4, 3), 64, dtype=np.uint8)
    return np.concatenate([clean, sep, noised], axis=1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[test] loading {WEIGHTS} (CPU)")
    model = YOLO(str(WEIGHTS))

    rows = []
    transform_names: list[str] = []
    for img_id in TEST_IDS:
        img_path = VAL_IMAGES / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"  skip {img_id} — file missing")
            continue
        clean_bgr = cv2.imread(str(img_path))
        clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        h, w = clean.shape[:2]
        gt_boxes = load_gt_boxes(img_id, w, h)

        res_clean = model.predict(
            clean, imgsz=1280, conf=0.25, iou=0.45,
            device="cpu", verbose=False,
        )[0]
        clean_xyxy = _xyxy(res_clean)
        n_clean = len(clean_xyxy)
        gray_clean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)

        transforms = _make_transforms(gt_boxes)
        if not transform_names:
            transform_names = [n for n, _ in transforms]

        print(f"\nimg={img_id}  clean={n_clean} boxes  gt={len(gt_boxes)} labels")
        for name, fn in transforms:
            perturbed = fn(clean)

            r = rmse(clean, perturbed)
            gray_p = cv2.cvtColor(perturbed, cv2.COLOR_RGB2GRAY)
            s_ssim = float(ssim(gray_clean, gray_p, data_range=255))

            res_p = model.predict(
                perturbed, imgsz=1280, conf=0.25, iou=0.45,
                device="cpu", verbose=False,
            )[0]
            p_xyxy = _xyxy(res_p)
            n_p = len(p_xyxy)

            retained, missed, false_pos = match_boxes(
                clean_xyxy, p_xyxy, iou_thr=0.5,
            )
            retention = retained / max(n_clean, 1)

            comp = side_by_side(clean, perturbed)
            out_path = OUT_DIR / f"{img_id}_{name}.jpg"
            cv2.imwrite(str(out_path),
                        cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
            # Also save the perturbed image alone (no clean half) for direct inspection.
            solo_path = OUT_DIR / f"{img_id}_{name}_only.jpg"
            cv2.imwrite(str(solo_path),
                        cv2.cvtColor(perturbed, cv2.COLOR_RGB2BGR))

            rows.append({
                "img": img_id, "transform": name,
                "rmse": round(r, 2), "ssim": round(s_ssim, 3),
                "n_clean": n_clean, "n_p": n_p,
                "retained": retained, "missed": missed,
                "false_pos": false_pos, "retention": round(retention, 3),
            })
            print(
                f"  {name:10s}  rmse={r:5.2f}  ssim={s_ssim:.3f}  "
                f"retained={retained}/{n_clean}  "
                f"missed={missed}  fp={false_pos}  "
                f"keep={retention:.0%}"
            )

    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n[test] wrote {csv_path}")
    print(f"[test] images in {OUT_DIR}/")
    print()

    if rows:
        print("Per-transform aggregate (4 images):")
        for name in transform_names:
            sub = [r for r in rows if r["transform"] == name]
            mean_rmse = np.mean([r["rmse"] for r in sub])
            mean_ssim = np.mean([r["ssim"] for r in sub])
            mean_keep = np.mean([r["retention"] for r in sub])
            total_fp = sum(r["false_pos"] for r in sub)
            total_missed = sum(r["missed"] for r in sub)
            print(
                f"  {name:10s}  rmse={mean_rmse:5.2f}  "
                f"ssim={mean_ssim:.3f}  keep={mean_keep:.0%}  "
                f"missed={total_missed}  fp={total_fp}"
            )


if __name__ == "__main__":
    main()
