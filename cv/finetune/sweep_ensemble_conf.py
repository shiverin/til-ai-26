"""Run sweep_conf.py per ensemble model, write ensemble_conf.json.

Each model's best conf is selected independently — under the score=1.0
harness, the optimal threshold differs per model.

Usage:
    python sweep_ensemble_conf.py \\
        --m1 runs/M1_yolov8n_R1_light/weights/epoch12.pt \\
        --m2 runs/M2_yolo11n_R2_heavy_color/weights/epoch27.pt \\
        --m3 runs/M3_yolo11n_R3_heavy_copy_paste/weights/epoch27.pt \\
        --m4 runs/M4_yolo26n_R4_mosaic_mixup/weights/epoch27.pt \\
        --m5 runs/M5_yolo26n_R5_anti_overfit/weights/epoch4.pt \\
        --output ../src/weights/ensemble_conf.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FINETUNE = Path(__file__).resolve().parent


def sweep_one(weights: Path, imgsz: int) -> float:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        out_csv = Path(tf.name)
    subprocess.run(
        [sys.executable, str(FINETUNE / "sweep_conf.py"),
         "--weights", str(weights),
         "--imgsz", str(imgsz),
         "--confs", "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55",
         "--output", str(out_csv)],
        check=True,
    )
    # Pick best from the csv (conf with max map_5095).
    import csv
    best_conf = 0.40
    best_map = -1.0
    with out_csv.open() as f:
        for row in csv.DictReader(f):
            m = float(row["map_5095"])
            if m > best_map:
                best_map = m
                best_conf = float(row["conf"])
    out_csv.unlink()
    return best_conf


def main() -> None:
    p = argparse.ArgumentParser()
    for slot in ("m1", "m2", "m3", "m4", "m5"):
        p.add_argument(f"--{slot}", type=Path, required=True,
                       help=f"path to model {slot.upper()} checkpoint (.pt)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--output", type=Path, required=True,
                   help="ensemble_conf.json output path")
    args = p.parse_args()

    confs: dict[str, float] = {}
    for slot in ("m1", "m2", "m3", "m4", "m5"):
        weights: Path = getattr(args, slot)
        if not weights.exists():
            raise SystemExit(f"missing weights: {weights}")
        print(f"[sweep] {slot} {weights}")
        confs[slot] = sweep_one(weights, args.imgsz)
        print(f"[sweep] {slot} best conf = {confs[slot]:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(confs, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
