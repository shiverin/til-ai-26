"""Compute local COCO mAP@.5:.95 on the val split for a given checkpoint.

WARNING: local val mAP is an anti-proxy for TIL grader score. This script is
for sanity checks only. Pick checkpoints by TIL submission, not by this number.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
VAL_IMAGES = ROOT / "dataset" / "valid"
VAL_JSON = VAL_IMAGES / "_annotations.coco.json"
INFER = ROOT / "infer.py"

BANNER = (
    "============================================================\n"
    "WARNING: local val mAP is an ANTI-PROXY for the TIL grader.\n"
    "         It ranks models OPPOSITE to TIL on average.\n"
    "         Submit checkpoints to TIL; do not pick by this number.\n"
    "         See the cv-qh-project memory entry for evidence.\n"
    "============================================================"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "large"), default="base")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the anti-proxy banner (don't).")
    args = parser.parse_args()

    if not VAL_JSON.exists():
        raise FileNotFoundError(f"{VAL_JSON} missing; run prepare_data.py first.")

    if not args.quiet:
        print(BANNER)

    with tempfile.TemporaryDirectory() as td:
        preds_path = Path(td) / "preds.json"
        infer_cmd = [
            sys.executable, str(INFER),
            "--weights", str(args.weights),
            "--images", str(VAL_IMAGES),
            "--out", str(preds_path),
            "--variant", args.variant,
            "--conf", str(args.conf),
        ]
        if args.resolution is not None:
            infer_cmd += ["--resolution", str(args.resolution)]
        print(f"Running: {' '.join(infer_cmd)}")
        subprocess.run(infer_cmd, check=True, cwd=ROOT)

        gt = COCO(str(VAL_JSON))
        dt = gt.loadRes(str(preds_path))
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    if not args.quiet:
        print("\n" + BANNER)


if __name__ == "__main__":
    main()
