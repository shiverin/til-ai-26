"""Save the GrabCut foreground mask for each box, overlaid on the image.

Diagnostic: see whether GrabCut is actually segmenting the vehicles or
just returning the box rectangle / nothing.

Outputs cv_squeeze/noise/test_out/dbg_{img}_gcmask.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2  # type: ignore
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_attacker import load_gt_boxes, VAL_IMAGES, OUT_DIR  # noqa: E402

TEST_IDS = [4662, 1567, 1852, 4969]


def grabcut_mask(img_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                 iter_count: int = 5) -> np.ndarray:
    """Return a per-pixel mask: 0 = bg, 64 = pr_bg, 192 = pr_fg, 255 = fg."""
    h, w = img_bgr.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        rw, rh = x2 - x1, y2 - y1
        if rw < 4 or rh < 4:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(img_bgr, mask, (x1, y1, rw, rh),
                    bgd, fgd, iter_count, cv2.GC_INIT_WITH_RECT)
        # Encode 4 grabCut states into a single channel for visualization
        encoded = np.zeros_like(mask)
        encoded[mask == cv2.GC_BGD] = 0
        encoded[mask == cv2.GC_PR_BGD] = 64
        encoded[mask == cv2.GC_PR_FGD] = 192
        encoded[mask == cv2.GC_FGD] = 255
        out = np.maximum(out, encoded)
    return out


def main() -> None:
    for img_id in TEST_IDS:
        path = VAL_IMAGES / f"{img_id}.jpg"
        if not path.exists():
            continue
        bgr = cv2.imread(str(path))
        h, w = bgr.shape[:2]
        boxes = load_gt_boxes(img_id, w, h)
        if len(boxes) == 0:
            print(f"{img_id}: no labels")
            continue
        mask = grabcut_mask(bgr, boxes)
        # Pink heatmap overlay on grayscale image so we see both
        gray3 = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                             cv2.COLOR_GRAY2BGR)
        heat = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(gray3, 0.5, heat, 0.5, 0)
        # Draw boxes on top
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out_path = OUT_DIR / f"dbg_{img_id}_gcmask.jpg"
        cv2.imwrite(str(out_path), overlay)
        n_fg = int((mask == 255).sum())
        n_pr_fg = int((mask == 192).sum())
        print(f"{img_id}: wrote {out_path.name}  fg={n_fg}  pr_fg={n_pr_fg}")


if __name__ == "__main__":
    main()
