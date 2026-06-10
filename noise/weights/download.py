"""Bootstrap the four surrogate weights into noise/weights/.

Copies v8m + yolo11n + yolo26n from the cv/ directory (already on disk),
downloads rtdetr-l.pt via ultralytics' auto-download mechanism.

Idempotent — skips files that already exist.
"""
import shutil
import sys
from pathlib import Path

NOISE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = NOISE_DIR / "weights"
CV_DIR = NOISE_DIR.parent / "cv"

LOCAL_COPIES = {
    "v8m.pt": CV_DIR / "src" / "weights" / "best_v8m_ep1.pt",
    "yolo11n.pt": CV_DIR / "yolo11n.pt",
    "yolo26n.pt": CV_DIR / "yolo26n.pt",
}

def main() -> int:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in LOCAL_COPIES.items():
        dst = WEIGHTS_DIR / name
        if dst.exists():
            print(f"[skip] {name} already exists")
            continue
        if not src.exists():
            print(f"[fail] source not found: {src}", file=sys.stderr)
            return 1
        shutil.copy2(src, dst)
        print(f"[copy] {src} -> {dst}")
    # rtdetr-l.pt: trigger ultralytics auto-download into noise/weights/.
    rtdetr = WEIGHTS_DIR / "rtdetr-l.pt"
    if not rtdetr.exists():
        from ultralytics import RTDETR
        print("[dl  ] rtdetr-l.pt (~80MB)")
        # Ultralytics caches in cwd; cd there first.
        import os
        cwd_before = os.getcwd()
        os.chdir(WEIGHTS_DIR)
        try:
            RTDETR("rtdetr-l.pt")  # auto-downloads on instantiation
        finally:
            os.chdir(cwd_before)
    print("[done]")
    return 0

if __name__ == "__main__":
    sys.exit(main())
