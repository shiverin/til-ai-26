"""infer.py output must be a list of LTWH int-pixel records inside image bounds."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "infer.py"
WEIGHTS_DIR = ROOT / "weights" / "base"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
IMG_FIXTURE = FIXTURES / "images"
SOURCE_IMAGES = Path("/home/jupyter/novice/cv/images")


def _find_checkpoint() -> Path | None:
    if not WEIGHTS_DIR.exists():
        return None
    candidates = sorted(WEIGHTS_DIR.glob("*.pth"))
    return candidates[-1] if candidates else None


@pytest.fixture(scope="module")
def fixture_images():
    """Symlink 3 source images into tests/fixtures/images."""
    if not SOURCE_IMAGES.exists():
        pytest.skip(f"Source images not at {SOURCE_IMAGES}")
    if IMG_FIXTURE.exists():
        shutil.rmtree(IMG_FIXTURE)
    IMG_FIXTURE.mkdir(parents=True)
    src = sorted(SOURCE_IMAGES.glob("*.jpg"))[:3]
    assert len(src) == 3
    for s in src:
        (IMG_FIXTURE / s.name).symlink_to(s)
    return IMG_FIXTURE


def test_infer_output_is_ltwh_int_within_bounds(tmp_path, fixture_images):
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("No RF-DETR checkpoint in weights/base/ — run train.py --smoke first.")

    out = tmp_path / "preds.json"
    subprocess.run(
        [sys.executable, str(SCRIPT),
         "--weights", str(ckpt),
         "--images", str(fixture_images),
         "--out", str(out),
         "--conf", "0.001"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    preds = json.loads(out.read_text())
    assert isinstance(preds, list)

    from PIL import Image
    sizes: dict = {}
    for p in sorted(fixture_images.glob("*.jpg")):
        with Image.open(p) as im:
            sizes[p.name] = im.size  # (w, h)

    for rec in preds:
        assert set(rec) >= {"image_id", "category_id", "bbox", "score"}, rec
        assert isinstance(rec["category_id"], int)
        assert 0 <= rec["category_id"] <= 17
        assert isinstance(rec["score"], float)

        bb = rec["bbox"]
        assert len(bb) == 4
        l, t, w, h = bb
        for v in bb:
            assert isinstance(v, int), f"bbox value {v!r} ({type(v).__name__}) is not int"
        assert l >= 0 and t >= 0
        assert w > 0 and h > 0
        img_w, img_h = sizes[f"{rec['image_id']}.jpg"]
        assert l + w <= img_w, f"box extends past image width: {bb} vs w={img_w}"
        assert t + h <= img_h, f"box extends past image height: {bb} vs h={img_h}"
