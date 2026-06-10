"""prepare_data.py must produce a deterministic, non-overlapping, schema-valid split."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "prepare_data.py"
SOURCE_IMAGES = Path("/home/jupyter/novice/cv/images")
DATASET = ROOT / "dataset"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


@pytest.fixture(autouse=True)
def _require_source():
    if not SOURCE_IMAGES.exists():
        pytest.skip(f"Source images not available at {SOURCE_IMAGES}")


def test_prepare_produces_two_split_dirs():
    _run([])
    assert (DATASET / "train" / "_annotations.coco.json").exists()
    assert (DATASET / "valid" / "_annotations.coco.json").exists()


def test_split_is_deterministic():
    _run([])
    a = json.loads((DATASET / "train" / "_annotations.coco.json").read_text())
    _run([])
    b = json.loads((DATASET / "train" / "_annotations.coco.json").read_text())
    a_ids = sorted(i["id"] for i in a["images"])
    b_ids = sorted(i["id"] for i in b["images"])
    assert a_ids == b_ids


def test_no_image_overlap():
    _run([])
    tr = json.loads((DATASET / "train" / "_annotations.coco.json").read_text())
    va = json.loads((DATASET / "valid" / "_annotations.coco.json").read_text())
    tr_ids = {i["id"] for i in tr["images"]}
    va_ids = {i["id"] for i in va["images"]}
    assert tr_ids.isdisjoint(va_ids)
    assert len(tr_ids) + len(va_ids) == 5000


def test_categories_preserved():
    _run([])
    tr = json.loads((DATASET / "train" / "_annotations.coco.json").read_text())
    va = json.loads((DATASET / "valid" / "_annotations.coco.json").read_text())
    assert tr["categories"] == va["categories"]
    assert {c["id"] for c in tr["categories"]} == set(range(18))


def test_annotations_match_their_split():
    _run([])
    for split in ("train", "valid"):
        d = json.loads((DATASET / split / "_annotations.coco.json").read_text())
        img_ids = {i["id"] for i in d["images"]}
        for ann in d["annotations"]:
            assert ann["image_id"] in img_ids


def test_symlinks_resolve():
    _run([])
    for split in ("train", "valid"):
        d = json.loads((DATASET / split / "_annotations.coco.json").read_text())
        sample = d["images"][:3]
        for img in sample:
            p = DATASET / split / img["file_name"]
            assert p.is_symlink(), f"{p} is not a symlink"
            assert p.resolve().exists(), f"{p} dangling"


def test_idempotent_rerun():
    _run([])
    _run([])  # must not raise
    assert (DATASET / "train" / "_annotations.coco.json").exists()


def test_val_fraction_close_to_10pct():
    _run([])
    va = json.loads((DATASET / "valid" / "_annotations.coco.json").read_text())
    assert 450 <= len(va["images"]) <= 550
