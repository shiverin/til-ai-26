"""CLASS_NAMES must match the source COCO annotations exactly."""
from __future__ import annotations

import json
from pathlib import Path

from class_names import CLASS_NAMES

SOURCE = Path("/home/jupyter/novice/cv/annotations.json")


def test_class_names_match_source():
    cats = json.loads(SOURCE.read_text())["categories"]
    by_id = {c["id"]: c["name"] for c in cats}
    assert set(by_id) == set(range(18)), f"Expected ids 0..17, got {sorted(by_id)}"
    for cid, name in by_id.items():
        assert CLASS_NAMES[cid] == name, f"id={cid}: src={name!r} vs CLASS_NAMES={CLASS_NAMES[cid]!r}"
