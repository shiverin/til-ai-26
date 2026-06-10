"""Unit tests for the vote-merge ensemble logic."""
import pytest

from src.ensemble import EnsembleMerger


def _box(x1, y1, x2, y2, cls):
    return {"xyxy": (float(x1), float(y1), float(x2), float(y2)),
            "category_id": int(cls)}


def test_k3_emits_when_three_models_agree():
    """3 models emit the same box → cluster size 3 ≥ K=3 → emitted."""
    merger = EnsembleMerger(k=3, iou_threshold=0.5)
    per_model = [
        [_box(10, 10, 50, 50, 7)],  # model 0
        [_box(11, 11, 51, 51, 7)],  # model 1
        [_box(12, 12, 52, 52, 7)],  # model 2
        [],                          # model 3 — no detection
        [],                          # model 4 — no detection
    ]
    out = merger.merge(per_model)
    assert len(out) == 1
    assert out[0]["category_id"] == 7
    # fused box ≈ mean of three near-identical boxes
    x1, y1, x2, y2 = out[0]["xyxy"]
    assert abs(x1 - 11.0) < 0.5
    assert abs(y2 - 51.0) < 0.5


def test_k3_drops_when_only_two_agree():
    """2 models agree → cluster size 2 < K=3 → suppressed."""
    merger = EnsembleMerger(k=3, iou_threshold=0.5)
    per_model = [
        [_box(10, 10, 50, 50, 7)],
        [_box(11, 11, 51, 51, 7)],
        [], [], [],
    ]
    assert merger.merge(per_model) == []


def test_different_classes_do_not_cluster():
    """3 boxes in same location but different classes → no cluster reaches K."""
    merger = EnsembleMerger(k=3, iou_threshold=0.5)
    per_model = [
        [_box(10, 10, 50, 50, 7)],
        [_box(10, 10, 50, 50, 8)],
        [_box(10, 10, 50, 50, 9)],
        [], [],
    ]
    assert merger.merge(per_model) == []


def test_low_iou_does_not_cluster():
    """3 same-class boxes far apart → not in one cluster."""
    merger = EnsembleMerger(k=3, iou_threshold=0.5)
    per_model = [
        [_box(10, 10, 50, 50, 7)],
        [_box(500, 500, 540, 540, 7)],
        [_box(1000, 1000, 1040, 1040, 7)],
        [], [],
    ]
    assert merger.merge(per_model) == []


def test_multiple_clusters_in_one_class():
    """Two separate spatial clusters, both ≥ K → emit both."""
    merger = EnsembleMerger(k=3, iou_threshold=0.5)
    per_model = [
        [_box(10, 10, 50, 50, 7), _box(500, 500, 540, 540, 7)],
        [_box(11, 11, 51, 51, 7), _box(501, 501, 541, 541, 7)],
        [_box(12, 12, 52, 52, 7), _box(502, 502, 542, 542, 7)],
        [], [],
    ]
    out = merger.merge(per_model)
    assert len(out) == 2
    # ordered by some stable key — just check both present
    centers = sorted(((b["xyxy"][0] + b["xyxy"][2]) / 2 for b in out))
    assert abs(centers[0] - 31.0) < 1.0
    assert abs(centers[1] - 521.0) < 1.0


def test_one_model_contributes_at_most_one_box_per_cluster():
    """Even if one model emits two near-duplicates, that's still one vote."""
    merger = EnsembleMerger(k=3, iou_threshold=0.5)
    per_model = [
        [_box(10, 10, 50, 50, 7), _box(11, 11, 51, 51, 7)],  # one model, two boxes
        [_box(10, 10, 50, 50, 7)],
        [], [], [],
    ]
    # only 2 distinct models contribute → < K=3 → suppressed
    assert merger.merge(per_model) == []


def test_empty_input():
    merger = EnsembleMerger(k=3, iou_threshold=0.5)
    assert merger.merge([[], [], [], [], []]) == []
