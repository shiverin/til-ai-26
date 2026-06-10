"""Vote-merge ensemble for K-of-N agreement under the score=1.0 harness.

Every per-model false positive is dropped unless ≥K models hallucinate the
same box at the same class. Reduces FPs precisely where the harness punishes
them the most.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _iou_xyxy(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    bb = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Indexed:
    model_idx: int
    xyxy: tuple[float, float, float, float]
    category_id: int


class EnsembleMerger:
    """Vote-merge: per-class IoU clustering, emit clusters with ≥K distinct
    model contributors. Each model contributes at most one vote per cluster
    (the first overlapping box from that model in iteration order — for
    ensemble use where same-model duplicates are rare and near-identical,
    this is equivalent to highest-IoU in practice)."""

    def __init__(self, k: int, iou_threshold: float = 0.5) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.iou_threshold = iou_threshold

    def merge(self, per_model: Sequence[Sequence[dict]]) -> list[dict]:
        """`per_model[i]` is model i's predictions for one image. Each prediction
        is a dict with keys `xyxy` (tuple of 4 floats) and `category_id` (int).
        Returns the merged emission list in the same dict shape."""
        flat: list[_Indexed] = []
        for mi, preds in enumerate(per_model):
            for p in preds:
                flat.append(_Indexed(mi, tuple(p["xyxy"]), int(p["category_id"])))

        out: list[dict] = []
        for cid in sorted({b.category_id for b in flat}):
            class_boxes = [b for b in flat if b.category_id == cid]
            out.extend(self._cluster_within_class(class_boxes))
        return out

    def _cluster_within_class(self, boxes: list[_Indexed]) -> list[dict]:
        """Greedy clustering: largest cluster first, then remaining."""
        remaining = list(boxes)
        emitted: list[dict] = []
        while remaining:
            # Build candidate clusters seeded at each remaining box; pick the
            # seed yielding the largest distinct-model cluster, tie-break by
            # box index for determinism.
            best_members: list[_Indexed] = []
            best_models: set[int] = set()
            for seed in remaining:
                members = [seed]
                models = {seed.model_idx}
                for other in remaining:
                    if other is seed:
                        continue
                    if other.model_idx in models:
                        continue
                    if _iou_xyxy(seed.xyxy, other.xyxy) >= self.iou_threshold:
                        members.append(other)
                        models.add(other.model_idx)
                if len(models) > len(best_models):
                    best_members = members
                    best_models = models
            if len(best_models) >= self.k:
                cid = best_members[0].category_id
                xs1 = sum(m.xyxy[0] for m in best_members) / len(best_members)
                ys1 = sum(m.xyxy[1] for m in best_members) / len(best_members)
                xs2 = sum(m.xyxy[2] for m in best_members) / len(best_members)
                ys2 = sum(m.xyxy[3] for m in best_members) / len(best_members)
                emitted.append({"xyxy": (xs1, ys1, xs2, ys2),
                                "category_id": cid})
            # Remove all members of this cluster from the pool, regardless of
            # whether it was emitted — they cannot vote again.
            remaining = [b for b in remaining if b not in best_members]
        return emitted
