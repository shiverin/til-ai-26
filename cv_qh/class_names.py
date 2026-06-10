"""Canonical 18-class id→name map for TIL Novice CV.

Verified against /home/jupyter/novice/cv/annotations.json on 2026-05-23.
Order matches the source COCO categories array (id == index).
"""
from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = (
    "cargo aircraft",      # 0
    "commercial aircraft", # 1
    "drone",               # 2
    "fighter jet",         # 3
    "fighter plane",       # 4
    "helicopter",          # 5
    "light aircraft",      # 6
    "missile",             # 7
    "truck",               # 8
    "car",                 # 9
    "tank",                # 10
    "bus",                 # 11
    "van",                 # 12
    "cargo ship",          # 13
    "yacht",               # 14
    "cruise ship",         # 15
    "warship",             # 16
    "sailboat",            # 17
)

assert len(CLASS_NAMES) == 18
