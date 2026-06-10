"""Pytest config: add cv_qh root to sys.path so tests can import siblings."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
