"""Pytest path setup: makes `src.*` and the finetune modules importable."""
import sys
from pathlib import Path

CV = Path(__file__).resolve().parent
for p in (str(CV), str(CV / "finetune")):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: requires a GPU and the trained weights/data on disk"
    )
