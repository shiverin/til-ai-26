"""Pytest configuration for AE training tests."""

import sys
from pathlib import Path

# Make ae/src importable (policy.py, ae_manager.py live there).
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

# Make ae/training importable (train_selfplay.py, critic.py, etc. live here).
_TRAINING = Path(__file__).parent
sys.path.insert(0, str(_TRAINING))
