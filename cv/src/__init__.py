# Submodules are imported lazily (cv_server instantiates CVManager at module
# level, which requires weights/GPU — importing it here would break unit tests).
from . import cv_manager
