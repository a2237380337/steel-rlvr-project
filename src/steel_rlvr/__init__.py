"""Qwen3.5 tail-aware RLVR for low-frequency steel leveling."""

import os

# AMD's WSL ROCm runtime needs DXG discovery enabled before torch is imported.
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")

from .reward import parse_leveling_prediction, tail_weight
from .split import split_grade_names

__all__ = ["parse_leveling_prediction", "split_grade_names", "tail_weight"]
__version__ = "0.2.0"
