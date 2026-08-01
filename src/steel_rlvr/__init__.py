"""Qwen3.5 frequency-aware DPO for low-frequency steel leveling."""

import os

# AMD's WSL ROCm runtime needs DXG discovery enabled before torch is imported.
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")

from .output_parsing import parse_leveling_prediction
from .preference import tail_sampling_weight
from .split import split_grade_names

__all__ = ["parse_leveling_prediction", "split_grade_names", "tail_sampling_weight"]
__version__ = "0.3.0"
