"""Pure-function detection, classification, and cleaning for LS observations.

All public functions are stateless; I/O is handled by 03b_clean_ls.py.

Public API:
    compute_robust_sigma     — MAD-based σ of day-to-day diffs
    detect_jumps             — flag points exceeding n_sigma threshold
    classify_jump            — 5-branch decision tree per spec §3.3
    clean_station            — iterative orchestrator
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


def compute_robust_sigma(z: pd.Series) -> float:
    """Median-absolute-deviation σ of z.diff(), robust to outliers in z.

    Returns σ in the same units as z.
    """
    dz = z.diff().dropna()
    if dz.empty:
        return 0.0
    median_dz = dz.median()
    mad = (dz - median_dz).abs().median()
    return float(MAD_TO_SIGMA * mad)
