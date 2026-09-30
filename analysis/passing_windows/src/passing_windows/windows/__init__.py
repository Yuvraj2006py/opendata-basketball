"""Stage 5 window segmentation package.

Causal smoothing, LOGO fold-internal thresholds, hysteresis+persistence
segmentation, window labels, and option-set frame features.
"""

from __future__ import annotations

from .labels import WINDOW_LABELS, assign_touch_summary_label, label_windows
from .option_set import compute_option_set_features
from .segment import (
    MODEL_DT_SECONDS,
    MIN_PERSISTENCE_SECONDS,
    open_signal,
    segment_candidate_series,
)
from .smooth import causal_ewma, smooth_score_columns
from .thresholds import ThresholdSelection, apply_thresholds_to_holdout, select_thresholds_logo

__all__ = [
    "WINDOW_LABELS",
    "MODEL_DT_SECONDS",
    "MIN_PERSISTENCE_SECONDS",
    "ThresholdSelection",
    "apply_thresholds_to_holdout",
    "assign_touch_summary_label",
    "causal_ewma",
    "compute_option_set_features",
    "label_windows",
    "open_signal",
    "segment_candidate_series",
    "select_thresholds_logo",
    "smooth_score_columns",
]
