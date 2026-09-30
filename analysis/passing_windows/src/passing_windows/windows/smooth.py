"""Causal smoothers for Stage 4 cross-fitted score series.

Binding: ``temporal_sampling.smoothing_causality: no_future_frames``.
Only past and current samples may influence the estimate at time t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Stage 4 score columns smoothed for window series (never geometry).
SCORE_COLUMNS = (
    ("q_passability_model", "q_smooth"),
    ("V_catch_model", "V_catch_smooth"),
    ("Q_option_model", "Q_smooth"),
    ("NOV_model", "NOV_smooth"),
)

DEFAULT_ALPHA = 0.5


def causal_ewma(x: np.ndarray | list[float], *, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """One-sided EWMA: y[t] = alpha * x[t] + (1-alpha) * y[t-1]; y[0]=x[0].

    NaN inputs are skipped (state held). Future values never enter past estimates.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    if arr.size == 0:
        return out
    state = np.nan
    for i, v in enumerate(arr):
        if not np.isfinite(v):
            out[i] = state
            continue
        if not np.isfinite(state):
            state = float(v)
        else:
            state = float(alpha * v + (1.0 - alpha) * state)
        out[i] = state
    return out


def smooth_score_columns(
    df: pd.DataFrame,
    *,
    alpha: float = DEFAULT_ALPHA,
    group_cols: tuple[str, ...] = ("gameId", "touchId", "candidateId"),
) -> pd.DataFrame:
    """Smooth Stage 4 OOS scores within each candidate-touch series (causal)."""
    out = df.copy()
    missing = [src for src, _ in SCORE_COLUMNS if src not in out.columns]
    if missing:
        raise KeyError(f"smooth_score_columns missing Stage 4 score columns: {missing}")

    # Prefer 5 Hz rows for the series; callers typically filter first.
    sort_cols = [c for c in (*group_cols, "frameIdx") if c in out.columns]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    for src, dst in SCORE_COLUMNS:
        smoothed = np.full(len(out), np.nan, dtype=float)
        if group_cols and all(c in out.columns for c in group_cols):
            for _, idx in out.groupby(list(group_cols), sort=False).groups.items():
                ix = list(idx)
                smoothed[ix] = causal_ewma(out.loc[ix, src].to_numpy(dtype=float), alpha=alpha)
        else:
            smoothed = causal_ewma(out[src].to_numpy(dtype=float), alpha=alpha)
        out[dst] = smoothed
    return out
