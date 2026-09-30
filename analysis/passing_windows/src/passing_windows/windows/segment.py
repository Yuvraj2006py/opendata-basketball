"""Hysteresis + persistence window segmentation on 5 Hz series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Binding freeze interpretation (auditor preferred):
# model_frame_stride=5 at 25 Hz → Δt = 0.20 s between consecutive model frames.
# min_persistence_seconds=0.20 ≡ require ≥2 consecutive True samples before commit.
MODEL_DT_SECONDS = 0.20
MIN_PERSISTENCE_SECONDS = 0.20
MIN_PERSISTENCE_SAMPLES = 2  # consecutive 5 Hz samples spanning ≥0.20 s
RENDER_HZ = 25.0


def frame_to_time_s(frame_idx: int | float) -> float:
    return float(frame_idx) / RENDER_HZ


def open_signal(
    df: pd.DataFrame,
    *,
    open_threshold: float,
    q_col: str = "q_smooth",
    nov_col: str = "NOV_smooth",
    rejected_col: str = "rejected_outside_support",
) -> pd.Series:
    """Raw open-signal: q ≥ open_thr AND NOV > 0 AND not rejected AND finite."""
    q = df[q_col].to_numpy(dtype=float)
    nov = df[nov_col].to_numpy(dtype=float)
    if rejected_col in df.columns:
        rejected = df[rejected_col].fillna(False).astype(bool).to_numpy()
    else:
        rejected = np.zeros(len(df), dtype=bool)
    ok = (
        np.isfinite(q)
        & np.isfinite(nov)
        & (q >= float(open_threshold))
        & (nov > 0.0)
        & (~rejected)
    )
    return pd.Series(ok, index=df.index)


def remain_open_signal(
    df: pd.DataFrame,
    *,
    close_threshold: float,
    q_col: str = "q_smooth",
    nov_col: str = "NOV_smooth",
    rejected_col: str = "rejected_outside_support",
) -> pd.Series:
    """Remain-open while q ≥ close_thr AND NOV > 0 AND not rejected."""
    q = df[q_col].to_numpy(dtype=float)
    nov = df[nov_col].to_numpy(dtype=float)
    if rejected_col in df.columns:
        rejected = df[rejected_col].fillna(False).astype(bool).to_numpy()
    else:
        rejected = np.zeros(len(df), dtype=bool)
    ok = (
        np.isfinite(q)
        & np.isfinite(nov)
        & (q >= float(close_threshold))
        & (nov > 0.0)
        & (~rejected)
    )
    return pd.Series(ok, index=df.index)


def _commit_run(flags: np.ndarray, need: int = MIN_PERSISTENCE_SAMPLES) -> np.ndarray:
    """Return boolean array True where a run of ``need`` consecutive Trues has been reached."""
    out = np.zeros(len(flags), dtype=bool)
    run = 0
    for i, v in enumerate(flags):
        if v:
            run += 1
            if run >= need:
                out[i] = True
        else:
            run = 0
    return out


def segment_one_series(
    frames: np.ndarray,
    enter: np.ndarray,
    remain: np.ndarray,
    nov: np.ndarray,
    *,
    open_threshold: float,
    close_threshold: float,
) -> list[dict[str, Any]]:
    """State machine on one sorted candidate-touch 5 Hz series."""
    n = len(frames)
    if n == 0:
        return []
    windows: list[dict[str, Any]] = []
    state = "closed"
    enter_run = 0
    exit_run = 0
    open_start_i: int | None = None
    committed_open = False

    def _emit(start_i: int, end_i: int, *, censored: bool) -> None:
        # Inclusive indices of committed open span
        span_nov = nov[start_i : end_i + 1]
        finite = np.isfinite(span_nov)
        if finite.any():
            peak_local = int(np.argmax(np.where(finite, span_nov, -np.inf)))
            peak_i = start_i + peak_local
            peak_nov = float(span_nov[peak_local])
        else:
            peak_i = start_i
            peak_nov = float("nan")
        # Trapezoid integrate NOV on 5 Hz (Δt = MODEL_DT_SECONDS)
        if end_i > start_i:
            y = np.nan_to_num(span_nov, nan=0.0)
            trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
            integrated = float(trap(y, dx=MODEL_DT_SECONDS))
        else:
            integrated = 0.0
        opening_f = int(frames[start_i])
        closing_f = int(frames[end_i])
        windows.append(
            {
                "opening_frameIdx": opening_f,
                "peak_frameIdx": int(frames[peak_i]),
                "closing_frameIdx": closing_f,
                "opening_time_s": frame_to_time_s(opening_f),
                "peak_time_s": frame_to_time_s(frames[peak_i]),
                "closing_time_s": frame_to_time_s(closing_f),
                "duration_s": frame_to_time_s(closing_f) - frame_to_time_s(opening_f),
                "peak_NOV": peak_nov,
                "integrated_NOV": integrated,
                "censored_at_touch_end": bool(censored),
                "open_threshold": float(open_threshold),
                "close_threshold": float(close_threshold),
                "existence_prob_status": "deferred_stage7",
                "window_existence_probability_under_tracking_perturbation": float("nan"),
            }
        )

    for i in range(n):
        if state == "closed":
            if enter[i]:
                enter_run += 1
            else:
                enter_run = 0
            if enter_run >= MIN_PERSISTENCE_SAMPLES:
                # Opening commits at the first sample of the qualifying run
                open_start_i = i - (MIN_PERSISTENCE_SAMPLES - 1)
                state = "open"
                committed_open = True
                exit_run = 0
        else:  # open
            if remain[i]:
                exit_run = 0
            else:
                exit_run += 1
            if exit_run >= MIN_PERSISTENCE_SAMPLES:
                # Close commits at last remain-True index before exit run
                end_i = i - MIN_PERSISTENCE_SAMPLES
                if end_i < open_start_i:  # type: ignore[operator]
                    end_i = open_start_i  # type: ignore[assignment]
                _emit(open_start_i, end_i, censored=False)  # type: ignore[arg-type]
                state = "closed"
                open_start_i = None
                committed_open = False
                enter_run = 1 if enter[i] else 0

    if state == "open" and open_start_i is not None and committed_open:
        _emit(open_start_i, n - 1, censored=True)
    return windows


def segment_candidate_series(
    df: pd.DataFrame,
    *,
    open_threshold: float,
    close_threshold: float,
    q_col: str = "q_smooth",
    nov_col: str = "NOV_smooth",
) -> pd.DataFrame:
    """Segment windows for all candidate-touch groups in ``df``.

    Caller must pass **model 5 Hz rows only** (``is_model_5hz_frame``).
    """
    if close_threshold > open_threshold:
        raise ValueError("close_threshold must be <= open_threshold")

    work = df.copy()
    if "is_model_5hz_frame" in work.columns:
        work = work[work["is_model_5hz_frame"].fillna(False).astype(bool)].copy()

    group_cols = [c for c in ("gameId", "touchId", "candidateId") if c in work.columns]
    rows: list[dict[str, Any]] = []

    if work.empty:
        return pd.DataFrame(rows)

    enter_all = open_signal(work, open_threshold=open_threshold, q_col=q_col, nov_col=nov_col)
    remain_all = remain_open_signal(work, close_threshold=close_threshold, q_col=q_col, nov_col=nov_col)

    for keys, grp in work.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        g = grp.sort_values("frameIdx", kind="mergesort")
        frames = g["frameIdx"].to_numpy(dtype=int)
        enter = enter_all.loc[g.index].to_numpy(dtype=bool)
        remain = remain_all.loc[g.index].to_numpy(dtype=bool)
        nov = g[nov_col].to_numpy(dtype=float)
        for w in segment_one_series(
            frames,
            enter,
            remain,
            nov,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
        ):
            w.update(key_map)
            for meta in ("fold_id", "model_version", "sampling_role"):
                if meta in g.columns:
                    w[meta] = g.iloc[0][meta]
            rows.append(w)

    out = pd.DataFrame(rows)
    if len(out):
        out["window_id"] = [
            f"{r.gameId}|{r.touchId}|{r.candidateId}|{int(r.opening_frameIdx)}"
            for r in out.itertuples(index=False)
        ]
    return out


def mark_frame_open_state(
    series: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``is_open_signal`` and ``is_window_open`` flags to the 5 Hz series."""
    out = series.copy()
    if "open_threshold" in windows.columns and len(windows):
        # Use per-row thresholds from first window of same fold if present
        open_thr = float(windows.iloc[0]["open_threshold"])
    else:
        open_thr = 0.5
    out["is_open_signal"] = open_signal(out, open_threshold=open_thr).to_numpy()
    out["is_window_open"] = False
    if windows is None or windows.empty:
        return out
    for w in windows.itertuples(index=False):
        mask = (
            (out["gameId"] == w.gameId)
            & (out["touchId"] == w.touchId)
            & (out["candidateId"] == w.candidateId)
            & (out["frameIdx"] >= w.opening_frameIdx)
            & (out["frameIdx"] <= w.closing_frameIdx)
        )
        out.loc[mask, "is_window_open"] = True
    return out
