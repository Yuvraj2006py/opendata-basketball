"""Per-frame option-set features on the 5 Hz lattice."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .segment import MODEL_DT_SECONDS, frame_to_time_s


def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    m = float(np.max(x))
    return float(m + np.log(np.sum(np.exp(x - m))))


def _entropy_softmax(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    m = float(np.max(x))
    e = np.exp(x - m)
    p = e / e.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_option_set_features(
    series: pd.DataFrame,
    *,
    open_threshold: float = 0.5,
) -> pd.DataFrame:
    """Grain: (gameId, touchId, frameIdx) on model 5 Hz rows.

    Viable options exclude ``rejected_outside_support``. Soft total uses
    ``logsumexp(Q_smooth)`` among non-rejected candidates.
    """
    if series.empty:
        return pd.DataFrame()

    work = series.copy()
    if "is_model_5hz_frame" in work.columns:
        work = work[work["is_model_5hz_frame"].fillna(False).astype(bool)].copy()

    if "rejected_outside_support" not in work.columns:
        work["rejected_outside_support"] = False
    if "is_open_signal" not in work.columns:
        q = work["q_smooth"].to_numpy(dtype=float)
        nov = work["NOV_smooth"].to_numpy(dtype=float)
        rej = work["rejected_outside_support"].fillna(False).astype(bool).to_numpy()
        work["is_open_signal"] = (
            np.isfinite(q) & np.isfinite(nov) & (q >= open_threshold) & (nov > 0) & (~rej)
        )

    rows: list[dict] = []
    group_cols = ["gameId", "touchId", "frameIdx"]
    for keys, grp in work.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        game_id, touch_id, frame_idx = keys
        viable = grp[~grp["rejected_outside_support"].fillna(False).astype(bool)]
        nov = viable["NOV_smooth"].to_numpy(dtype=float) if len(viable) else np.array([])
        q_smooth = viable["Q_smooth"].to_numpy(dtype=float) if "Q_smooth" in viable.columns and len(viable) else nov

        finite_nov = nov[np.isfinite(nov)]
        if finite_nov.size:
            order = np.argsort(finite_nov)[::-1]
            best = float(finite_nov[order[0]])
            second = float(finite_nov[order[1]]) if finite_nov.size >= 2 else float("nan")
            gap = best - second if finite_nov.size >= 2 else float("nan")
        else:
            best = second = gap = float("nan")

        n_viable = int(viable["is_open_signal"].fillna(False).astype(bool).sum()) if len(viable) else 0
        soft = _logsumexp(q_smooth)
        entropy = _entropy_softmax(nov if nov.size else q_smooth)

        row = {
            "gameId": game_id,
            "touchId": touch_id,
            "frameIdx": int(frame_idx),
            "frame_time_s": frame_to_time_s(frame_idx),
            "best_option_value": best,
            "n_viable_options": n_viable,
            "soft_total_option_value": soft,
            "best_second_gap": gap,
            "option_entropy": entropy,
            "open_threshold_ref": float(open_threshold),
        }
        for meta in ("fold_id", "model_version"):
            if meta in grp.columns:
                row[meta] = grp.iloc[0][meta]
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Causal accumulated area of best NOV within touch
    out = out.sort_values(["gameId", "touchId", "frameIdx"], kind="mergesort")
    acc = []
    open_rate = []
    close_rate = []
    for (_, _), g in out.groupby(["gameId", "touchId"], sort=False):
        bests = g["best_option_value"].to_numpy(dtype=float)
        running = 0.0
        prev_n = 0
        for i, b in enumerate(bests):
            if np.isfinite(b):
                running += float(b) * MODEL_DT_SECONDS
            acc.append(running)
            # crude open/close event from n_viable changes (frame-level)
            n = int(g.iloc[i]["n_viable_options"])
            open_rate.append(1.0 if n > prev_n else 0.0)
            close_rate.append(1.0 if n < prev_n else 0.0)
            prev_n = n
    out["accumulated_option_value_area"] = acc
    out["opening_rate"] = open_rate
    out["closing_rate"] = close_rate
    return out.reset_index(drop=True)
