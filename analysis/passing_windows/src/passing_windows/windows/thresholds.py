"""Fold-internal open/close/late-delta threshold selection (nested LOGO)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Locked MVP procedure (documented in stage5_fold_thresholds.json).
PROCEDURE_ID = "train_release_q_median_close_delta_0p05_late_nov_p25"
OPEN_COMPLETION_TARGET = 0.50  # median of train release∩eligible q as open thr proxy
CLOSE_DELTA = 0.05
LATE_NOV_QUANTILE = 0.25
LATE_AFTER_CLOSE_EPSILON_S = 0.20
ROBUSTNESS_OPEN_DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)


@dataclass(frozen=True)
class ThresholdSelection:
    fold_id: str
    held_out_game_id: int
    train_game_ids: list[int]
    open_threshold: float
    close_threshold: float
    late_use_material_loss_delta: float
    late_after_close_epsilon_s: float = LATE_AFTER_CLOSE_EPSILON_S
    procedure: str = PROCEDURE_ID
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.close_threshold > self.open_threshold:
            raise ValueError(
                f"close_threshold ({self.close_threshold}) must be <= open_threshold ({self.open_threshold})"
            )
        if int(self.held_out_game_id) in {int(g) for g in self.train_game_ids}:
            raise ValueError("held_out_game_id must not appear in train_game_ids")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["train_game_ids"] = [int(g) for g in self.train_game_ids]
        d["held_out_game_id"] = int(self.held_out_game_id)
        return d

    def robustness_grid(self) -> list[dict[str, float]]:
        grid = []
        for dlt in ROBUSTNESS_OPEN_DELTAS:
            open_t = float(np.clip(self.open_threshold + dlt, 0.0, 1.0))
            close_t = float(min(open_t, max(0.0, open_t - CLOSE_DELTA)))
            grid.append({"open_threshold": open_t, "close_threshold": close_t, "delta_from_primary": float(dlt)})
        return grid


def select_thresholds_logo(
    predictions: pd.DataFrame,
    *,
    train_game_ids: list[int],
    held_out_game_id: int,
    fold_id: str,
) -> ThresholdSelection:
    """Select open/close/late delta using **train games only**.

    Procedure (MVP, locked):
    1. Among train ``is_pass_release_frame ∩ completion_label_eligible`` rows,
       take the median of ``q_passability_model`` as ``open_threshold``
       (clipped to [0.05, 0.95]).
    2. ``close_threshold = open_threshold - 0.05`` (floored at 0).
    3. ``late_use_material_loss_delta`` = max(0.01, 25th percentile of positive
       train ``NOV_model`` on the same release-eligible rows), else 0.05 fallback.
    """
    train_ids = [int(g) for g in train_game_ids]
    held = int(held_out_game_id)
    if held in train_ids:
        raise ValueError(f"held_out_game_id {held} must not be in train_game_ids {train_ids}")

    train = predictions[predictions["gameId"].isin(train_ids)].copy()
    if train.empty:
        raise ValueError(f"{fold_id}: no train rows for games {train_ids}")

    q_col = "q_passability_model" if "q_passability_model" in train.columns else "q_smooth"
    nov_col = "NOV_model" if "NOV_model" in train.columns else "NOV_smooth"

    release = train[
        train.get("is_pass_release_frame", pd.Series(False, index=train.index)).fillna(False).astype(bool)
        & train.get("completion_label_eligible", pd.Series(False, index=train.index)).fillna(False).astype(bool)
    ]
    q_vals = release[q_col].to_numpy(dtype=float) if len(release) else np.array([], dtype=float)
    q_vals = q_vals[np.isfinite(q_vals)]
    if q_vals.size:
        open_thr = float(np.clip(np.median(q_vals), 0.05, 0.95))
    else:
        # Fallback: median of train 5 Hz q
        series = train[train.get("is_model_5hz_frame", pd.Series(True, index=train.index)).fillna(False).astype(bool)]
        q5 = series[q_col].to_numpy(dtype=float)
        q5 = q5[np.isfinite(q5)]
        open_thr = float(np.clip(np.median(q5) if q5.size else 0.5, 0.05, 0.95))

    close_thr = float(max(0.0, open_thr - CLOSE_DELTA))

    nov_vals = release[nov_col].to_numpy(dtype=float) if len(release) else np.array([], dtype=float)
    nov_pos = nov_vals[np.isfinite(nov_vals) & (nov_vals > 0)]
    if nov_pos.size:
        late_delta = float(max(0.01, np.quantile(nov_pos, LATE_NOV_QUANTILE)))
    else:
        late_delta = 0.05

    diag = {
        "n_train_rows": int(len(train)),
        "n_release_eligible": int(len(release)),
        "n_q_vals": int(q_vals.size),
        "open_completion_target": OPEN_COMPLETION_TARGET,
        "close_delta": CLOSE_DELTA,
        "q_col": q_col,
        "nov_col": nov_col,
    }
    return ThresholdSelection(
        fold_id=fold_id,
        held_out_game_id=held,
        train_game_ids=train_ids,
        open_threshold=open_thr,
        close_threshold=close_thr,
        late_use_material_loss_delta=late_delta,
        late_after_close_epsilon_s=LATE_AFTER_CLOSE_EPSILON_S,
        procedure=PROCEDURE_ID,
        diagnostics=diag,
    )


def apply_thresholds_to_holdout(holdout: pd.DataFrame, selection: ThresholdSelection) -> ThresholdSelection:
    """Return the stored selection for holdout application (no recompute)."""
    if holdout.empty:
        return selection
    gids = set(int(g) for g in holdout["gameId"].unique())
    if gids and gids != {int(selection.held_out_game_id)}:
        # Allow multi-candidate rows of the single held-out game only.
        if any(g != int(selection.held_out_game_id) for g in gids):
            raise ValueError(
                f"holdout games {gids} do not match selection held_out {selection.held_out_game_id}"
            )
    return selection
