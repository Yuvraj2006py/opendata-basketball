"""Keep-state baseline V_keep(t) from matched no-pass states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import KEEP_STATE_FEATURES, make_regressor_pipeline, matrix_from_frame
from .metrics import regression_metrics
from .outcomes import attach_next3s_outcomes


@dataclass
class KeepStateFoldFit:
    fold_id: str
    held_out_gameId: int
    pipeline: Any
    train_metrics: dict[str, float] = field(default_factory=dict)
    feature_names: tuple[str, ...] = KEEP_STATE_FEATURES
    n_train: int = 0

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df.loc[:, list(self.feature_names)]
        return np.clip(self.pipeline.predict(X), 0.0, 4.0)


def _no_pass_frame_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, touch, frame) that is not a pass-release state."""
    mask = ~candidates["is_pass_release_frame"].fillna(False).astype(bool)
    # Prefer model 5 Hz frames for keep-state matching.
    if "is_model_5hz_frame" in candidates.columns:
        mask &= candidates["is_model_5hz_frame"].fillna(False).astype(bool)
    sub = candidates.loc[mask]
    if sub.empty:
        return sub
    # One candidate row per frame is enough (features are passer-level).
    return sub.drop_duplicates(subset=["gameId", "touchId", "frameIdx"], keep="first").copy()


def fit_keep_state(
    train_candidates: pd.DataFrame,
    shots: pd.DataFrame,
    free_throws: pd.DataFrame,
    *,
    fold_id: str,
    held_out_gameId: int,
    alpha: float = 1.0,
    max_train_rows: int | None = 60_000,
    random_state: int = 0,
    events_index=None,
) -> KeepStateFoldFit:
    frames = _no_pass_frame_rows(train_candidates)
    if frames.empty:
        raise ValueError(f"{fold_id}: no no-pass frames for V_keep training")
    if "keep_points_next_3s" in frames.columns:
        enriched = frames
    else:
        if max_train_rows is not None and len(frames) > max_train_rows:
            frames = frames.sample(n=max_train_rows, random_state=random_state)
        enriched = attach_next3s_outcomes(
            frames,
            shots,
            free_throws,
            use_projected_catch=False,
            events_index=events_index,
        )
    if max_train_rows is not None and len(enriched) > max_train_rows:
        enriched = enriched.sample(n=max_train_rows, random_state=random_state)
    y = enriched["keep_points_next_3s"].to_numpy(dtype=float)
    feats = KEEP_STATE_FEATURES
    matrix_from_frame(enriched, feats)
    X = enriched.loc[:, list(feats)]
    pipe = make_regressor_pipeline(feats, alpha=alpha, use_splines=True)
    pipe.fit(X, y)
    pred = pipe.predict(X)
    return KeepStateFoldFit(
        fold_id=fold_id,
        held_out_gameId=held_out_gameId,
        pipeline=pipe,
        train_metrics=regression_metrics(y, pred),
        feature_names=feats,
        n_train=int(len(y)),
    )


def predict_keep_state(df: pd.DataFrame, fit: KeepStateFoldFit) -> pd.Series:
    """Predict V_keep per candidate row (constant across candidates at a frame)."""
    return pd.Series(fit.predict(df), index=df.index, name="V_keep_model")
