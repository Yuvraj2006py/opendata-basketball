"""Post-catch value model V_catch(j,t)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from passing_windows.velocities import FRAME_RATE_HZ

from .features import CATCH_VALUE_FEATURES, make_regressor_pipeline, matrix_from_frame
from .metrics import regression_metrics
from .outcomes import attach_next3s_outcomes

TrainingPopulation = Literal["observed_catch", "all_candidates"]
OBSERVED_CATCH_HALF_WINDOW_S = 0.4


@dataclass
class CatchValueFoldFit:
    fold_id: str
    held_out_gameId: int
    pipeline: Any
    secondary_pipelines: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    feature_names: tuple[str, ...] = CATCH_VALUE_FEATURES
    n_train: int = 0
    training_population: str = "observed_catch"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df.loc[:, list(self.feature_names)]
        return np.clip(self.pipeline.predict(X), 0.0, 4.0)

    def predict_secondary(self, df: pd.DataFrame, name: str) -> np.ndarray:
        pipe = self.secondary_pipelines[name]
        X = df.loc[:, list(self.feature_names)]
        return np.clip(pipe.predict(X), 0.0, 1.0)


def observed_catch_training_mask(
    df: pd.DataFrame,
    *,
    half_window_s: float = OBSERVED_CATCH_HALF_WINDOW_S,
    frame_rate_hz: float = FRAME_RATE_HZ,
) -> pd.Series:
    """Boolean mask: true-receiver release ∪ nearby 5 Hz true-receiver rows.

    Release anchors require ``completion_label_eligible`` and ``is_true_receiver``.
    Nearby rows require ``is_model_5hz_frame``, same ``(gameId, touchId)``, matching
    ``candidateId``, and ``|frameIdx - release_frameIdx| / frame_rate_hz <= half_window_s``.
    """
    out = pd.Series(False, index=df.index)
    if df.empty:
        return out
    eligible = df["completion_label_eligible"].fillna(False).astype(bool)
    true_rec = df["is_true_receiver"].fillna(False).astype(bool)
    release_true = eligible & true_rec
    out |= release_true
    if not bool(release_true.any()):
        return out

    half_frames = float(half_window_s) * float(frame_rate_hz)
    anchors = (
        df.loc[release_true, ["gameId", "touchId", "candidateId", "frameIdx"]]
        .rename(columns={"frameIdx": "release_frameIdx", "candidateId": "true_candidateId"})
        .drop_duplicates()
    )
    base = df.reset_index()
    merged = base.merge(anchors, on=["gameId", "touchId"], how="inner")
    if merged.empty:
        return out
    is_5hz = merged["is_model_5hz_frame"].fillna(False).astype(bool)
    near = (
        is_5hz
        & (merged["candidateId"] == merged["true_candidateId"])
        & ((merged["frameIdx"] - merged["release_frameIdx"]).abs() <= half_frames)
    )
    near_idx = merged.loc[near, "index"].unique()
    out.loc[near_idx] = True
    return out


def prepare_catch_training(
    candidates: pd.DataFrame,
    shots: pd.DataFrame,
    free_throws: pd.DataFrame,
    *,
    events_index=None,
) -> pd.DataFrame:
    """Attach catch-window outcomes and keep rows with finite catch geometry."""
    if "catch_points_next_3s" in candidates.columns:
        enriched = candidates
    else:
        enriched = attach_next3s_outcomes(
            candidates,
            shots,
            free_throws,
            use_projected_catch=True,
            events_index=events_index,
        )
    mask = np.isfinite(pd.to_numeric(enriched["dist_to_rim_ft"], errors="coerce"))
    mask &= np.isfinite(pd.to_numeric(enriched["catch_x_norm"], errors="coerce"))
    return enriched.loc[mask].copy()


def evaluate_catch_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """RMSE/MAE/R² plus Spearman on finite pairs."""
    metrics = regression_metrics(y_true, y_pred)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if int(ok.sum()) < 2:
        metrics["spearman"] = float("nan")
        return metrics
    metrics["spearman"] = float(pd.Series(y[ok]).corr(pd.Series(p[ok]), method="spearman"))
    return metrics


def fit_catch_value(
    train_candidates: pd.DataFrame,
    shots: pd.DataFrame,
    free_throws: pd.DataFrame,
    *,
    fold_id: str,
    held_out_gameId: int,
    alpha: float = 1.0,
    max_train_rows: int | None = 80_000,
    random_state: int = 0,
    events_index=None,
    training_population: TrainingPopulation = "observed_catch",
    half_window_s: float = OBSERVED_CATCH_HALF_WINDOW_S,
) -> CatchValueFoldFit:
    base = train_candidates
    if training_population == "observed_catch":
        mask = observed_catch_training_mask(base, half_window_s=half_window_s)
        base = base.loc[mask].copy()
        if base.empty:
            raise ValueError(f"{fold_id}: empty observed-catch training mask")
    elif training_population != "all_candidates":
        raise ValueError(f"unknown training_population={training_population!r}")

    # Subsample before expensive outcome attach when outcomes not precomputed.
    if "catch_points_next_3s" not in base.columns and max_train_rows is not None and len(base) > max_train_rows:
        base = base.sample(n=max_train_rows, random_state=random_state)
    train = prepare_catch_training(base, shots, free_throws, events_index=events_index)
    if train.empty:
        raise ValueError(f"{fold_id}: empty catch-value training set")
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=random_state)
    y = train["catch_points_next_3s"].to_numpy(dtype=float)
    feats = CATCH_VALUE_FEATURES
    matrix_from_frame(train, feats)
    X = train.loc[:, list(feats)]
    pipe = make_regressor_pipeline(feats, alpha=alpha, use_splines=True)
    pipe.fit(X, y)
    pred = pipe.predict(X)

    secondary: dict[str, Any] = {}
    for col in ("catch_any_shot_next_3s", "catch_open_shot_next_3s"):
        if col not in train.columns:
            continue
        y2 = train[col].to_numpy(dtype=float)
        # Ridge as linear probability model for secondary rates.
        p2 = make_regressor_pipeline(feats, alpha=alpha, use_splines=False)
        p2.fit(X, y2)
        secondary[col.replace("catch_", "")] = p2

    return CatchValueFoldFit(
        fold_id=fold_id,
        held_out_gameId=held_out_gameId,
        pipeline=pipe,
        secondary_pipelines=secondary,
        train_metrics=regression_metrics(y, pred),
        feature_names=feats,
        n_train=int(len(y)),
        training_population=training_population,
    )


def predict_catch_value(df: pd.DataFrame, fit: CatchValueFoldFit) -> pd.DataFrame:
    out = df[["gameId", "touchId", "frameIdx", "candidateId"]].copy()
    out["V_catch_model"] = fit.predict(df)
    for name in fit.secondary_pipelines:
        out[f"V_catch_sec_{name}"] = fit.predict_secondary(df, name)
    return out
