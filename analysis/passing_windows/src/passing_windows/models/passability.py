"""Passability model q(j,t) with freeze ablations and LOGO cross-fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import ABLATION_NAMES, PRIMARY_ABLATION
from .features import make_classifier_pipeline, matrix_from_frame, passability_feature_names
from .metrics import LogisticCalibrator, binary_metrics, clip_prob


@dataclass
class PassabilityFoldFit:
    fold_id: str
    held_out_gameId: int
    ablation: str
    pipeline: Any
    calibrator: LogisticCalibrator
    train_metrics_raw: dict[str, float] = field(default_factory=dict)
    train_metrics_cal: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_pos: int = 0
    feature_names: tuple[str, ...] = ()

    def predict_proba(self, df: pd.DataFrame, *, calibrate: bool = True) -> np.ndarray:
        X = df.loc[:, list(self.feature_names)]
        raw = self.pipeline.predict_proba(X)[:, 1]
        if calibrate:
            return clip_prob(self.calibrator.transform(raw))
        return clip_prob(raw)


def _balance_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights computed on the training fold only."""
    y = np.asarray(y).astype(int)
    n = len(y)
    if n == 0:
        return y.astype(float)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(n - int(y.sum()), 1)
    w = np.ones(n, dtype=float)
    w[y == 1] = 0.5 * n / n_pos
    w[y == 0] = 0.5 * n / n_neg
    return w


def fit_passability_ablation(
    train_df: pd.DataFrame,
    *,
    fold_id: str,
    held_out_gameId: int,
    ablation: str = PRIMARY_ABLATION,
    C: float = 1.0,
    random_state: int = 0,
) -> PassabilityFoldFit:
    """Fit one ablation on completion_label_eligible training rows."""
    feats = passability_feature_names(ablation)
    labeled = train_df[train_df["completion_label_eligible"].fillna(False).astype(bool)].copy()
    if labeled.empty:
        raise ValueError(f"{fold_id}/{ablation}: no completion_label_eligible training rows")
    y = labeled["is_true_receiver"].fillna(False).astype(int).to_numpy()
    X = labeled.loc[:, list(feats)]
    # Touch matrix_from_frame to enforce forbidden-column guards.
    matrix_from_frame(labeled, feats)
    pipe = make_classifier_pipeline(feats, use_splines=True, C=C, random_state=random_state)
    sw = _balance_sample_weights(y)
    pipe.fit(X, y, clf__sample_weight=sw)
    raw = pipe.predict_proba(X)[:, 1]
    cal = LogisticCalibrator().fit(raw, y)
    cal_p = cal.transform(raw)
    return PassabilityFoldFit(
        fold_id=fold_id,
        held_out_gameId=held_out_gameId,
        ablation=ablation,
        pipeline=pipe,
        calibrator=cal,
        train_metrics_raw=binary_metrics(y, raw),
        train_metrics_cal=binary_metrics(y, cal_p),
        n_train=int(len(y)),
        n_pos=int(y.sum()),
        feature_names=feats,
    )


def fit_all_passability_ablations(
    train_df: pd.DataFrame,
    *,
    fold_id: str,
    held_out_gameId: int,
    ablations: tuple[str, ...] = ABLATION_NAMES,
    random_state: int = 0,
) -> dict[str, PassabilityFoldFit]:
    return {
        name: fit_passability_ablation(
            train_df,
            fold_id=fold_id,
            held_out_gameId=held_out_gameId,
            ablation=name,
            random_state=random_state,
        )
        for name in ablations
    }


def predict_passability_table(
    df: pd.DataFrame,
    fits: dict[str, PassabilityFoldFit],
    *,
    primary: str = PRIMARY_ABLATION,
) -> pd.DataFrame:
    """Wide predictions: primary q plus ablation columns."""
    out = df[["gameId", "touchId", "frameIdx", "candidateId"]].copy()
    for name, fit in fits.items():
        col = "q_passability_model" if name == primary else f"q_ablation_{name}"
        out[col] = fit.predict_proba(df, calibrate=True)
    if primary in fits and "q_passability_model" not in out.columns:
        out["q_passability_model"] = fits[primary].predict_proba(df, calibrate=True)
    return out


def ablation_long_table(
    pred_wide: pd.DataFrame,
    *,
    primary: str = PRIMARY_ABLATION,
) -> pd.DataFrame:
    """Stack ablation q columns into a long table for reporting."""
    keys = ["gameId", "touchId", "frameIdx", "candidateId"]
    rows = []
    primary_col = "q_passability_model"
    if primary_col in pred_wide.columns:
        chunk = pred_wide[keys].copy()
        chunk["ablation"] = primary
        chunk["q"] = pred_wide[primary_col]
        rows.append(chunk)
    for c in pred_wide.columns:
        if c.startswith("q_ablation_"):
            name = c[len("q_ablation_") :]
            chunk = pred_wide[keys].copy()
            chunk["ablation"] = name
            chunk["q"] = pred_wide[c]
            rows.append(chunk)
    if not rows:
        return pd.DataFrame(columns=[*keys, "ablation", "q"])
    return pd.concat(rows, ignore_index=True)
