"""Descriptive choice model at actual pass-release states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

CHOICE_FEATURES: tuple[str, ...] = (
    "q_passability_model",
    "V_catch_model",
    "NOV_model",
    "pass_distance_ft",
    "receiver_sep_ft",
    "dist_to_rim_ft",
    "geometry_passability_score",
)


@dataclass
class ChoiceFoldFit:
    fold_id: str
    held_out_gameId: int
    pipeline_clf: Any
    scaler: StandardScaler
    feature_names: tuple[str, ...] = CHOICE_FEATURES
    train_metrics: dict[str, float] = field(default_factory=dict)
    n_train_releases: int = 0

    def predict_utilities(self, df: pd.DataFrame) -> np.ndarray:
        X = df.loc[:, list(self.feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # Median fill using training medians stored on scaler (via nan replacement).
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
        Xs = self.scaler.transform(X)
        # Decision function as utility.
        if hasattr(self.pipeline_clf, "decision_function"):
            return self.pipeline_clf.decision_function(Xs)
        return self.pipeline_clf.predict_proba(Xs)[:, 1]


def _softmax(u: np.ndarray) -> np.ndarray:
    u = u - np.max(u)
    e = np.exp(u)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def fit_choice_model(
    release_candidates: pd.DataFrame,
    *,
    fold_id: str,
    held_out_gameId: int,
    feature_names: tuple[str, ...] = CHOICE_FEATURES,
    random_state: int = 0,
) -> ChoiceFoldFit:
    """Fit a descriptive logistic utility on completion-eligible release rows.

    Held-out evaluation uses softmax over the four candidates at each release
    (conditional choice approximation).
    """
    labeled = release_candidates[
        release_candidates["completion_label_eligible"].fillna(False).astype(bool)
    ].copy()
    if labeled.empty:
        raise ValueError(f"{fold_id}: no release rows for choice model")
    feats = list(feature_names)
    X = labeled.loc[:, feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = labeled["is_true_receiver"].fillna(False).astype(int).to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=400,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(Xs, y)

    # Train choice metrics via within-release softmax ranking.
    metrics = evaluate_choice(labeled.assign(**{c: labeled[c] for c in feats}), ChoiceFoldFit(
        fold_id=fold_id,
        held_out_gameId=held_out_gameId,
        pipeline_clf=clf,
        scaler=scaler,
        feature_names=tuple(feats),
    ))
    n_rel = labeled.drop_duplicates(subset=["gameId", "touchId", "frameIdx", "passId"], keep="first")
    return ChoiceFoldFit(
        fold_id=fold_id,
        held_out_gameId=held_out_gameId,
        pipeline_clf=clf,
        scaler=scaler,
        feature_names=tuple(feats),
        train_metrics=metrics,
        n_train_releases=int(len(n_rel)),
    )


def evaluate_choice(df: pd.DataFrame, fit: ChoiceFoldFit) -> dict[str, float]:
    """Rank accuracy + choice log loss over 4-candidate release sets."""
    labeled = df[df["completion_label_eligible"].fillna(False).astype(bool)].copy()
    if labeled.empty:
        return {"n_releases": 0.0, "rank_accuracy": float("nan"), "choice_log_loss": float("nan")}
    utilities = fit.predict_utilities(labeled)
    labeled = labeled.copy()
    labeled["_u"] = utilities
    group_cols = ["gameId", "touchId", "frameIdx"]
    if "passId" in labeled.columns:
        group_cols = ["gameId", "passId"]

    n_ok = 0
    n_groups = 0
    losses: list[float] = []
    for _, grp in labeled.groupby(group_cols, sort=False):
        if len(grp) < 2:
            continue
        n_groups += 1
        u = grp["_u"].to_numpy(dtype=float)
        y = grp["is_true_receiver"].fillna(False).astype(int).to_numpy()
        if y.sum() != 1:
            continue
        probs = _softmax(u)
        pred_idx = int(np.argmax(u))
        true_idx = int(np.argmax(y))
        if pred_idx == true_idx:
            n_ok += 1
        # Choice log loss for the true alternative.
        p_true = float(probs[true_idx])
        losses.append(-np.log(max(p_true, 1e-12)))

    return {
        "n_releases": float(n_groups),
        "rank_accuracy": float(n_ok / n_groups) if n_groups else float("nan"),
        "choice_log_loss": float(np.mean(losses)) if losses else float("nan"),
        # Also report binary log_loss on independent logits for diagnostics.
        "binary_log_loss": float(
            log_loss(
                labeled["is_true_receiver"].fillna(False).astype(int),
                np.clip(
                    1.0 / (1.0 + np.exp(-utilities)),
                    1e-6,
                    1 - 1e-6,
                ),
                labels=[0, 1],
            )
        )
        if len(labeled) and labeled["is_true_receiver"].nunique() > 1
        else float("nan"),
    }


def predict_choice_probs(df: pd.DataFrame, fit: ChoiceFoldFit) -> pd.Series:
    """Softmax choice probability within each release group; NaN off-release."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    release = df["is_pass_release_frame"].fillna(False).astype(bool)
    if not release.any():
        return out
    sub = df.loc[release].copy()
    sub["_u"] = fit.predict_utilities(sub)
    group_cols = ["gameId", "touchId", "frameIdx"]
    if "passId" in sub.columns:
        group_cols = ["gameId", "passId"]
    for _, grp in sub.groupby(group_cols, sort=False):
        probs = _softmax(grp["_u"].to_numpy(dtype=float))
        out.loc[grp.index] = probs
    return out
