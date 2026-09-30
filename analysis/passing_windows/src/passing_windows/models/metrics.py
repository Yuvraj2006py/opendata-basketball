"""Fold-internal probability recalibration and proper scoring metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


class LogisticCalibrator:
    """Platt-style logistic recalibration fit on training-fold scores only."""

    def __init__(self) -> None:
        self._model = LogisticRegression(solver="lbfgs", max_iter=200)
        self._fitted = False

    def fit(self, raw_scores: np.ndarray, y: np.ndarray) -> "LogisticCalibrator":
        y = np.asarray(y).astype(int)
        x = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        if len(np.unique(y)) < 2:
            # Degenerate: identity map via intercept-only fallback
            self._fitted = False
            self._const = float(np.mean(y)) if len(y) else 0.5
            return self
        self._model.fit(x, y)
        self._fitted = True
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        if not self._fitted:
            return np.full(len(x), getattr(self, "_const", 0.5), dtype=float)
        return self._model.predict_proba(x)[:, 1]


def calibration_slope_intercept(
    y_true: np.ndarray,
    p_pred: np.ndarray,
) -> dict[str, float]:
    """Calibration slope/intercept via logistic regression of y on logit(p)."""
    y = np.asarray(y_true).astype(int)
    p = clip_prob(p_pred)
    if len(np.unique(y)) < 2 or len(y) < 5:
        return {"slope": float("nan"), "intercept": float("nan"), "n": int(len(y))}
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=200, fit_intercept=True)
    model.fit(logit, y)
    return {
        "slope": float(model.coef_.ravel()[0]),
        "intercept": float(model.intercept_.ravel()[0]),
        "n": int(len(y)),
    }


def binary_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = clip_prob(p_pred)
    out: dict[str, float] = {"n": float(len(y)), "base_rate": float(np.mean(y)) if len(y) else float("nan")}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update(
            {
                "log_loss": float("nan"),
                "brier": float("nan"),
                "calibration_slope": float("nan"),
                "calibration_intercept": float("nan"),
            }
        )
        return out
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y, p))
    cal = calibration_slope_intercept(y, p)
    out["calibration_slope"] = cal["slope"]
    out["calibration_intercept"] = cal["intercept"]
    return out


def per_game_binary_metrics(
    frame: pd.DataFrame,
    *,
    y_col: str,
    p_col: str,
    game_col: str = "gameId",
) -> list[dict[str, Any]]:
    rows = []
    for gid, grp in frame.groupby(game_col, sort=True):
        m = binary_metrics(grp[y_col].to_numpy(), grp[p_col].to_numpy())
        m["gameId"] = int(gid)
        rows.append(m)
    return rows


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {"n": 0.0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    resid = y - p
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": float(len(y)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }
