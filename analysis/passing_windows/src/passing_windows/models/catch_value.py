"""Post-catch value model V_catch(j,t)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from passing_windows.velocities import FRAME_RATE_HZ

from .features import (
    CATCH_VALUE_FEATURES,
    make_classifier_pipeline,
    make_hgb_classifier_pipeline,
    make_hgb_regressor_pipeline,
    make_regressor_pipeline,
    matrix_from_frame,
)
from .metrics import regression_metrics
from .outcomes import attach_next3s_outcomes

TrainingPopulation = Literal["observed_catch", "all_candidates"]
CatchFamily = Literal["ridge_splines", "hist_gbrt", "two_stage"]
CATCH_FAMILIES: tuple[CatchFamily, ...] = ("ridge_splines", "hist_gbrt", "two_stage")
OBSERVED_CATCH_HALF_WINDOW_S = 0.4


@dataclass
class _TwoStageBundle:
    shot_clf: Any
    points_reg: Any
    fallback_mean: float


@dataclass
class CatchValueFoldFit:
    fold_id: str
    held_out_gameId: int
    pipeline: Any = None  # primary ridge path (compat) or unused when stack
    secondary_pipelines: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    feature_names: tuple[str, ...] = CATCH_VALUE_FEATURES
    n_train: int = 0
    training_population: str = "observed_catch"
    model_family: str = "ridge_splines"
    family_models: dict[str, Any] = field(default_factory=dict)
    blend_weights: dict[str, float] = field(default_factory=dict)
    family_train_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def _predict_family(self, family: str, df: pd.DataFrame) -> np.ndarray:
        X = df.loc[:, list(self.feature_names)]
        model = self.family_models[family]
        if family == "two_stage":
            bundle: _TwoStageBundle = model
            p_shot = bundle.shot_clf.predict_proba(X)[:, 1]
            e_pts = np.clip(bundle.points_reg.predict(X), 0.0, 4.0)
            # If classifier is degenerate, fall back to constant mean points.
            if not np.isfinite(p_shot).all():
                return np.full(len(df), bundle.fallback_mean, dtype=float)
            return np.clip(p_shot * e_pts, 0.0, 4.0)
        return np.clip(model.predict(X), 0.0, 4.0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.blend_weights and self.family_models:
            parts = []
            for fam, w in self.blend_weights.items():
                if w == 0.0 or fam not in self.family_models:
                    continue
                parts.append(float(w) * self._predict_family(fam, df))
            if parts:
                return np.clip(np.sum(parts, axis=0), 0.0, 4.0)
        if self.model_family in self.family_models:
            return self._predict_family(self.model_family, df)
        if self.pipeline is not None:
            X = df.loc[:, list(self.feature_names)]
            return np.clip(self.pipeline.predict(X), 0.0, 4.0)
        raise RuntimeError("CatchValueFoldFit has no fitted predictor")

    def predict_family(self, df: pd.DataFrame, family: str) -> np.ndarray:
        return self._predict_family(family, df)

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
    """Boolean mask: true-receiver release ∪ nearby 5 Hz true-receiver rows."""
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


def _select_population(
    train_candidates: pd.DataFrame,
    *,
    training_population: TrainingPopulation,
    half_window_s: float,
    fold_id: str,
) -> pd.DataFrame:
    base = train_candidates
    if training_population == "observed_catch":
        mask = observed_catch_training_mask(base, half_window_s=half_window_s)
        base = base.loc[mask].copy()
        if base.empty:
            raise ValueError(f"{fold_id}: empty observed-catch training mask")
    elif training_population != "all_candidates":
        raise ValueError(f"unknown training_population={training_population!r}")
    return base


def _fit_ridge(X: pd.DataFrame, y: np.ndarray, *, alpha: float, random_state: int) -> Any:
    pipe = make_regressor_pipeline(CATCH_VALUE_FEATURES, alpha=alpha, use_splines=True)
    pipe.fit(X, y)
    return pipe


def _fit_hgb(X: pd.DataFrame, y: np.ndarray, *, random_state: int) -> Any:
    pipe = make_hgb_regressor_pipeline(CATCH_VALUE_FEATURES, random_state=random_state)
    pipe.fit(X, y)
    return pipe


def _fit_two_stage(X: pd.DataFrame, y: np.ndarray, shot: np.ndarray, *, random_state: int) -> _TwoStageBundle:
    fallback = float(np.mean(y)) if len(y) else 0.0
    # Shot occurrence classifier
    if len(np.unique(shot.astype(int))) < 2:
        # Degenerate: constant shot rate
        class _ConstClf:
            def predict_proba(self, X_):
                p = np.full(len(X_), float(np.mean(shot)) if len(shot) else 0.0)
                return np.column_stack([1.0 - p, p])

        shot_clf: Any = _ConstClf()
    else:
        try:
            shot_clf = make_hgb_classifier_pipeline(CATCH_VALUE_FEATURES, random_state=random_state)
            shot_clf.fit(X, shot.astype(int))
        except Exception:
            shot_clf = make_classifier_pipeline(CATCH_VALUE_FEATURES, use_splines=False, random_state=random_state)
            shot_clf.fit(X, shot.astype(int))

    # E[points | state] among shot-window rows; if too few, use all-row ridge.
    pos = shot.astype(bool)
    if int(pos.sum()) >= 25:
        points_reg = make_hgb_regressor_pipeline(CATCH_VALUE_FEATURES, random_state=random_state)
        points_reg.fit(X.loc[pos], y[pos])
    else:
        points_reg = make_regressor_pipeline(CATCH_VALUE_FEATURES, alpha=1.0, use_splines=True)
        points_reg.fit(X, y)
    return _TwoStageBundle(shot_clf=shot_clf, points_reg=points_reg, fallback_mean=fallback)


def _fit_family(
    train: pd.DataFrame,
    family: CatchFamily,
    *,
    alpha: float,
    random_state: int,
) -> Any:
    feats = CATCH_VALUE_FEATURES
    matrix_from_frame(train, feats)
    X = train.loc[:, list(feats)]
    y = train["catch_points_next_3s"].to_numpy(dtype=float)
    if family == "ridge_splines":
        return _fit_ridge(X, y, alpha=alpha, random_state=random_state)
    if family == "hist_gbrt":
        return _fit_hgb(X, y, random_state=random_state)
    if family == "two_stage":
        shot = (
            train["catch_any_shot_next_3s"].to_numpy(dtype=float)
            if "catch_any_shot_next_3s" in train.columns
            else (y > 0).astype(float)
        )
        return _fit_two_stage(X, y, shot, random_state=random_state)
    raise ValueError(family)


def _predict_with_model(model: Any, family: str, df: pd.DataFrame) -> np.ndarray:
    X = df.loc[:, list(CATCH_VALUE_FEATURES)]
    if family == "two_stage":
        bundle: _TwoStageBundle = model
        p_shot = bundle.shot_clf.predict_proba(X)[:, 1]
        e_pts = np.clip(bundle.points_reg.predict(X), 0.0, 4.0)
        return np.clip(p_shot * e_pts, 0.0, 4.0)
    return np.clip(model.predict(X), 0.0, 4.0)


def _blend_weights_from_oof(
    oof_preds: dict[str, np.ndarray],
    y: np.ndarray,
) -> dict[str, float]:
    """Non-negative least-squares blend with sum-to-one; equal weights fallback."""
    families = [f for f in CATCH_FAMILIES if f in oof_preds]
    if not families:
        return {}
    Y = np.asarray(y, dtype=float)
    mats = []
    keep = []
    for f in families:
        p = np.asarray(oof_preds[f], dtype=float)
        ok = np.isfinite(p) & np.isfinite(Y)
        if ok.sum() < 10:
            continue
        mats.append(p)
        keep.append(f)
    if not keep:
        return {f: 1.0 / len(families) for f in families}
    P = np.column_stack(mats)
    # Mask to finite rows across all
    ok = np.isfinite(Y) & np.all(np.isfinite(P), axis=1)
    if int(ok.sum()) < 10:
        return {f: 1.0 / len(keep) for f in keep}
    # Constrained: w>=0 via clipped OLS then renormalize
    try:
        lr = LinearRegression(positive=True, fit_intercept=False)
        lr.fit(P[ok], Y[ok])
        w = np.asarray(lr.coef_, dtype=float)
    except Exception:
        w = np.ones(len(keep), dtype=float)
    w = np.clip(w, 0.0, None)
    if float(w.sum()) <= 1e-12:
        w = np.ones(len(keep), dtype=float)
    w = w / w.sum()
    return {f: float(wi) for f, wi in zip(keep, w)}


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
    model_family: CatchFamily | Literal["stack"] = "stack",
    stack_inner_logo: bool = True,
) -> CatchValueFoldFit:
    """Fit V_catch.

    Default ``model_family='stack'`` fits ridge + HistGBRT + two-stage shot
    composition on the observed-catch population and blends them with
    non-negative weights from leave-one-game OOF predictions inside the
    training fold (no outer holdout peeking).
    """
    base = _select_population(
        train_candidates,
        training_population=training_population,
        half_window_s=half_window_s,
        fold_id=fold_id,
    )
    if "catch_points_next_3s" not in base.columns and max_train_rows is not None and len(base) > max_train_rows:
        base = base.sample(n=max_train_rows, random_state=random_state)
    train = prepare_catch_training(base, shots, free_throws, events_index=events_index)
    if train.empty:
        raise ValueError(f"{fold_id}: empty catch-value training set")
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=random_state)
    train = train.reset_index(drop=True)

    feats = CATCH_VALUE_FEATURES
    matrix_from_frame(train, feats)
    y = train["catch_points_next_3s"].to_numpy(dtype=float)

    families_to_fit: tuple[CatchFamily, ...]
    if model_family == "stack":
        families_to_fit = CATCH_FAMILIES
    else:
        families_to_fit = (model_family,)  # type: ignore[assignment]

    blend_weights: dict[str, float] = {}
    if model_family == "stack" and stack_inner_logo and "gameId" in train.columns:
        games = sorted(int(g) for g in train["gameId"].dropna().unique())
        if len(games) >= 3:
            oof: dict[str, np.ndarray] = {f: np.full(len(train), np.nan) for f in families_to_fit}
            for g in games:
                inner_mask = train["gameId"].astype(int) != g
                hold_mask = ~inner_mask
                if int(hold_mask.sum()) == 0 or int(inner_mask.sum()) < 30:
                    continue
                inner_df = train.loc[inner_mask]
                hold_df = train.loc[hold_mask]
                hold_pos = np.flatnonzero(hold_mask.to_numpy())
                for fam in families_to_fit:
                    m = _fit_family(inner_df, fam, alpha=alpha, random_state=random_state)
                    oof[fam][hold_pos] = _predict_with_model(m, fam, hold_df)
            blend_weights = _blend_weights_from_oof(oof, y)
        else:
            blend_weights = {f: 1.0 / len(families_to_fit) for f in families_to_fit}
    elif model_family == "stack":
        blend_weights = {f: 1.0 / len(families_to_fit) for f in families_to_fit}
    else:
        blend_weights = {model_family: 1.0}

    family_models: dict[str, Any] = {}
    family_metrics: dict[str, dict[str, float]] = {}
    for fam in families_to_fit:
        family_models[fam] = _fit_family(train, fam, alpha=alpha, random_state=random_state)
        pred_f = _predict_with_model(family_models[fam], fam, train)
        family_metrics[fam] = regression_metrics(y, pred_f)

    # Secondary linear-prob targets from ridge features (diagnostic).
    secondary: dict[str, Any] = {}
    X = train.loc[:, list(feats)]
    for col in ("catch_any_shot_next_3s", "catch_open_shot_next_3s"):
        if col not in train.columns:
            continue
        y2 = train[col].to_numpy(dtype=float)
        p2 = make_regressor_pipeline(feats, alpha=alpha, use_splines=False)
        p2.fit(X, y2)
        secondary[col.replace("catch_", "")] = p2

    # Build a temporary fit object for stacked train metrics.
    tmp = CatchValueFoldFit(
        fold_id=fold_id,
        held_out_gameId=held_out_gameId,
        pipeline=family_models.get("ridge_splines"),
        secondary_pipelines=secondary,
        feature_names=feats,
        n_train=int(len(y)),
        training_population=training_population,
        model_family="stack" if model_family == "stack" else model_family,
        family_models=family_models,
        blend_weights=blend_weights,
        family_train_metrics=family_metrics,
    )
    train_pred = tmp.predict(train)
    tmp.train_metrics = regression_metrics(y, train_pred)
    return tmp


def predict_catch_value(df: pd.DataFrame, fit: CatchValueFoldFit) -> pd.DataFrame:
    out = df[["gameId", "touchId", "frameIdx", "candidateId"]].copy()
    out["V_catch_model"] = fit.predict(df)
    for fam in fit.family_models:
        out[f"V_catch_ablation_{fam}"] = fit.predict_family(df, fam)
    for name in fit.secondary_pipelines:
        out[f"V_catch_sec_{name}"] = fit.predict_secondary(df, name)
    return out
