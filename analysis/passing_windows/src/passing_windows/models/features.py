"""Feature sets and freeze-faithful transforms for Stage 4 models.

GAM-style nonlinear effects are approximated with sklearn ``SplineTransformer``
(B-splines) on continuous geometry/margin terms. Documented in
``STAGE4_VERIFICATION.md``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import ABLATION_NAMES, PRIMARY_ABLATION

# Forbidden as covariates (leakage / narrow-path / freeze).
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "interceptorId",
        "toReceiverId",
        "inferred_targetId",
        "season",
        "ptsScored",
        "chances.ptsScored",
        "shotQuality",  # contemporaneous feature forbidden; OK as validation target
        "q_passability_model",
        "V_catch_model",
        "NOV_model",
        "is_true_receiver",
        "completion_label_eligible",
        "true_receiverId",
        "pass_outcome_class",
    }
)

# Passability ablation feature lists (chosen-trajectory surface).
PASSABILITY_FEATURES: dict[str, tuple[str, ...]] = {
    "distance_only": ("pass_distance_ft",),
    "nearest_defender_distance": ("passer_pressure_ft",),
    "static_lane_geometry_no_velocity": (
        "direct_pass_distance_ft",
        "direct_min_lane_clearance_ft",
        "direct_min_temporal_margin_const35_s",
        "direct_receiver_sep_ft",
        "direct_passer_pressure_ft",
        "direct_dist_to_rim_ft",
        "direct_sideline_prox_ft",
        "direct_baseline_prox_ft",
    ),
    "kinematic_geometry_no_uncertainty": (
        "pass_distance_ft",
        "min_lane_clearance_ft",
        "min_temporal_margin_s",
        "receiver_sep_ft",
        "passer_pressure_ft",
        "dist_to_rim_ft",
        "target_defender_recovery_s",
        "help_density_12ft",
        "sideline_prox_ft",
        "baseline_prox_ft",
        "receiver_toward_rim_fps",
        "shotClock",
        "touch_age_s",
    ),
    "full_model": (
        "pass_distance_ft",
        "min_lane_clearance_ft",
        "min_temporal_margin_s",
        "receiver_sep_ft",
        "passer_pressure_ft",
        "dist_to_rim_ft",
        "target_defender_recovery_s",
        "help_density_12ft",
        "sideline_prox_ft",
        "baseline_prox_ft",
        "receiver_toward_rim_fps",
        "shotClock",
        "touch_age_s",
        "lane_predError_mean",
        "passer_predError",
        "receiver_predError",
        "geometry_passability_score",
    ),
}

# Nonlinear spline targets inside full / kinematic ablations.
SPLINE_COLUMNS: frozenset[str] = frozenset(
    {
        "pass_distance_ft",
        "direct_pass_distance_ft",
        "min_lane_clearance_ft",
        "direct_min_lane_clearance_ft",
        "min_temporal_margin_s",
        "direct_min_temporal_margin_const35_s",
        "receiver_sep_ft",
        "direct_receiver_sep_ft",
        "passer_pressure_ft",
        "direct_passer_pressure_ft",
        "lane_predError_mean",
        "dist_to_rim_ft",
        "direct_dist_to_rim_ft",
    }
)

# Catch-value features at projected catch only (no future / no release-only labels).
CATCH_VALUE_FEATURES: tuple[str, ...] = (
    "catch_x_norm",
    "catch_y_norm",
    "dist_to_rim_ft",
    "angle_to_rim_deg",
    "receiver_sep_ft",
    "passer_pressure_ft",
    "help_density_12ft",
    "target_defender_recovery_s",
    "sideline_prox_ft",
    "baseline_prox_ft",
    "shotClock",
    "receiver_toward_rim_fps",
    "pass_distance_ft",
    "flight_time_s",
)

CATCH_VALUE_BASELINES: dict[str, tuple[str, ...]] = {
    "location_only": ("catch_x_norm", "catch_y_norm", "dist_to_rim_ft"),
    "location_defender": (
        "catch_x_norm",
        "catch_y_norm",
        "dist_to_rim_ft",
        "receiver_sep_ft",
        "help_density_12ft",
    ),
    "full_catch_state": CATCH_VALUE_FEATURES,
}

# Keep-state: ballhandler / touch state at decision frame (causal only).
KEEP_STATE_FEATURES: tuple[str, ...] = (
    "passer_x_event",
    "passer_y_event",
    "passer_pressure_ft",
    "shotClock",
    "touch_age_s",
    "help_density_12ft",
    "passer_predError",
)


def assert_no_forbidden_features(columns: Sequence[str], *, name: str = "features") -> None:
    bad = [c for c in columns if c in FORBIDDEN_FEATURE_COLUMNS]
    if bad:
        raise ValueError(f"{name}: forbidden feature columns {bad}")


def matrix_from_frame(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Extract a float feature matrix; missing stays NaN (imputer handles later)."""
    assert_no_forbidden_features(columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def make_classifier_pipeline(
    feature_names: Sequence[str],
    *,
    use_splines: bool = True,
    C: float = 1.0,
    max_iter: int = 500,
    random_state: int = 0,
) -> Pipeline:
    """Regularized logistic with optional B-spline nonlinear transforms."""
    assert_no_forbidden_features(feature_names)
    names = list(feature_names)
    spline_cols = [c for c in names if use_splines and c in SPLINE_COLUMNS]
    linear_cols = [c for c in names if c not in spline_cols]

    transformers = []
    if spline_cols:
        transformers.append(
            (
                "spline",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=5,
                                degree=3,
                                include_bias=False,
                                knots="quantile",
                            ),
                        ),
                    ]
                ),
                spline_cols,
            )
        )
    if linear_cols:
        transformers.append(
            (
                "linear",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                linear_cols,
            )
        )
    if not transformers:
        raise ValueError("no feature columns for classifier pipeline")

    pre = ColumnTransformer(transformers, remainder="drop")
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def make_regressor_pipeline(
    feature_names: Sequence[str],
    *,
    alpha: float = 1.0,
    use_splines: bool = True,
) -> Pipeline:
    assert_no_forbidden_features(feature_names)
    names = list(feature_names)
    spline_cols = [c for c in names if use_splines and c in SPLINE_COLUMNS]
    linear_cols = [c for c in names if c not in spline_cols]
    transformers = []
    if spline_cols:
        transformers.append(
            (
                "spline",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=5,
                                degree=3,
                                include_bias=False,
                                knots="quantile",
                            ),
                        ),
                    ]
                ),
                spline_cols,
            )
        )
    if linear_cols:
        transformers.append(
            (
                "linear",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                linear_cols,
            )
        )
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("reg", Ridge(alpha=alpha))])


def make_hgb_regressor_pipeline(
    feature_names: Sequence[str],
    *,
    random_state: int = 0,
    max_depth: int = 4,
    max_iter: int = 160,
    learning_rate: float = 0.06,
    min_samples_leaf: int = 35,
    l2_regularization: float = 1.0,
) -> Pipeline:
    """Impute + HistGradientBoostingRegressor (native NaN handling after impute)."""
    assert_no_forbidden_features(feature_names)
    names = list(feature_names)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "reg",
                HistGradientBoostingRegressor(
                    max_depth=max_depth,
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_hgb_classifier_pipeline(
    feature_names: Sequence[str],
    *,
    random_state: int = 0,
    max_depth: int = 3,
    max_iter: int = 120,
    learning_rate: float = 0.06,
    min_samples_leaf: int = 40,
    l2_regularization: float = 1.0,
) -> Pipeline:
    assert_no_forbidden_features(feature_names)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=max_depth,
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    random_state=random_state,
                ),
            ),
        ]
    )


def passability_feature_names(ablation: str = PRIMARY_ABLATION) -> tuple[str, ...]:
    if ablation not in PASSABILITY_FEATURES:
        raise KeyError(f"unknown ablation {ablation!r}; expected one of {ABLATION_NAMES}")
    return PASSABILITY_FEATURES[ablation]
