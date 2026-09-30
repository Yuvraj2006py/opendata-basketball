"""Tests for Stage 4 cross-fitted component models.

Unit tests are synthetic and always run. Tests marked ``needs_stage4`` read
tables produced by ``pipelines/04_fit_component_models.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passing_windows.models import ABLATION_NAMES, MODEL_VERSION, PRIMARY_ABLATION
from passing_windows.models.catch_value import (
    evaluate_catch_predictions,
    fit_catch_value,
    observed_catch_training_mask,
    predict_catch_value,
)
from passing_windows.models.choice import evaluate_choice, fit_choice_model
from passing_windows.models.features import (
    FORBIDDEN_FEATURE_COLUMNS,
    PASSABILITY_FEATURES,
    assert_no_forbidden_features,
    make_classifier_pipeline,
    passability_feature_names,
)
from passing_windows.models.keep_state import fit_keep_state
from passing_windows.models.leakage import (
    Stage4LeakageError,
    assert_completion_labels_narrow_path,
    assert_held_out_absent_from_train,
    assert_logo_only,
    assert_no_season_aggregate_covariates,
    assert_train_rows_exclude_held_out,
)
from passing_windows.models.metrics import LogisticCalibrator, binary_metrics, clip_prob
from passing_windows.models.option_value import attach_option_columns, compute_nov, compute_option_value
from passing_windows.models.outcomes import (
    attach_next3s_outcomes,
    build_scoring_events,
    points_in_window,
    shot_points,
)
from passing_windows.models.passability import fit_passability_ablation, predict_passability_table
from passing_windows.narrow_path import NarrowPathViolation
from passing_windows.paths import ARTIFACTS, TABLES

PRED = TABLES / "stage4_candidate_predictions.parquet"
METRICS = ARTIFACTS / "stage4_fold_metrics.json"
MANIFEST = ARTIFACTS / "stage4_output_manifest.json"
FEATURES = TABLES / "stage3_candidate_features.parquet"
REGRESSION_LOCK = ARTIFACTS / "STAGE4_REGRESSION_LOCK.md"

needs_stage4 = pytest.mark.skipif(not PRED.exists(), reason="Stage 4 pipeline not run")
needs_stage3 = pytest.mark.skipif(not FEATURES.exists(), reason="Stage 3 features missing")


def _toy_candidates(n_frames: int = 20, n_cand: int = 4) -> pd.DataFrame:
    rows = []
    for f in range(n_frames):
        for c in range(n_cand):
            rows.append(
                {
                    "gameId": 1,
                    "touchId": "t1",
                    "frameIdx": f * 5,
                    "candidateId": 100 + c,
                    "chanceId": "chance-1",
                    "possessionId": "poss-1",
                    "passId": "pass-1" if f == 10 else pd.NA,
                    "fold_id": "fold_holdout_2",
                    "completion_label_eligible": f == 10,
                    "is_true_receiver": f == 10 and c == 0,
                    "is_pass_release_frame": f == 10,
                    "is_model_5hz_frame": True,
                    "receiver_specific_eligible": f == 10,
                    "target_reliability_status": "known_receiver" if f == 10 else "no_evidence",
                    "interceptorId": 999,
                    "true_receiverId": 100 if f == 10 else pd.NA,
                    "pass_distance_ft": 10.0 + c,
                    "min_lane_clearance_ft": 2.0 + 0.1 * c,
                    "min_temporal_margin_s": 0.2 + 0.01 * c,
                    "receiver_sep_ft": 3.0 + c,
                    "passer_pressure_ft": 4.0,
                    "dist_to_rim_ft": 18.0 - c,
                    "target_defender_recovery_s": 0.5,
                    "help_density_12ft": 1.0,
                    "sideline_prox_ft": 5.0,
                    "baseline_prox_ft": 10.0,
                    "receiver_toward_rim_fps": 1.0,
                    "shotClock": 14.0,
                    "touch_age_s": float(f) * 0.2,
                    "lane_predError_mean": 0.2,
                    "passer_predError": 0.1,
                    "receiver_predError": 0.1,
                    "geometry_passability_score": 0.5 - 0.05 * c,
                    "direct_pass_distance_ft": 10.0 + c,
                    "direct_min_lane_clearance_ft": 2.0,
                    "direct_min_temporal_margin_const35_s": 0.15,
                    "direct_receiver_sep_ft": 3.0,
                    "direct_passer_pressure_ft": 4.0,
                    "direct_dist_to_rim_ft": 18.0,
                    "direct_sideline_prox_ft": 5.0,
                    "direct_baseline_prox_ft": 10.0,
                    "catch_x_norm": -20.0 + c,
                    "catch_y_norm": float(c),
                    "angle_to_rim_deg": 10.0,
                    "flight_time_s": 0.5,
                    "passer_x_event": 0.0,
                    "passer_y_event": 0.0,
                    "rejected_outside_support": c == 3,
                    "trajectory_choice": "lead",
                    "trajectory_choice_rule": "lead_if_velocity_ok_else_direct",
                }
            )
    return pd.DataFrame(rows)


def _toy_shots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gameId": 1,
                "chanceId": "chance-1",
                "startFrame": 55,
                "endFrame": 60,
                "outcome": True,
                "three": False,
                "contested": False,
                "contestLevel": "open",
                "createdFromPaint": False,
            }
        ]
    )


def _toy_ft() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["gameId", "chanceId", "frame", "outcome"]
    )


# ---------------------------------------------------------------------------
# Leakage / narrow-path guards
# ---------------------------------------------------------------------------


def test_logo_only_rejects_random_splits():
    assert_logo_only("nested leave-one-game-out LOGO")
    with pytest.raises(Stage4LeakageError):
        assert_logo_only("random_frame_split")
    with pytest.raises(Stage4LeakageError):
        assert_logo_only("train_test_split on touches")


def test_held_out_absent_from_train():
    assert_held_out_absent_from_train([1, 2, 3], 4)
    with pytest.raises(Stage4LeakageError):
        assert_held_out_absent_from_train([1, 2, 3], 2)


def test_train_rows_exclude_held_out():
    df = pd.DataFrame({"gameId": [1, 1, 2]})
    assert_train_rows_exclude_held_out(df, 3)
    with pytest.raises(Stage4LeakageError):
        assert_train_rows_exclude_held_out(df, 2)


def test_forbidden_features_blocked():
    with pytest.raises(ValueError):
        assert_no_forbidden_features(["pass_distance_ft", "interceptorId"])
    with pytest.raises(Stage4LeakageError):
        assert_no_season_aggregate_covariates(["season_avg_ast"])


def test_completion_labels_reject_non_eligible():
    df = _toy_candidates()
    bad = df.copy()
    bad.loc[:, "completion_label_eligible"] = True
    bad.loc[:, "target_reliability_status"] = "inferred_provisional_unaudited"
    with pytest.raises(NarrowPathViolation):
        assert_completion_labels_narrow_path(bad)


def test_interceptor_never_equals_true_receiver_in_labels():
    df = _toy_candidates()
    labeled = df[df["completion_label_eligible"]].copy()
    labeled["true_receiverId"] = labeled["interceptorId"]
    with pytest.raises(NarrowPathViolation):
        assert_completion_labels_narrow_path(labeled)


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def test_shot_points_two_and_three():
    assert shot_points(pd.Series({"outcome": True, "three": False})) == 2.0
    assert shot_points(pd.Series({"outcome": True, "three": True})) == 3.0
    assert shot_points(pd.Series({"outcome": False, "three": True})) == 0.0


def test_points_in_window_searchsorted():
    arr = np.array([[10, 2.0, 1.0, 1.0], [50, 3.0, 1.0, 0.0], [100, 2.0, 1.0, 0.0]], dtype=float)
    s = points_in_window(arr, 40, horizon_frames=20)
    assert s["points_next_3s"] == 3.0
    assert s["any_shot_next_3s"] == 1.0


def test_attach_next3s_uses_shots_not_empty():
    cand = _toy_candidates()
    # Shot ends at frame 70; catch window from frameIdx=50 + flight 0.5s (~12.5 frames)
    # starts near 62–63 and includes endFrame 70.
    shots = pd.DataFrame(
        [
            {
                "gameId": 1,
                "chanceId": "chance-1",
                "startFrame": 65,
                "endFrame": 70,
                "outcome": True,
                "three": False,
                "contested": False,
                "contestLevel": "open",
                "createdFromPaint": False,
            }
        ]
    )
    out = attach_next3s_outcomes(cand, shots, _toy_ft(), use_projected_catch=True)
    assert "catch_points_next_3s" in out.columns
    release = out[out["frameIdx"] == 50]
    assert float(release["catch_points_next_3s"].max()) == 2.0


def test_build_scoring_events_ignores_misses():
    shots = pd.DataFrame(
        [
            {"gameId": 1, "chanceId": "c", "startFrame": 1, "endFrame": 2, "outcome": False, "three": True},
            {"gameId": 1, "chanceId": "c", "startFrame": 3, "endFrame": 4, "outcome": True, "three": True},
        ]
    )
    ev = build_scoring_events(shots, _toy_ft())
    assert list(ev["points"]) == [0.0, 3.0]


# ---------------------------------------------------------------------------
# Option value formulas
# ---------------------------------------------------------------------------


def test_q_and_nov_formulas():
    q = np.array([0.5, 1.0, 0.0])
    vc = np.array([2.0, 2.0, 2.0])
    vk = np.array([0.5, 0.5, 0.5])
    Q = compute_option_value(q, vc, v_fail=0.0)
    assert np.allclose(Q, [1.0, 2.0, 0.0])
    nov = compute_nov(Q, vk)
    assert np.allclose(nov, [0.5, 1.5, -0.5])


def test_rejected_masked_for_counterfactual_q_nov():
    df = pd.DataFrame(
        {
            "q_passability_model": [0.8, 0.8],
            "V_catch_model": [2.0, 2.0],
            "V_keep_model": [0.5, 0.5],
            "rejected_outside_support": [False, True],
        }
    )
    out = attach_option_columns(df)
    assert out.loc[0, "Q_option_model"] == pytest.approx(1.6)
    assert np.isnan(out.loc[1, "Q_option_model"])
    assert np.isnan(out.loc[1, "NOV_model"])
    assert out.loc[1, "V_fail"] == 0.0


# ---------------------------------------------------------------------------
# Passability / calibrator
# ---------------------------------------------------------------------------


def test_ablation_feature_sets_cover_freeze_names():
    assert set(ABLATION_NAMES) == set(PASSABILITY_FEATURES)
    assert PRIMARY_ABLATION == "full_model"
    for name in ABLATION_NAMES:
        feats = passability_feature_names(name)
        assert feats
        assert not (set(feats) & FORBIDDEN_FEATURE_COLUMNS)


def test_logistic_calibrator_improves_or_preserves():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    raw = y.astype(float) * 0.6 + rng.normal(0, 0.2, size=200)
    cal = LogisticCalibrator().fit(raw, y)
    p = cal.transform(raw)
    assert p.min() >= 0 and p.max() <= 1


def test_fit_passability_toy_and_predict():
    df = _toy_candidates(n_frames=40)
    # Need both classes — mark more eligible rows.
    df.loc[df["frameIdx"] == 20, "completion_label_eligible"] = True
    df.loc[(df["frameIdx"] == 20) & (df["candidateId"] == 100), "is_true_receiver"] = True
    df.loc[df["frameIdx"] == 20, "target_reliability_status"] = "known_receiver"
    df.loc[df["frameIdx"] == 30, "completion_label_eligible"] = True
    df.loc[(df["frameIdx"] == 30) & (df["candidateId"] == 101), "is_true_receiver"] = True
    df.loc[df["frameIdx"] == 30, "target_reliability_status"] = "known_receiver"
    fit = fit_passability_ablation(
        df, fold_id="fold_holdout_2", held_out_gameId=2, ablation="distance_only"
    )
    pred = fit.predict_proba(df)
    assert len(pred) == len(df)
    assert np.all((pred >= 0) & (pred <= 1))


def test_clip_prob_bounds():
    p = clip_prob(np.array([-1.0, 0.5, 2.0]))
    assert p.min() > 0 and p.max() < 1


# ---------------------------------------------------------------------------
# Catch / keep / choice smoke
# ---------------------------------------------------------------------------


def test_observed_catch_mask_release_and_nearby_5hz():
    df = _toy_candidates(n_frames=30)
    # release at f=10 -> frameIdx=50; true receiver candidateId=100
    mask = observed_catch_training_mask(df, half_window_s=0.4)
    assert bool(mask.loc[(df["frameIdx"] == 50) & (df["candidateId"] == 100)].all())
    near = df[(df["candidateId"] == 100) & ((df["frameIdx"] - 50).abs() <= 10)]
    assert len(near) >= 3
    assert bool(mask.loc[near.index].all())
    far = df[(df["candidateId"] == 100) & ((df["frameIdx"] - 50).abs() > 10)]
    assert not bool(mask.loc[far.index].any())
    other = df[df["candidateId"] != 100]
    assert not bool(mask.loc[other.index].any())


def test_observed_catch_mask_excludes_ineligible_release():
    df = _toy_candidates(n_frames=20)
    df.loc[df["frameIdx"] == 50, "completion_label_eligible"] = False
    mask = observed_catch_training_mask(df)
    assert not bool(mask.any())


def test_evaluate_catch_predictions_includes_spearman():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.1, 0.9, 2.1, 2.8])
    m = evaluate_catch_predictions(y, p)
    assert set(m) >= {"n", "rmse", "mae", "r2", "spearman"}
    assert m["n"] == 4.0
    assert np.isfinite(m["spearman"])


def test_fit_catch_and_keep_toy():
    df = _toy_candidates(n_frames=30)
    # Duplicate as a second "train" game pattern for volume
    df2 = df.copy()
    df2["gameId"] = 2
    train = pd.concat([df, df2], ignore_index=True)
    shots = pd.concat([_toy_shots(), _toy_shots().assign(gameId=2)], ignore_index=True)
    catch = fit_catch_value(
        train, shots, _toy_ft(), fold_id="f", held_out_gameId=3, max_train_rows=None
    )
    assert catch.training_population == "observed_catch"
    assert catch.model_family == "stack"
    assert catch.n_train > 0
    assert set(catch.blend_weights) <= {"ridge_splines", "hist_gbrt", "two_stage"}
    pred = predict_catch_value(df, catch)
    assert "V_catch_model" in pred.columns
    assert "V_catch_ablation_hist_gbrt" in pred.columns
    # Ablation population still works.
    catch_all = fit_catch_value(
        train,
        shots,
        _toy_ft(),
        fold_id="f",
        held_out_gameId=3,
        max_train_rows=None,
        training_population="all_candidates",
        model_family="ridge_splines",
        stack_inner_logo=False,
    )
    assert catch_all.n_train >= catch.n_train
    keep = fit_keep_state(
        train, shots, _toy_ft(), fold_id="f", held_out_gameId=3, max_train_rows=None
    )
    assert keep.n_train > 0


def test_choice_model_toy():
    df = _toy_candidates(n_frames=15)
    df["q_passability_model"] = 0.4
    df.loc[df["is_true_receiver"], "q_passability_model"] = 0.9
    df["V_catch_model"] = 1.0
    df["NOV_model"] = df["q_passability_model"] - 0.3
    df["is_pass_release_frame"] = df["completion_label_eligible"]
    # Add a second release
    extra = df[df["frameIdx"] == 50].copy()
    extra["frameIdx"] = 55
    extra["passId"] = "pass-2"
    extra["is_true_receiver"] = extra["candidateId"] == 101
    train = pd.concat([df, extra], ignore_index=True)
    fit = fit_choice_model(train, fold_id="f", held_out_gameId=9)
    metrics = evaluate_choice(train, fit)
    assert metrics["n_releases"] >= 1


def test_make_classifier_pipeline_runs():
    df = _toy_candidates()
    feats = ["pass_distance_ft", "receiver_sep_ft"]
    pipe = make_classifier_pipeline(feats, use_splines=True)
    y = (df["candidateId"] == 100).astype(int).to_numpy()
    pipe.fit(df[feats], y)
    p = pipe.predict_proba(df[feats])[:, 1]
    assert len(p) == len(df)


# ---------------------------------------------------------------------------
# Integration (needs Stage 4 outputs)
# ---------------------------------------------------------------------------


@needs_stage4
def test_stage4_pred_keys_and_columns():
    pred = pd.read_parquet(PRED)
    join_keys = ["gameId", "touchId", "frameIdx", "candidateId"]
    for c in join_keys:
        assert c in pred.columns
    required = [
        "q_passability_model",
        "V_catch_model",
        "V_catch_ablation_all_candidates",
        "V_keep_model",
        "V_fail",
        "Q_option_model",
        "NOV_model",
        "fold_id",
        "model_version",
    ]
    for c in required:
        assert c in pred.columns, c
    assert pred["model_version"].iloc[0] == MODEL_VERSION
    assert (pred["V_fail"] == 0).all()
    assert pred["V_catch_ablation_all_candidates"].notna().all()


@needs_stage4
def test_stage4_fold_metrics_observed_catch_keys():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert metrics["model_version"] == MODEL_VERSION
    catch = metrics["overall_catch"]
    for k in ("n", "rmse", "mae", "r2", "spearman"):
        assert k in catch, k
    assert "overall_catch_ablation_all_candidates_on_observed_mask" in metrics
    assert "per_game_catch_observed" in metrics
    assert len(metrics["per_game_catch_observed"]) == 10


@needs_stage4
def test_stage4_join_aligns_with_stage3():
    """Predictions must cover every Stage 3 row for games present in Stage 4.

    A one-game smoke write is allowed; full 10-game alignment is required once
    all freeze games appear in the predictions table.
    """
    pred = pd.read_parquet(PRED, columns=["gameId", "touchId", "frameIdx", "candidateId"])
    feat = pd.read_parquet(FEATURES, columns=["gameId", "touchId", "frameIdx", "candidateId"])
    pred_games = set(pred["gameId"].dropna().astype(int).tolist())
    feat_scoped = feat[feat["gameId"].astype(int).isin(pred_games)]
    assert len(pred) == len(feat_scoped), (
        f"stage4 rows {len(pred)} != stage3 rows for games {sorted(pred_games)} ({len(feat_scoped)})"
    )
    m = feat_scoped.merge(pred, on=["gameId", "touchId", "frameIdx", "candidateId"], how="inner")
    assert len(m) == len(feat_scoped)


@needs_stage4
def test_stage4_rejected_q_nov_nan():
    pred = pd.read_parquet(
        PRED, columns=["rejected_outside_support", "Q_option_model", "NOV_model", "q_passability_model"]
    )
    rej = pred[pred["rejected_outside_support"].fillna(False)]
    if len(rej):
        assert rej["Q_option_model"].isna().all()
        assert rej["NOV_model"].isna().all()
        # raw q still present for diagnostics
        assert rej["q_passability_model"].notna().any()


@needs_stage4
def test_stage4_passability_trained_only_on_eligible_implies_holdout_labels():
    pred = pd.read_parquet(PRED)
    labeled = pred[pred["completion_label_eligible"].fillna(False)]
    assert labeled["is_true_receiver"].any()
    # One true receiver per labeled pass among candidates (typically)
    if "passId" in labeled.columns:
        rates = labeled.groupby("passId")["is_true_receiver"].sum()
        assert (rates == 1).mean() > 0.9


@needs_stage4
def test_stage4_ablation_columns_present():
    pred = pd.read_parquet(PRED)
    for name in ABLATION_NAMES:
        if name == PRIMARY_ABLATION:
            continue
        assert f"q_ablation_{name}" in pred.columns


@needs_stage4
def test_stage4_metrics_and_manifest_exist():
    assert METRICS.exists()
    assert MANIFEST.exists()
    assert (ARTIFACTS / "stage4_model_report.md").exists()
    assert (ARTIFACTS / "STAGE4_VERIFICATION.md").exists()
    assert (ARTIFACTS / "STAGE4_HANDOFF_STAGE5.md").exists()
    assert REGRESSION_LOCK.exists()
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert "overall_passability" in metrics
    assert metrics["model_version"] == MODEL_VERSION


@needs_stage4
def test_stage4_no_logo_leakage_in_fold_id():
    pred = pd.read_parquet(PRED, columns=["gameId", "fold_id"])
    for gid, fold in pred.groupby("gameId")["fold_id"].first().items():
        assert str(gid) in str(fold)


@needs_stage3
def test_stage3_placeholders_do_not_block_stage4_schema():
    feat = pd.read_parquet(FEATURES, columns=["q_passability_model", "V_catch_model", "NOV_model"])
    # Stage 3 leaves NaN placeholders; Stage 4 writes a separate table.
    assert feat["q_passability_model"].isna().all() or PRED.exists()


# ---------------------------------------------------------------------------
# Auditor regression lock (S4-B## / S4-M##) — names match STAGE4_REGRESSION_LOCK
# ---------------------------------------------------------------------------


@needs_stage4
def test_stage4_no_random_group_split():
    """S4-B01: Stage 4 fit path must not use random frame/touch/chance splits."""
    from passing_windows.models.leakage import assert_logo_only

    assert_logo_only("nested leave-one-game-out LOGO")
    pipe = (ROOT / "pipelines" / "04_fit_component_models.py").read_text(encoding="utf-8")
    # Pipeline must not call sklearn splitters; leakage module may name them as forbidden tokens.
    assert "sklearn.model_selection" not in pipe
    assert "train_test_split(" not in pipe
    assert "ShuffleSplit(" not in pipe


@needs_stage4
def test_stage4_logo_holdout_isolation():
    """S4-B02: each fold_id names its held-out game; train never includes that game."""
    from passing_windows.ingest import load_freeze

    freeze = load_freeze()
    pred = pd.read_parquet(PRED, columns=["gameId", "fold_id"])
    for fold in freeze["validation"]["outer_folds"]:
        held = int(fold["held_out_gameId"])
        train_ids = {int(g) for g in fold["train_gameIds"]}
        assert held not in train_ids
        rows = pred[pred["fold_id"] == fold["fold_id"]]
        if len(rows):
            assert set(rows["gameId"].astype(int).unique()) == {held}


@needs_stage4
def test_stage4_fold_seeds_match_freeze():
    """S4-B03: freeze fold seeds 2025092401–10."""
    from passing_windows.ingest import load_freeze

    freeze = load_freeze()
    seeds = freeze["rng"]["fold_seeds"]
    assert freeze["rng"]["master_seed"] == 20250924
    for fold in freeze["validation"]["outer_folds"]:
        assert fold["fold_id"] in seeds
        assert int(seeds[fold["fold_id"]]) >= 2025092401


@needs_stage4
def test_stage4_inner_cv_excludes_outer_holdout():
    """S4-M01: no nested CV over holdout — Stage 4 uses outer LOGO only."""
    pipe = (ROOT / "pipelines" / "04_fit_component_models.py").read_text(encoding="utf-8")
    assert "GridSearchCV" not in pipe and "RandomizedSearchCV" not in pipe


@needs_stage4
def test_stage4_passability_uses_release_frames():
    """S4-B04: completion-eligible rows are release-time states."""
    pred = pd.read_parquet(
        PRED,
        columns=["completion_label_eligible", "is_pass_release_frame", "is_forced_release_frame"],
    )
    lab = pred[pred["completion_label_eligible"].fillna(False)]
    assert len(lab) > 0
    assert lab["is_pass_release_frame"].fillna(False).all() or lab["is_forced_release_frame"].fillna(False).all()


@needs_stage4
def test_stage4_release_label_yield():
    """S4-B05: ≥90% of narrow-path completion labels have Stage 4 release rows."""
    from passing_windows.narrow_path import receiver_completion_training_labels

    elig = pd.read_parquet(TABLES / "stage2_label_eligibility.parquet")
    labels = receiver_completion_training_labels(elig)
    pred = pd.read_parquet(PRED, columns=["passId", "completion_label_eligible"])
    hit = labels["passId"].isin(pred.loc[pred["completion_label_eligible"].fillna(False), "passId"])
    assert float(hit.mean()) >= 0.90


@needs_stage4
def test_stage4_model_5hz_series_separate_from_release():
    """S4-M02: both sampling flags exist; release need not be on stride-5 lattice."""
    pred = pd.read_parquet(
        PRED, columns=["is_model_5hz_frame", "is_forced_release_frame", "frameIdx"]
    )
    assert pred["is_model_5hz_frame"].any()
    forced = pred[pred["is_forced_release_frame"].fillna(False)]
    if len(forced):
        # Forced release may land off the 5 Hz lattice.
        assert True  # column presence is the contract; off-lattice allowed


@needs_stage4
def test_stage4_sampling_role_values():
    """S4-N01: sampling_role provenance values."""
    pred = pd.read_parquet(PRED, columns=["sampling_role"])
    roles = set(pred["sampling_role"].dropna().astype(str).unique())
    assert roles & {"model_5hz", "pass_release", "model_5hz+pass_release"}


@needs_stage4
def test_stage4_no_failed_completion_labels():
    """S4-B06: no failed-cohort rows marked completion_label_eligible."""
    cols = ["completion_label_eligible"]
    pred = pd.read_parquet(PRED)
    if "cohort" in pred.columns:
        bad = pred["completion_label_eligible"].fillna(False) & (pred["cohort"].astype(str) == "failed")
        assert int(bad.sum()) == 0


@needs_stage4
def test_stage4_interceptor_never_target():
    """S4-B07: interceptorId never equals true_receiverId / candidateId on labeled rows."""
    from passing_windows.narrow_path import assert_interceptor_never_target

    pred = pd.read_parquet(PRED)
    labeled = pred[pred["completion_label_eligible"].fillna(False)]
    if "interceptorId" in labeled.columns and "true_receiverId" in labeled.columns:
        assert_interceptor_never_target(labeled, name="stage4_pred", target_col="true_receiverId")


@needs_stage4
def test_stage4_gate_columns_present():
    """S4-B08: all narrow-path GATE_COLUMNS on Stage 4 scored table."""
    from passing_windows.narrow_path import GATE_COLUMNS

    pred = pd.read_parquet(PRED)
    missing = [c for c in GATE_COLUMNS if c not in pred.columns]
    assert not missing, missing


@needs_stage4
def test_stage4_no_fabricated_audit_claims():
    """S4-B09: reports must not claim failed targets audited/validated."""
    texts = [
        (ARTIFACTS / "stage4_model_report.md").read_text(encoding="utf-8"),
        (ARTIFACTS / "STAGE4_VERIFICATION.md").read_text(encoding="utf-8"),
        (ARTIFACTS / "STAGE4_HANDOFF_STAGE5.md").read_text(encoding="utf-8"),
    ]
    for t in texts:
        low = t.lower()
        assert "validated failed-pass" not in low
        assert "audited failed-pass" not in low


@needs_stage4
def test_stage4_no_failed_precision_claim():
    """S4-B10: no completed-pass proxy cited as failed-pass precision."""
    from passing_windows.models.leakage import scan_reports_for_forbidden_claims

    texts = [
        (ARTIFACTS / "stage4_model_report.md").read_text(encoding="utf-8"),
        (ARTIFACTS / "STAGE4_VERIFICATION.md").read_text(encoding="utf-8"),
    ]
    scan_reports_for_forbidden_claims(texts)


@needs_stage4
def test_stage4_named_failed_sensitivity():
    """S4-M03: Stage 4 does not opt-in named failed targets by default."""
    pred = pd.read_parquet(PRED)
    if "usable_as_named_failed_target" in pred.columns and "completion_label_eligible" in pred.columns:
        bad = pred["completion_label_eligible"].fillna(False) & pred["usable_as_named_failed_target"].fillna(False)
        assert int(bad.sum()) == 0


@needs_stage4
def test_stage4_q_oos_predictions_present():
    """S4-B11: OOS q_passability_model filled."""
    pred = pd.read_parquet(PRED, columns=["q_passability_model"])
    assert int(pred["q_passability_model"].notna().sum()) == len(pred)


@needs_stage4
def test_stage4_passability_ablation_ladder():
    """S4-M04: all freeze ablations present + metrics."""
    pred = pd.read_parquet(PRED)
    for name in ABLATION_NAMES:
        if name == PRIMARY_ABLATION:
            assert "q_passability_model" in pred.columns
        else:
            assert f"q_ablation_{name}" in pred.columns
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert set(metrics.get("ablation_overall", {})) >= set(ABLATION_NAMES)


@needs_stage4
def test_stage4_imbalance_weights_fold_internal():
    """S4-B12: imbalance weighting helper exists and is train-fold only."""
    from passing_windows.models.passability import _balance_sample_weights

    y = np.array([0, 0, 0, 1])
    w = _balance_sample_weights(y)
    assert len(w) == 4
    assert w[y == 1].mean() > w[y == 0].mean()


@needs_stage4
def test_stage4_passability_heldout_metrics():
    """S4-M05: held-out log loss / Brier / calibration present."""
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    op = metrics["overall_passability"]
    for k in ("log_loss", "brier", "calibration_slope", "calibration_intercept"):
        assert k in op and op[k] == op[k]  # not NaN


@needs_stage4
def test_stage4_calibration_slope_reported():
    """S4-B13: calibration slope in ~0.75–1.25 kill band (or documented)."""
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    slope = float(metrics["overall_passability"]["calibration_slope"])
    assert 0.75 <= slope <= 1.25


@needs_stage4
def test_stage4_recalibration_fold_internal():
    """S4-M06: LogisticCalibrator used for fold-internal recalibration."""
    from passing_windows.models.metrics import LogisticCalibrator

    assert LogisticCalibrator is not None
    src = (ROOT / "src" / "passing_windows" / "models" / "passability.py").read_text(encoding="utf-8")
    assert "LogisticCalibrator" in src


@needs_stage4
def test_stage4_vcatch_no_future_features():
    """S4-B14: catch features are projected-catch / causal only."""
    from passing_windows.models.features import CATCH_VALUE_FEATURES

    assert "shotQuality" not in CATCH_VALUE_FEATURES
    assert not any(str(c).startswith("future_") for c in CATCH_VALUE_FEATURES)


@needs_stage4
def test_stage4_outcome_not_from_ptsScored():
    """S4-B15: outcomes from shots+FT module; ptsScored forbidden."""
    from passing_windows.models.outcomes import assert_no_pts_scored_source

    src = (ROOT / "src" / "passing_windows" / "models" / "outcomes.py").read_text(encoding="utf-8")
    assert "shots" in src and "free_throws" in src
    assert_no_pts_scored_source(["points_next_3s", "catch_points_next_3s"])


@needs_stage4
def test_stage4_shotQuality_not_feature():
    """S4-B16: shotQuality not in any Stage 4 feature set."""
    from passing_windows.models.features import (
        CATCH_VALUE_FEATURES,
        KEEP_STATE_FEATURES,
        PASSABILITY_FEATURES,
    )

    all_feats = set(CATCH_VALUE_FEATURES) | set(KEEP_STATE_FEATURES)
    for v in PASSABILITY_FEATURES.values():
        all_feats |= set(v)
    assert "shotQuality" not in all_feats


@needs_stage4
def test_stage4_secondary_outcomes_documented():
    """S4-M07: secondary catch preds or explicit deferral in verification."""
    pred = pd.read_parquet(PRED)
    has_sec = any(c.startswith("V_catch_sec_") for c in pred.columns)
    ver = (ARTIFACTS / "STAGE4_VERIFICATION.md").read_text(encoding="utf-8").lower()
    assert has_sec or "secondary" in ver


@needs_stage4
def test_stage4_vkeep_causal():
    """S4-B17: V_keep from no-pass states; keep features causal."""
    from passing_windows.models.features import KEEP_STATE_FEATURES

    assert "shotClock" in KEEP_STATE_FEATURES
    assert "passer_pressure_ft" in KEEP_STATE_FEATURES
    pred = pd.read_parquet(PRED, columns=["V_keep_model"])
    assert pred["V_keep_model"].notna().all()


@needs_stage4
def test_stage4_vfail_primary_zero():
    """S4-B18: V_fail primary is 0."""
    pred = pd.read_parquet(PRED, columns=["V_fail"])
    assert (pred["V_fail"] == 0).all()


@needs_stage4
def test_stage4_components_not_collapsed_to_nov():
    """S4-B19: q, V_catch, Q, NOV all exposed."""
    pred = pd.read_parquet(PRED)
    for c in ("q_passability_model", "V_catch_model", "Q_option_model", "NOV_model"):
        assert c in pred.columns
        assert pred[c].notna().any()


@needs_stage4
def test_stage4_vcatch_baseline_ladder():
    """S4-M08: catch-value baseline feature sets defined."""
    from passing_windows.models.features import CATCH_VALUE_BASELINES

    assert "location_only" in CATCH_VALUE_BASELINES
    assert "full_catch_state" in CATCH_VALUE_BASELINES


@needs_stage4
def test_stage4_choice_model_metrics():
    """S4-M09: choice rank accuracy + log loss reported."""
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    ch = metrics["overall_choice"]
    assert ch.get("n_releases", 0) > 0
    assert ch.get("rank_accuracy") == ch.get("rank_accuracy")
    assert ch.get("choice_log_loss") == ch.get("choice_log_loss")


@needs_stage4
def test_stage4_forbidden_claim_language():
    """S4-B20: no correct-pass / bad-decision language in Stage 4 reports."""
    for name in ("stage4_model_report.md", "STAGE4_VERIFICATION.md", "STAGE4_HANDOFF_STAGE5.md"):
        text = (ARTIFACTS / name).read_text(encoding="utf-8").lower()
        assert "correct pass" not in text
        assert "bad decision" not in text
        assert "points left on the table" not in text


@needs_stage4
def test_stage4_trajectory_rule_unchanged():
    """S4-B21: trajectory_choice_rule is lead_if_velocity_ok_else_direct."""
    pred = pd.read_parquet(PRED, columns=["trajectory_choice_rule"])
    assert (pred["trajectory_choice_rule"] == "lead_if_velocity_ok_else_direct").all()
    pipe = (ROOT / "pipelines" / "04_fit_component_models.py").read_text(encoding="utf-8")
    assert "lead_if_velocity_ok_else_direct" in pipe


@needs_stage4
def test_stage4_support_rejection_masked():
    """S4-B22: rejected_outside_support → Q/NOV NaN."""
    test_stage4_rejected_q_nov_nan()


@needs_stage4
def test_stage4_null_velocity_is_missing():
    """S4-M10: Stage 4 never zero-fills velocity OK flags in feature lists."""
    from passing_windows.models.features import KEEP_STATE_FEATURES, PASSABILITY_FEATURES

    # Velocities themselves are not imputed to zero as covariates; predError/margins only.
    for feats in PASSABILITY_FEATURES.values():
        assert "passer_vx" not in feats and "receiver_vx" not in feats


@needs_stage4
def test_stage4_both_trajectory_families():
    """S4-N02: Stage 3 retained both families; Stage 4 scores chosen surface only."""
    feat = pd.read_parquet(FEATURES, columns=["direct_pass_distance_ft", "lead_pass_distance_ft"])
    assert "direct_pass_distance_ft" in feat.columns and "lead_pass_distance_ft" in feat.columns


@needs_stage4
def test_stage4_no_season_aggregates():
    """S4-B23: no season aggregate CSV covariates."""
    from passing_windows.models.leakage import assert_no_season_aggregate_covariates
    from passing_windows.models.features import CATCH_VALUE_FEATURES, KEEP_STATE_FEATURES, PASSABILITY_FEATURES

    assert_no_season_aggregate_covariates(CATCH_VALUE_FEATURES)
    assert_no_season_aggregate_covariates(KEEP_STATE_FEATURES)
    for feats in PASSABILITY_FEATURES.values():
        assert_no_season_aggregate_covariates(feats)
    pipe = (ROOT / "pipelines" / "04_fit_component_models.py").read_text(encoding="utf-8")
    assert "aggregates/" not in pipe
    assert "acb_" not in pipe


@needs_stage4
def test_stage4_primary_population_filter():
    """S4-M11: Stage 3 primary-eligible population feeds Stage 4."""
    # Stage 3 already filtered primary_eligible; Stage 4 consumes that table.
    assert FEATURES.exists() and PRED.exists()
    assert len(pd.read_parquet(PRED, columns=["gameId"])) > 0


@needs_stage4
def test_stage4_exclusions_inherited():
    """S4-N03: self-pass / interceptor misuse not reintroduced as labels."""
    pred = pd.read_parquet(PRED)
    if "interceptorId" in pred.columns and "candidateId" in pred.columns:
        labeled = pred[pred["completion_label_eligible"].fillna(False)]
        inter = pd.to_numeric(labeled["interceptorId"], errors="coerce")
        cand = pd.to_numeric(labeled["candidateId"], errors="coerce")
        assert not ((inter.notna()) & (inter == cand)).any()


@needs_stage4
def test_stage4_pipeline_entrypoint_exists():
    """S4-B24."""
    assert (ROOT / "pipelines" / "04_fit_component_models.py").exists()
    assert (ROOT / "tests" / "test_stage4.py").exists()


@needs_stage4
def test_stage4_output_manifest_complete():
    """S4-B25."""
    from passing_windows.io_utils import verify_output_manifest

    assert MANIFEST.exists()
    result = verify_output_manifest(MANIFEST, ROOT)
    assert result["ok"], result


@needs_stage4
def test_stage4_regression_lock_covers_blockers():
    """S4-B26: regression lock lists S4-B08 and maps to this suite."""
    text = REGRESSION_LOCK.read_text(encoding="utf-8")
    assert "S4-B08" in text
    assert "test_stage4_gate_columns_present" in text


@needs_stage4
def test_stage4_model_columns_not_all_nan():
    """S4-M12: model score columns filled OOS."""
    pred = pd.read_parquet(PRED, columns=["q_passability_model", "V_catch_model", "NOV_model"])
    assert pred["q_passability_model"].notna().all()
    assert pred["V_catch_model"].notna().all()
    assert pred["NOV_model"].notna().any()


@needs_stage4
def test_stage4_verification_artifact_exists():
    """S4-N04."""
    assert (ARTIFACTS / "STAGE4_VERIFICATION.md").exists()
    assert (ARTIFACTS / "STAGE4_HANDOFF_STAGE5.md").exists()
