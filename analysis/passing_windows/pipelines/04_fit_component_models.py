#!/usr/bin/env python
"""Stage 4: fit cross-fitted component models (nested leave-one-game-out).

Restartable per held-out game. Fold training always sees the other nine freeze
games (even under ``--games``).

    python pipelines/04_fit_component_models.py
    python pipelines/04_fit_component_models.py --force
    python pipelines/04_fit_component_models.py --games 114086
    python pipelines/04_fit_component_models.py --verify-manifest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from passing_windows.ingest import load_freeze  # noqa: E402
from passing_windows.io_utils import (  # noqa: E402
    build_output_manifest,
    verify_output_manifest,
    write_json,
    write_markdown,
    write_table,
)
from passing_windows.models import MODEL_VERSION, PRIMARY_ABLATION, SCHEMA_VERSION  # noqa: E402
from passing_windows.models.catch_value import (  # noqa: E402
    evaluate_catch_predictions,
    fit_catch_value,
    observed_catch_training_mask,
    predict_catch_value,
)
from passing_windows.models.choice import evaluate_choice, fit_choice_model, predict_choice_probs  # noqa: E402
from passing_windows.models.features import (  # noqa: E402
    CATCH_VALUE_FEATURES,
    KEEP_STATE_FEATURES,
    PASSABILITY_FEATURES,
)
from passing_windows.models.keep_state import fit_keep_state, predict_keep_state  # noqa: E402
from passing_windows.models.leakage import (  # noqa: E402
    assert_completion_labels_narrow_path,
    assert_held_out_absent_from_train,
    assert_logo_only,
    assert_train_rows_exclude_held_out,
    assert_trajectory_choice_honored,
    scan_reports_for_forbidden_claims,
)
from passing_windows.models.metrics import binary_metrics, per_game_binary_metrics, regression_metrics  # noqa: E402
from passing_windows.models.option_value import attach_option_columns  # noqa: E402
from passing_windows.models.outcomes import (  # noqa: E402
    attach_next3s_outcomes,
    precompute_outcome_index,
)
from passing_windows.models.passability import (  # noqa: E402
    ablation_long_table,
    fit_all_passability_ablations,
    predict_passability_table,
)
from passing_windows.narrow_path import GATE_COLUMNS, load_narrow_path_lock  # noqa: E402
from passing_windows.paths import ARTIFACTS, PACKAGE_ROOT, TABLES  # noqa: E402

STAGE = 4
FREEZE_ID = "passing_windows_stage0_20250924"
# Binding Stage 3 trajectory rule — never silently score-max among families.
TRAJECTORY_RULE = "lead_if_velocity_ok_else_direct"
MARKER_NAME = "_DONE_STAGE4"
PRED_NAME = "stage4_candidate_predictions.parquet"
JOIN_KEYS = ["gameId", "touchId", "frameIdx", "candidateId"]


def _game_dir(game_id: int) -> Path:
    return TABLES / "by_game" / str(game_id)


def _fold_seed(freeze: dict[str, Any], fold_id: str) -> int:
    seeds = freeze.get("rng", {}).get("fold_seeds", {})
    return int(seeds.get(fold_id, 2025092400))


def load_stage3_features(game_ids: list[int] | None = None) -> pd.DataFrame:
    path = TABLES / "stage3_candidate_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run Stage 3 first")
    df = pd.read_parquet(path)
    if game_ids is not None:
        df = df[df["gameId"].isin(game_ids)].copy()
    return df


def load_outcomes() -> tuple[pd.DataFrame, pd.DataFrame]:
    shots = pd.read_parquet(TABLES / "shots.parquet")
    free_throws = pd.read_parquet(TABLES / "free_throws.parquet")
    return shots, free_throws


def process_fold(
    fold: dict[str, Any],
    features: pd.DataFrame,
    shots: pd.DataFrame,
    free_throws: pd.DataFrame,
    *,
    force: bool = False,
    random_state: int = 0,
    events_index: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_id = str(fold["fold_id"])
    held_out = int(fold["held_out_gameId"])
    train_ids = [int(g) for g in fold["train_gameIds"]]
    assert_logo_only("nested leave-one-game-out (LOGO) outer fold")
    assert_held_out_absent_from_train(train_ids, held_out)

    gdir = _game_dir(held_out)
    marker = gdir / MARKER_NAME
    out_path = gdir / PRED_NAME
    if marker.exists() and out_path.exists() and not force:
        marker_txt = marker.read_text(encoding="utf-8")
        if f"hash={MODEL_VERSION}" not in marker_txt:
            force = True
    if marker.exists() and out_path.exists() and not force:
        pred = pd.read_parquet(out_path)
        if "V_catch_ablation_all_candidates" not in pred.columns:
            force = True
    if marker.exists() and out_path.exists() and not force:
        pred = pd.read_parquet(out_path)
        # Ensure narrow-path gates survived older caches.
        missing_gates = [c for c in GATE_COLUMNS if c not in pred.columns]
        if missing_gates:
            feat_path = TABLES / "stage3_candidate_features.parquet"
            if feat_path.exists():
                feat = pd.read_parquet(feat_path)
                carry = JOIN_KEYS + [c for c in list(GATE_COLUMNS) + ["cohort", "true_receiverId", "interceptorId"] if c in feat.columns]
                pred = pred.drop(columns=[c for c in carry if c in pred.columns and c not in JOIN_KEYS], errors="ignore")
                pred = pred.merge(feat[carry], on=JOIN_KEYS, how="left")
                write_table(pred, out_path)
        summary_path = gdir / "stage4_game_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {
            "gameId": held_out,
            "fold_id": fold_id,
            "cached": True,
        }
        summary["cached"] = True
        return pred, summary

    t0 = time.time()
    train = features[features["gameId"].isin(train_ids)].copy()
    test = features[features["gameId"] == held_out].copy()
    if test.empty:
        raise ValueError(f"{fold_id}: no Stage 3 rows for held-out game {held_out}")
    assert_train_rows_exclude_held_out(train, held_out)
    assert_trajectory_choice_honored(test)

    labeled_train = train[train["completion_label_eligible"].fillna(False).astype(bool)]
    assert_completion_labels_narrow_path(labeled_train)

    # A. Passability (all ablations)
    pass_fits = fit_all_passability_ablations(
        train,
        fold_id=fold_id,
        held_out_gameId=held_out,
        random_state=random_state,
    )
    pass_pred = predict_passability_table(test, pass_fits, primary=PRIMARY_ABLATION)

    # B. Catch value (primary: stacked observed-catch; ablation: all-candidates ridge)
    catch_fit = fit_catch_value(
        train,
        shots,
        free_throws,
        fold_id=fold_id,
        held_out_gameId=held_out,
        random_state=random_state,
        events_index=events_index,
        training_population="observed_catch",
        model_family="stack",
    )
    catch_pred = predict_catch_value(test, catch_fit)
    catch_fit_all = fit_catch_value(
        train,
        shots,
        free_throws,
        fold_id=fold_id,
        held_out_gameId=held_out,
        random_state=random_state,
        events_index=events_index,
        training_population="all_candidates",
        model_family="ridge_splines",
        stack_inner_logo=False,
    )
    catch_pred_all = predict_catch_value(test, catch_fit_all)

    # C. Keep state
    keep_fit = fit_keep_state(
        train,
        shots,
        free_throws,
        fold_id=fold_id,
        held_out_gameId=held_out,
        random_state=random_state,
        events_index=events_index,
    )
    v_keep = predict_keep_state(test, keep_fit)

    # Merge predictions onto held-out candidates (carry narrow-path GATE_COLUMNS).
    carry = JOIN_KEYS + [
        "fold_id",
        "chanceId",
        "possessionId",
        "passId",
        "is_pass_release_frame",
        "is_model_5hz_frame",
        "is_forced_release_frame",
        "completion_label_eligible",
        "is_true_receiver",
        "rejected_outside_support",
        "trajectory_choice",
        "trajectory_choice_rule",
        "pass_distance_ft",
        "receiver_sep_ft",
        "dist_to_rim_ft",
        "geometry_passability_score",
        "flight_time_s",
        "sampling_role",
        "cohort",
        "true_receiverId",
        "interceptorId",
        *list(GATE_COLUMNS),
    ]
    carry = [c for c in carry if c in test.columns]
    pred = test[carry].copy()
    # Affirm Stage 3 trajectory rule (lead_if_velocity_ok_else_direct); never score-max.
    if "trajectory_choice_rule" in pred.columns:
        pred["trajectory_choice_rule"] = TRAJECTORY_RULE
    pred["fold_id"] = fold_id
    pred["model_version"] = MODEL_VERSION
    pred["stage4_schema_version"] = SCHEMA_VERSION

    for c in pass_pred.columns:
        if c not in JOIN_KEYS:
            pred[c] = pass_pred[c].to_numpy()
    pred["V_catch_model"] = catch_pred["V_catch_model"].to_numpy()
    pred["V_catch_ablation_all_candidates"] = catch_pred_all["V_catch_model"].to_numpy()
    for c in catch_pred.columns:
        if c.startswith("V_catch_ablation_") or c.startswith("V_catch_sec_"):
            pred[c] = catch_pred[c].to_numpy()
    pred["V_keep_model"] = v_keep.to_numpy()
    pred = attach_option_columns(pred, mask_rejected_for_counterfactual=True)

    # E. Choice model — score train release rows only (not full train matrix)
    release_mask = train["is_pass_release_frame"].fillna(False).astype(bool)
    train_release = train.loc[release_mask].copy()
    train_pass = predict_passability_table(train_release, pass_fits, primary=PRIMARY_ABLATION)
    train_catch = predict_catch_value(train_release, catch_fit)
    train_keep = predict_keep_state(train_release, keep_fit)
    train_enriched = train_release[JOIN_KEYS + [
        "passId",
        "is_pass_release_frame",
        "completion_label_eligible",
        "is_true_receiver",
        "rejected_outside_support",
        "pass_distance_ft",
        "receiver_sep_ft",
        "dist_to_rim_ft",
        "geometry_passability_score",
    ]].copy()
    train_enriched["q_passability_model"] = train_pass["q_passability_model"].to_numpy()
    train_enriched["V_catch_model"] = train_catch["V_catch_model"].to_numpy()
    train_enriched["V_keep_model"] = train_keep.to_numpy()
    train_enriched = attach_option_columns(train_enriched, mask_rejected_for_counterfactual=False)
    choice_fit = fit_choice_model(
        train_enriched,
        fold_id=fold_id,
        held_out_gameId=held_out,
        random_state=random_state,
    )
    pred["choice_prob_model"] = predict_choice_probs(pred, choice_fit).to_numpy()

    # Held-out metrics
    labeled_test = pred[pred["completion_label_eligible"].fillna(False).astype(bool)]
    holdout_pass = binary_metrics(
        labeled_test["is_true_receiver"].fillna(False).astype(int).to_numpy(),
        labeled_test["q_passability_model"].to_numpy(),
    ) if len(labeled_test) else {"n": 0.0}

    test_catch_y = attach_next3s_outcomes(
        test, shots, free_throws, use_projected_catch=True, events_index=events_index
    )
    obs_mask = observed_catch_training_mask(test_catch_y)
    holdout_catch = evaluate_catch_predictions(
        test_catch_y.loc[obs_mask, "catch_points_next_3s"].to_numpy(),
        pred.loc[obs_mask, "V_catch_model"].to_numpy(),
    )
    holdout_catch_all = evaluate_catch_predictions(
        test_catch_y.loc[obs_mask, "catch_points_next_3s"].to_numpy(),
        pred.loc[obs_mask, "V_catch_ablation_all_candidates"].to_numpy(),
    )
    holdout_catch_all_rows = evaluate_catch_predictions(
        test_catch_y["catch_points_next_3s"].to_numpy(),
        pred["V_catch_model"].to_numpy(),
    )

    choice_holdout = evaluate_choice(pred, choice_fit)

    ablation_holdout: dict[str, Any] = {}
    for name, fit in pass_fits.items():
        col = "q_passability_model" if name == PRIMARY_ABLATION else f"q_ablation_{name}"
        if col in labeled_test.columns and len(labeled_test):
            ablation_holdout[name] = binary_metrics(
                labeled_test["is_true_receiver"].fillna(False).astype(int).to_numpy(),
                labeled_test[col].to_numpy(),
            )

    write_table(pred, out_path)
    summary = {
        "gameId": held_out,
        "fold_id": fold_id,
        "cached": False,
        "elapsed_seconds": round(time.time() - t0, 2),
        "n_rows": int(len(pred)),
        "n_completion_eligible": int(pred["completion_label_eligible"].fillna(False).sum()),
        "n_true_receiver": int(pred["is_true_receiver"].fillna(False).sum()),
        "n_rejected_outside_support": int(pred["rejected_outside_support"].fillna(False).sum()),
        "passability_train": {
            name: {
                "n_train": fit.n_train,
                "n_pos": fit.n_pos,
                "train_cal": fit.train_metrics_cal,
            }
            for name, fit in pass_fits.items()
        },
        "passability_holdout": holdout_pass,
        "passability_ablation_holdout": ablation_holdout,
        "catch_train": catch_fit.train_metrics,
        "catch_holdout": holdout_catch,
        "catch_holdout_ablation_all_candidates_on_observed_mask": holdout_catch_all,
        "catch_holdout_all_candidate_rows": holdout_catch_all_rows,
        "catch_train_n": catch_fit.n_train,
        "catch_train_population": catch_fit.training_population,
        "catch_model_family": catch_fit.model_family,
        "catch_blend_weights": catch_fit.blend_weights,
        "catch_family_train_metrics": catch_fit.family_train_metrics,
        "keep_train": keep_fit.train_metrics,
        "choice_train": choice_fit.train_metrics,
        "choice_holdout": choice_holdout,
        "model_version": MODEL_VERSION,
    }
    write_json(summary, gdir / "stage4_game_summary.json")
    marker.write_text(
        f"hash={MODEL_VERSION}\nschema={SCHEMA_VERSION}\nfold={fold_id}\n",
        encoding="utf-8",
    )
    return pred, summary


def render_model_report(
    fold_metrics: dict[str, Any],
    game_summaries: list[dict[str, Any]],
    lock: dict[str, Any],
    n_rows: int,
) -> str:
    overall = fold_metrics.get("overall_passability", {})
    lines = [
        "# Stage 4 component model report",
        "",
        f"**Freeze ID:** `{FREEZE_ID}`  ",
        f"**Narrow-path lock:** `{lock.get('lock_id')}`  ",
        f"**Model version:** `{MODEL_VERSION}`  ",
        f"**Validation:** nested leave-one-game-out (never random frame/touch/chance splits)",
        "",
        "## Population",
        "",
        f"- Candidate prediction rows: **{n_rows}**",
        f"- Games / outer folds: **{len(game_summaries)}**",
        f"- Primary passability ablation: `{PRIMARY_ABLATION}`",
        "",
        "## Passability (held-out, completion_label_eligible)",
        "",
        f"- Log loss: **{overall.get('log_loss', float('nan')):.4f}**",
        f"- Brier: **{overall.get('brier', float('nan')):.4f}**",
        f"- Calibration slope / intercept: "
        f"**{overall.get('calibration_slope', float('nan')):.3f}** / "
        f"**{overall.get('calibration_intercept', float('nan')):.3f}**",
        f"- N labeled: **{int(overall.get('n', 0))}** (base rate "
        f"{overall.get('base_rate', float('nan')):.3f})",
        "",
        "### Ablation held-out log loss",
        "",
        "| Ablation | Log loss | Brier | N |",
        "|---|---:|---:|---:|",
    ]
    for name, m in fold_metrics.get("ablation_overall", {}).items():
        lines.append(
            f"| `{name}` | {m.get('log_loss', float('nan')):.4f} | "
            f"{m.get('brier', float('nan')):.4f} | {int(m.get('n', 0))} |"
        )
    catch = fold_metrics.get("overall_catch", {})
    catch_abl = fold_metrics.get("overall_catch_ablation_all_candidates_on_observed_mask", {})
    choice = fold_metrics.get("overall_choice", {})
    lines.extend(
        [
            "",
            "## Catch value V_catch (held-out, observed-catch mask)",
            "",
            "- Training population: true-receiver release ∪ ±0.4s 5 Hz true-receiver rows",
            "- Primary predictor: stacked ridge + HistGBRT + two-stage shot composition",
            f"- RMSE: **{catch.get('rmse', float('nan')):.4f}**",
            f"- MAE: **{catch.get('mae', float('nan')):.4f}**",
            f"- R²: **{catch.get('r2', float('nan')):.4f}**",
            f"- Spearman: **{catch.get('spearman', float('nan')):.4f}**",
            f"- N: **{int(catch.get('n', 0))}**",
            f"- Ablation all-candidates ridge on same mask — RMSE: "
            f"**{catch_abl.get('rmse', float('nan')):.4f}**, R²: "
            f"**{catch_abl.get('r2', float('nan')):.4f}**",
            "- Outcome source: shots + free_throws next-3s points (not chances.ptsScored)",
            "",
            "### Family ablations on observed-catch mask",
            "",
            "| Family | RMSE | MAE | R² | Spearman | N |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, m in fold_metrics.get("catch_family_holdout", {}).items():
        lines.append(
            f"| `{name}` | {m.get('rmse', float('nan')):.4f} | "
            f"{m.get('mae', float('nan')):.4f} | {m.get('r2', float('nan')):.4f} | "
            f"{m.get('spearman', float('nan')):.4f} | {int(m.get('n', 0))} |"
        )
    lines.extend(
        [
            "",
            "### Per-game observed-catch V_catch",
            "",
            "| Game | RMSE | MAE | R² | Spearman | N |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fold_metrics.get("per_game_catch_observed", []):
        lines.append(
            f"| {row.get('gameId')} | {row.get('rmse', float('nan')):.4f} | "
            f"{row.get('mae', float('nan')):.4f} | {row.get('r2', float('nan')):.4f} | "
            f"{row.get('spearman', float('nan')):.4f} | {int(row.get('n', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Choice model (held-out release states)",
            "",
            f"- Rank accuracy: **{choice.get('rank_accuracy', float('nan')):.3f}**",
            f"- Choice log loss: **{choice.get('choice_log_loss', float('nan')):.4f}**",
            f"- N release sets: **{int(choice.get('n_releases', 0))}**",
            "",
            "## Formulas",
            "",
            "- `Q = q * V_catch + (1 - q) * V_fail` with primary `V_fail = 0`",
            "- `NOV = Q - V_keep`",
            "- Counterfactual `Q`/`NOV` masked to NaN where `rejected_outside_support`",
            "",
            "## Modeling notes",
            "",
            "- Passability: L2 logistic + B-spline transforms (sklearn SplineTransformer;",
            "  freeze GAM substituted — see STAGE4_VERIFICATION.md)",
            "- Loss weighting for class imbalance applied **inside training folds only**",
            "- Fold-internal logistic (Platt) recalibration of passability scores",
            "- Narrow-path: completion labels only on `completion_label_eligible`",
            "- V_catch v3: observed-catch training + stacked ridge/HistGBRT/two-stage",
            "  (inner-LOGO non-negative blend); `V_catch_ablation_*` retain components",
            "",
            "## Observed-catch accept check (design §2)",
            "",
        ]
    )
    try:
        delta_r2 = float(catch.get("r2", float("nan"))) - float(catch_abl.get("r2", float("nan")))
        rmse_rel = 1.0 - (
            float(catch.get("rmse", float("nan"))) / float(catch_abl.get("rmse", float("nan")))
        )
    except (TypeError, ValueError, ZeroDivisionError):
        delta_r2 = float("nan")
        rmse_rel = float("nan")
    lines.extend(
        [
            f"- ΔR² vs all-candidate ablation on same mask: **{delta_r2:.4f}** (accept ≥ +0.05)",
            f"- Relative RMSE drop vs ablation: **{rmse_rel:.4f}** (accept ≥ 0.10)",
            "- Per-game R² wins vs ablation: see table above (target ≥ 7/10)",
            "",
            "## Per-game passability",
            "",
            "| Game | Log loss | Brier | Slope | N |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fold_metrics.get("per_game_passability", []):
        lines.append(
            f"| {row.get('gameId')} | {row.get('log_loss', float('nan')):.4f} | "
            f"{row.get('brier', float('nan')):.4f} | "
            f"{row.get('calibration_slope', float('nan')):.3f} | {int(row.get('n', 0))} |"
        )
    lines.extend(["", "## Game run summaries", ""])
    for gs in game_summaries:
        lines.append(
            f"- game `{gs.get('gameId')}` fold `{gs.get('fold_id')}`: "
            f"{gs.get('n_rows')} rows, cached={gs.get('cached')}, "
            f"elapsed={gs.get('elapsed_seconds', 'n/a')}s"
        )
    lines.append("")
    return "\n".join(lines)


def render_verification(fold_metrics: dict[str, Any], n_rows: int) -> str:
    return "\n".join(
        [
            "# Stage 4 verification",
            "",
            f"**Model version:** `{MODEL_VERSION}`",
            "",
            "## Freeze faithfulness",
            "",
            "- [x] Nested leave-one-game-out only (no random frame/touch/chance splits)",
            "- [x] Passability trained on `completion_label_eligible`; label = `is_true_receiver`",
            "- [x] Predictions written for all candidate rows (5 Hz + release)",
            "- [x] Ablations: distance_only, nearest_defender_distance,",
            "      static_lane_geometry_no_velocity, kinematic_geometry_no_uncertainty, full_model",
            "- [x] Imbalance loss weights inside train folds only",
            "- [x] Fold-internal logistic recalibration",
            "- [x] V_catch from shots+free_throws next-3s points (not chances.ptsScored)",
            "- [x] V_catch features at projected catch only",
            "- [x] V_catch trained on observed-catch mask (true receiver ±0.4s); scored on all candidates",
            "- [x] V_keep from no-pass / keep states, causal features only",
            "- [x] Q = q*V_catch; V_fail = 0; NOV = Q - V_keep",
            "- [x] Choice model at pass-release over four candidates (descriptive)",
            "- [x] `rejected_outside_support` masked for counterfactual Q/NOV",
            "- [x] Never use interceptorId as intended target",
            "- [x] Never use season aggregates as covariates",
            "- [x] Trajectory choice honored (not score-max)",
            "",
            "## GAM → sklearn substitution",
            "",
            "The freeze specifies a regularized logistic GAM. This Stage 4 ship uses",
            "`sklearn.preprocessing.SplineTransformer` (B-splines, 5 knots, degree 3)",
            "on distance, clearance, temporal margin, separation, pressure, and",
            "`predError`, followed by L2 `LogisticRegression` / `Ridge`. This is a",
            "documented pragmatic substitute; coefficients are not GAM smooths.",
            "",
            "## V_catch training population (v3)",
            "",
            "`stage4_sklearn_logo_v3` trains `V_catch` on observed true-receiver states",
            "(completion-eligible release ∪ `is_model_5hz_frame` within ±0.4s).",
            "Primary score is a non-negative blend of ridge+splines, HistGradientBoosting,",
            "and two-stage P(any_shot)×E[points|shot] with blend weights from leave-one-game",
            "OOF inside each outer training fold. Predictions remain on all candidate rows.",
            "",
            "## Key metrics snapshot",
            "",
            f"- Rows: {n_rows}",
            f"- Passability log loss: {fold_metrics.get('overall_passability', {}).get('log_loss')}",
            f"- Passability Brier: {fold_metrics.get('overall_passability', {}).get('brier')}",
            f"- V_catch observed-catch R²: {fold_metrics.get('overall_catch', {}).get('r2')}",
            f"- V_catch observed-catch RMSE: {fold_metrics.get('overall_catch', {}).get('rmse')}",
            f"- Choice rank accuracy: {fold_metrics.get('overall_choice', {}).get('rank_accuracy')}",
            "",
            "## Residual risks",
            "",
            "- Secondary catch outcomes use linear probability Ridge, not multinomial GAM.",
            "- Choice model is a conditional-softmax over independent utilities (descriptive).",
            "- Catch/keep training may subsample large training folds for runtime.",
            "- Observed-catch labels remain associative (actual possession path), not causal",
            "  counterfactuals for unused options.",
            "",
        ]
    )


def render_handoff(n_rows: int, fold_metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 4 → Stage 5 handoff",
            "",
            f"**Status:** Stage 4 component models complete (`{MODEL_VERSION}`).",
            "",
            "## Primary table Stage 5 must consume",
            "",
            "`tables/stage4_candidate_predictions.parquet`",
            "",
            "Join keys: `(gameId, touchId, frameIdx, candidateId)` — aligns 1:1 with",
            "`tables/stage3_candidate_features.parquet`.",
            "",
            "### Required score columns",
            "",
            "| Column | Meaning |",
            "|---|---|",
            "| `q_passability_model` | Calibrated P(complete to j \\| state) |",
            "| `q_ablation_*` | Ablation probabilities |",
            "| `V_catch_model` | Stacked E[next-3s points \\| projected catch]; observed-catch train |",
            "| `V_catch_ablation_ridge_splines` / `_hist_gbrt` / `_two_stage` | Family components |",
            "| `V_catch_ablation_all_candidates` | All-candidate ridge sensitivity |",
            "| `V_keep_model` | E[next-3s points \\| keep / no-pass state] |",
            "| `V_fail` | Primary 0 |",
            "| `Q_option_model` | q·V_catch (NaN if rejected_outside_support) |",
            "| `NOV_model` | Q − V_keep (NaN if rejected) |",
            "| `choice_prob_model` | Softmax choice prob at release (else NaN) |",
            "| `fold_id` | Outer LOGO fold |",
            "| `model_version` | Model version flag |",
            "",
            f"- Rows: **{n_rows}**",
            f"- Held-out passability log loss: "
            f"**{fold_metrics.get('overall_passability', {}).get('log_loss')}**",
            f"- Choice rank accuracy: "
            f"**{fold_metrics.get('overall_choice', {}).get('rank_accuracy')}**",
            "",
            "## Dual sampling (unchanged from Stage 3)",
            "",
            "- Window series: `is_model_5hz_frame`",
            "- Passability / choice labels: `completion_label_eligible` / release frames",
            "",
            "## Stage 5 scope",
            "",
            "Smooth cross-fitted `q` / `V_catch` / `NOV` series and segment windows.",
            "Do not refit component models. Open threshold remains TBD_TRAINING_FOLD.",
            "",
            "## Binding constraints carried forward",
            "",
            "1. Narrow-path completion labels only.",
            "2. Mask `rejected_outside_support` for counterfactual claims.",
            "3. Report q and V_catch separately before collapsing to NOV.",
            "4. Nested LOGO only.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if _DONE_STAGE4 exists")
    parser.add_argument(
        "--games",
        nargs="+",
        type=int,
        default=None,
        help="limit held-out games to process (training still uses other nine)",
    )
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="re-verify stage4_output_manifest.json and exit",
    )
    args = parser.parse_args(argv)

    freeze = load_freeze()
    lock = load_narrow_path_lock()
    folds = freeze["validation"]["outer_folds"]

    if args.verify_manifest:
        manifest_path = ARTIFACTS / "stage4_output_manifest.json"
        result = verify_output_manifest(manifest_path, PACKAGE_ROOT)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2

    features = load_stage3_features()
    shots, free_throws = load_outcomes()
    events_index = precompute_outcome_index(shots, free_throws)
    print(f"[stage4] loaded {len(features)} candidate rows; scoring events indexed")

    selected_folds = folds
    if args.games:
        want = {int(g) for g in args.games}
        selected_folds = [f for f in folds if int(f["held_out_gameId"]) in want]
        if not selected_folds:
            print(f"no folds match --games {args.games}", file=sys.stderr)
            return 2

    game_summaries: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []
    errors: list[str] = []

    for fold in selected_folds:
        held_out = int(fold["held_out_gameId"])
        seed = _fold_seed(freeze, str(fold["fold_id"]))
        try:
            pred, summary = process_fold(
                fold,
                features,
                shots,
                free_throws,
                force=args.force,
                random_state=seed,
                events_index=events_index,
            )
            pred_parts.append(pred)
            game_summaries.append(summary)
            print(
                f"[stage4] fold {fold['fold_id']} game {held_out}: "
                f"{summary.get('n_rows')} rows "
                f"pass_ll={summary.get('passability_holdout', {}).get('log_loss')} "
                f"cached={summary.get('cached')}"
            )
        except Exception as exc:  # noqa: BLE001 — surface per-fold, continue others
            errors.append(f"{fold['fold_id']}: {exc}")
            traceback.print_exc()
            print(f"[stage4] FAIL {fold['fold_id']}: {exc}", file=sys.stderr)

    if not pred_parts:
        print("no Stage 4 predictions produced", file=sys.stderr)
        return 2

    # If partial --games run, merge with any existing full table rows for other games.
    all_pred = pd.concat(pred_parts, ignore_index=True)
    out_path = TABLES / "stage4_candidate_predictions.parquet"
    if args.games and out_path.exists() and not args.force:
        existing = pd.read_parquet(out_path)
        keep = existing[~existing["gameId"].isin(all_pred["gameId"].unique())]
        all_pred = pd.concat([keep, all_pred], ignore_index=True)
    elif args.games and out_path.exists() and args.force:
        existing = pd.read_parquet(out_path)
        keep = existing[~existing["gameId"].isin(all_pred["gameId"].unique())]
        all_pred = pd.concat([keep, all_pred], ignore_index=True)

    all_pred = all_pred.sort_values(JOIN_KEYS).reset_index(drop=True)
    write_table(all_pred, out_path)

    # Ablation long table
    abl = ablation_long_table(all_pred)
    write_table(abl, TABLES / "stage4_passability_ablations_long.parquet")

    # Aggregate metrics on labeled holdout
    labeled = all_pred[all_pred["completion_label_eligible"].fillna(False).astype(bool)]
    overall_pass = binary_metrics(
        labeled["is_true_receiver"].fillna(False).astype(int).to_numpy(),
        labeled["q_passability_model"].to_numpy(),
    ) if len(labeled) else {}
    per_game = per_game_binary_metrics(
        labeled, y_col="is_true_receiver", p_col="q_passability_model"
    ) if len(labeled) else []

    ablation_overall: dict[str, Any] = {}
    for name in PASSABILITY_FEATURES:
        col = "q_passability_model" if name == PRIMARY_ABLATION else f"q_ablation_{name}"
        if col in labeled.columns and len(labeled):
            ablation_overall[name] = binary_metrics(
                labeled["is_true_receiver"].fillna(False).astype(int).to_numpy(),
                labeled[col].to_numpy(),
            )

    # Catch overall: primary metrics on observed-catch mask; diagnostics retained
    catch_y = attach_next3s_outcomes(
        features, shots, free_throws, use_projected_catch=True, events_index=events_index
    )
    merged_catch = all_pred[
        JOIN_KEYS
        + [
            c
            for c in all_pred.columns
            if c == "V_catch_model" or c.startswith("V_catch_ablation_")
        ]
    ].merge(
        catch_y[
            JOIN_KEYS
            + [
                "catch_points_next_3s",
                "completion_label_eligible",
                "is_true_receiver",
                "is_model_5hz_frame",
                "is_pass_release_frame",
            ]
        ],
        on=JOIN_KEYS,
        how="left",
    )
    # Reconstruct mask columns on merged table (join may drop frame flags if missing)
    for col in ("completion_label_eligible", "is_true_receiver", "is_model_5hz_frame", "is_pass_release_frame"):
        if col not in merged_catch.columns:
            merged_catch[col] = False
    obs_mask = observed_catch_training_mask(merged_catch)
    overall_catch = evaluate_catch_predictions(
        merged_catch.loc[obs_mask, "catch_points_next_3s"].to_numpy(),
        merged_catch.loc[obs_mask, "V_catch_model"].to_numpy(),
    )
    overall_catch_ablation = evaluate_catch_predictions(
        merged_catch.loc[obs_mask, "catch_points_next_3s"].to_numpy(),
        merged_catch.loc[obs_mask, "V_catch_ablation_all_candidates"].to_numpy(),
    ) if "V_catch_ablation_all_candidates" in merged_catch.columns else {}
    overall_catch_all_rows = evaluate_catch_predictions(
        merged_catch["catch_points_next_3s"].to_numpy(),
        merged_catch["V_catch_model"].to_numpy(),
    )
    catch_family_holdout: dict[str, Any] = {}
    for fam in ("ridge_splines", "hist_gbrt", "two_stage"):
        col = f"V_catch_ablation_{fam}"
        if col in merged_catch.columns:
            catch_family_holdout[fam] = evaluate_catch_predictions(
                merged_catch.loc[obs_mask, "catch_points_next_3s"].to_numpy(),
                merged_catch.loc[obs_mask, col].to_numpy(),
            )
    per_game_catch: list[dict[str, Any]] = []
    for gid, grp in merged_catch.loc[obs_mask].groupby("gameId", sort=True):
        m = evaluate_catch_predictions(
            grp["catch_points_next_3s"].to_numpy(),
            grp["V_catch_model"].to_numpy(),
        )
        m["gameId"] = int(gid)
        per_game_catch.append(m)

    # Choice overall from game summaries
    choice_parts = [gs.get("choice_holdout", {}) for gs in game_summaries if not gs.get("cached")]
    if not choice_parts:
        choice_parts = [gs.get("choice_holdout", {}) for gs in game_summaries]
    n_rel = sum(int(c.get("n_releases", 0) or 0) for c in choice_parts)
    # Recompute choice metrics on full table requires refit — use weighted avg of holdouts.
    rank_vals = [c.get("rank_accuracy") for c in choice_parts if c.get("rank_accuracy") == c.get("rank_accuracy")]
    ll_vals = [c.get("choice_log_loss") for c in choice_parts if c.get("choice_log_loss") == c.get("choice_log_loss")]
    # Prefer direct recompute via stored choice_prob if available
    release_labeled = labeled[labeled["is_pass_release_frame"].fillna(False).astype(bool)]
    overall_choice: dict[str, Any] = {
        "n_releases": float(n_rel),
        "rank_accuracy": float(np.nanmean(rank_vals)) if rank_vals else float("nan"),
        "choice_log_loss": float(np.nanmean(ll_vals)) if ll_vals else float("nan"),
    }
    if "choice_prob_model" in release_labeled.columns and len(release_labeled):
        # Rank accuracy: max choice_prob among group equals true receiver
        n_ok = n_g = 0
        losses = []
        gcols = ["gameId", "passId"] if "passId" in release_labeled.columns else ["gameId", "touchId", "frameIdx"]
        for _, grp in release_labeled.groupby(gcols, sort=False):
            if grp["is_true_receiver"].fillna(False).sum() != 1:
                continue
            n_g += 1
            probs = grp["choice_prob_model"].to_numpy(dtype=float)
            if not np.all(np.isfinite(probs)):
                continue
            true_idx = int(np.argmax(grp["is_true_receiver"].fillna(False).astype(int).to_numpy()))
            if int(np.argmax(probs)) == true_idx:
                n_ok += 1
            losses.append(-np.log(max(float(probs[true_idx]), 1e-12)))
        if n_g:
            overall_choice = {
                "n_releases": float(n_g),
                "rank_accuracy": float(n_ok / n_g),
                "choice_log_loss": float(np.mean(losses)) if losses else float("nan"),
            }

    fold_metrics = {
        "model_version": MODEL_VERSION,
        "freeze_id": FREEZE_ID,
        "n_rows": int(len(all_pred)),
        "overall_passability": overall_pass,
        "per_game_passability": per_game,
        "ablation_overall": ablation_overall,
        "overall_catch": overall_catch,
        "overall_catch_ablation_all_candidates_on_observed_mask": overall_catch_ablation,
        "overall_catch_all_candidate_rows": overall_catch_all_rows,
        "catch_family_holdout": catch_family_holdout,
        "per_game_catch_observed": per_game_catch,
        "overall_choice": overall_choice,
        "feature_sets": {
            "passability": {k: list(v) for k, v in PASSABILITY_FEATURES.items()},
            "catch_value": list(CATCH_VALUE_FEATURES),
            "keep_state": list(KEEP_STATE_FEATURES),
        },
        "game_summaries": game_summaries,
        "errors": errors,
    }
    write_json(fold_metrics, ARTIFACTS / "stage4_fold_metrics.json")

    report = render_model_report(fold_metrics, game_summaries, lock, len(all_pred))
    verification = render_verification(fold_metrics, len(all_pred))
    handoff = render_handoff(len(all_pred), fold_metrics)
    write_markdown(report, ARTIFACTS / "stage4_model_report.md")
    write_markdown(verification, ARTIFACTS / "STAGE4_VERIFICATION.md")
    write_markdown(handoff, ARTIFACTS / "STAGE4_HANDOFF_STAGE5.md")

    # Regression lock bootstrap
    lock_path = ARTIFACTS / "STAGE4_REGRESSION_LOCK.md"
    if not lock_path.exists():
        write_markdown(
            "\n".join(
                [
                    "# Stage 4 regression lock",
                    "",
                    "Append every auditor finding ID and the regression test that prevents recurrence.",
                    "",
                    "| Finding ID | Summary | Regression test |",
                    "|---|---|---|",
                    "| S4-BOOTSTRAP | Stage 4 initial ship | `tests/test_stage4.py` |",
                    "",
                ]
            ),
            lock_path,
        )

    scan_reports_for_forbidden_claims([report, verification, handoff])

    manifest_paths = [
        out_path,
        TABLES / "stage4_passability_ablations_long.parquet",
        ARTIFACTS / "stage4_fold_metrics.json",
        ARTIFACTS / "stage4_model_report.md",
        ARTIFACTS / "STAGE4_VERIFICATION.md",
        ARTIFACTS / "STAGE4_HANDOFF_STAGE5.md",
        ARTIFACTS / "STAGE4_REGRESSION_LOCK.md",
    ]
    for gs in game_summaries:
        gid = int(gs["gameId"])
        manifest_paths.append(_game_dir(gid) / PRED_NAME)
        manifest_paths.append(_game_dir(gid) / "stage4_game_summary.json")

    manifest = build_output_manifest(
        manifest_paths,
        root=PACKAGE_ROOT,
        manifest_path=ARTIFACTS / "stage4_output_manifest.json",
    )
    write_json(manifest, ARTIFACTS / "stage4_output_manifest.json")

    write_json(
        {"games": game_summaries, "n_rows": len(all_pred), "errors": errors},
        ARTIFACTS / "stage4_game_summaries.json",
    )

    if errors:
        print(f"[stage4] completed with {len(errors)} fold errors", file=sys.stderr)
        return 2
    print(f"[stage4] wrote {len(all_pred)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
