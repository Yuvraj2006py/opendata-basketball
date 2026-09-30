"""Tests for Stage 3 candidate reconstruction.

Unit tests are synthetic and always run. Tests marked ``needs_stage3`` read
tables produced by ``pipelines/03_build_candidate_states.py`` and skip when
the pipeline has not been run.
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

from passing_windows.candidates import (
    TRAJECTORY_DIRECT,
    TRAJECTORY_LEAD,
    TRAJECTORY_RULE,
    attach_narrow_path_labels,
    choose_trajectory_explicit,
    reconstruct_game_candidates,
    reconstruct_one_candidate,
    summarize_candidates,
)
from passing_windows.flight_time import (
    DistanceSupport,
    FoldFlightBundle,
    fit_distance_support,
    fit_fold_flight_bundles,
    outside_distance_support,
    predict_flight_time,
)
from passing_windows.lane_geometry import (
    geometry_only_passability_score,
    project_receiver_catch,
)
from passing_windows.narrow_path import (
    GATE_COLUMNS,
    assert_interceptor_never_target,
    load_narrow_path_lock,
    receiver_completion_training_labels,
)
from passing_windows.paths import ARTIFACTS, TABLES
from passing_windows.target_audit import synthetic_flight_time_model

FEATURES = TABLES / "stage3_candidate_features.parquet"
ELIGIBILITY = TABLES / "stage2_label_eligibility.parquet"
DONE_MARKERS = list((TABLES / "by_game").glob("*/_DONE_STAGE3")) if (TABLES / "by_game").exists() else []
MANIFEST = ARTIFACTS / "stage3_output_manifest.json"

needs_stage3 = pytest.mark.skipif(
    not FEATURES.exists(), reason="Stage 3 pipeline not run"
)
needs_eligibility = pytest.mark.skipif(
    not ELIGIBILITY.exists(), reason="Stage 2 eligibility table not built"
)


def _bundle(
    *,
    held_out: int = 1,
    train: tuple[int, ...] = (2, 3),
    d_min: float = 1.0,
    d_max: float = 40.0,
) -> FoldFlightBundle:
    model = synthetic_flight_time_model()
    return FoldFlightBundle(
        fold_id=f"logo_holdout_{held_out}",
        held_out_gameId=held_out,
        train_gameIds=train,
        flight_model=model,
        distance_support=DistanceSupport(d_min, d_max, 0.99, 100),
        n_train_completed=50,
    )


def _player(
    pid: int,
    x: float,
    y: float,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    velocity_ok: bool = True,
    pred_error: float = 0.1,
) -> dict:
    return {
        "playerId": pid,
        "x": x,
        "y": y,
        "vx": vx if velocity_ok else float("nan"),
        "vy": vy if velocity_ok else float("nan"),
        "velocity_ok": velocity_ok,
        "predError": pred_error,
        "isDetected": True,
    }


# ---------------------------------------------------------------------------
# Explicit trajectory rule (never score-maximizing)
# ---------------------------------------------------------------------------


def test_trajectory_rule_constant_and_explicit():
    assert TRAJECTORY_RULE == "lead_if_velocity_ok_else_direct"
    assert choose_trajectory_explicit(receiver_velocity_ok=True) == TRAJECTORY_LEAD
    assert choose_trajectory_explicit(receiver_velocity_ok=False) == TRAJECTORY_DIRECT


def test_reconstruct_keeps_both_trajectories_and_explicit_choice():
    passer = _player(1, 0.0, 0.0)
    receiver = _player(2, 15.0, 0.0, vx=5.0, vy=0.0, velocity_ok=True)
    defenders = [_player(21, 8.0, 3.0)]
    out = reconstruct_one_candidate(
        passer=passer,
        receiver=receiver,
        defenders=defenders,
        assigned_defender=defenders[0],
        matchup_matched=True,
        flight_bundle=_bundle(),
    )
    assert out["trajectory_choice"] == TRAJECTORY_LEAD
    assert out["trajectory_choice_rule"] == TRAJECTORY_RULE
    assert "direct_geometry_passability_score" in out
    assert "lead_geometry_passability_score" in out
    # Choice is rule-driven, not argmax of geometry scores.
    direct_s = out["direct_geometry_passability_score"]
    lead_s = out["lead_geometry_passability_score"]
    if np.isfinite(direct_s) and np.isfinite(lead_s) and direct_s > lead_s:
        assert out["trajectory_choice"] == TRAJECTORY_LEAD  # still lead


def test_missing_velocity_forces_direct_and_stays_nan():
    passer = _player(1, 0.0, 0.0)
    receiver = _player(2, 12.0, 2.0, velocity_ok=False)
    out = reconstruct_one_candidate(
        passer=passer,
        receiver=receiver,
        defenders=[_player(21, 6.0, 1.0)],
        assigned_defender=None,
        matchup_matched=False,
        flight_bundle=_bundle(),
    )
    assert out["trajectory_choice"] == TRAJECTORY_DIRECT
    assert np.isnan(out["receiver_vx"])
    assert np.isnan(out["receiver_vy"])
    assert out["receiver_velocity_ok"] is False
    assert np.isnan(out["target_defender_recovery_s"])
    assert out["matchup_matched"] is False


def test_project_receiver_catch_does_not_zero_missing_velocity():
    px, py, used, _ = project_receiver_catch(
        10.0, 5.0, float("nan"), float("nan"), 0.5, velocity_ok=False
    )
    assert (px, py) == (10.0, 5.0)
    assert used is False


# ---------------------------------------------------------------------------
# Support rejection
# ---------------------------------------------------------------------------


def test_outside_distance_support_rejects_extremes():
    support = DistanceSupport(2.0, 30.0, 0.99, 50)
    assert bool(outside_distance_support(1.0, support))
    assert bool(outside_distance_support(35.0, support))
    assert not bool(outside_distance_support(15.0, support))


def test_reconstruct_rejects_beyond_fold_support():
    passer = _player(1, 0.0, 0.0)
    # ~45 ft — outside [1, 40]
    receiver = _player(2, 45.0, 0.0)
    out = reconstruct_one_candidate(
        passer=passer,
        receiver=receiver,
        defenders=[],
        assigned_defender=None,
        matchup_matched=False,
        flight_bundle=_bundle(d_max=40.0),
    )
    assert out["rejected_outside_support"] is True
    assert out["direct_rejected_outside_support"] is True
    assert np.isnan(out["geometry_passability_score"])


def test_geometry_score_nan_when_outside_support():
    score = geometry_only_passability_score(
        distance_ft=50.0,
        min_clearance_ft=5.0,
        min_temporal_margin_s=0.2,
        receiver_sep_ft=8.0,
        outside_support=True,
    )
    assert np.isnan(score)


# ---------------------------------------------------------------------------
# LOGO flight-time fit — no held-out leakage
# ---------------------------------------------------------------------------


def test_fold_flight_bundles_exclude_held_out_game():
    # Synthetic completed passes across three games.
    passes = pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(30)],
            "gameId": [10] * 10 + [20] * 10 + [30] * 10,
            "startFrame": [100] * 30,
            "endFrame": [110] * 30,
            "distance": np.linspace(5, 25, 30),
            "pass_outcome_class": ["complete"] * 30,
        }
    )
    eligibility = pd.DataFrame(
        {
            "passId": passes["id"],
            "gameId": passes["gameId"],
            "cohort": ["completed"] * 30,
            "true_receiverId": [101] * 30,
            "interceptorId": [pd.NA] * 30,
            "target_reliability_status": ["known_receiver"] * 30,
            "receiver_specific_eligible": [True] * 30,
            "usable_as_receiver_completion_label": [True] * 30,
            "usable_as_named_failed_target": [False] * 30,
            "named_failed_target_requires_sensitivity_check": [False] * 30,
            "usable_for_touch_level_turnover": [False] * 30,
            "narrow_path_active": [True] * 30,
            "target_inference_mode": ["narrow_path"] * 30,
            "human_audit_status": ["deferred_waived"] * 30,
            "label_gate_reason": ["known_receiver"] * 30,
            "pass_outcome_class": ["complete"] * 30,
            "passerId": [1] * 30,
            "decision": [pd.NA] * 30,
            "decision_reason": [pd.NA] * 30,
            "p_top": [np.nan] * 30,
            "prob_margin": [np.nan] * 30,
            "geom_margin_ft": [np.nan] * 30,
            "inferred_targetId": [pd.NA] * 30,
            "fold_id": ["f"] * 30,
        }
    )
    folds = [
        {"fold_id": "h10", "held_out_gameId": 10, "train_gameIds": [20, 30]},
        {"fold_id": "h20", "held_out_gameId": 20, "train_gameIds": [10, 30]},
        {"fold_id": "h30", "held_out_gameId": 30, "train_gameIds": [10, 20]},
    ]
    bundles = fit_fold_flight_bundles(passes, eligibility, folds)
    assert set(bundles) == {10, 20, 30}
    for gid, b in bundles.items():
        assert gid == b.held_out_gameId
        assert gid not in b.train_gameIds
        assert b.n_train_completed == 20  # other two games × 10
        # Monotonic in distance
        d = np.array([5.0, 15.0, 25.0])
        t = predict_flight_time(b.flight_model, d)
        assert np.all(np.diff(t) >= -1e-9)


def test_fit_distance_support_quantiles():
    support = fit_distance_support(np.linspace(0, 100, 101), low_quantile=0.01, high_quantile=0.99)
    assert support.distance_min_ft == pytest.approx(1.0, abs=0.5)
    assert support.distance_max_ft == pytest.approx(99.0, abs=0.5)


# ---------------------------------------------------------------------------
# Four candidates / frame + primary-eligible filter
# ---------------------------------------------------------------------------


def _synthetic_game_tables() -> dict[str, pd.DataFrame]:
    """Minimal Stage-1-like tables for one touch × two frames × four candidates."""
    frames = [100, 105]
    candidates = [11, 12, 13, 14]
    defenders = [21, 22, 23, 24, 25]
    rows = []
    for fi in frames:
        for rank, cid in enumerate(candidates):
            rows.append(
                {
                    "gameId": 99,
                    "touchId": "t1",
                    "chanceId": "c1",
                    "possessionId": "p1",
                    "frameIdx": fi,
                    "period": 1,
                    "gameClock": 500.0,
                    "shotClock": 14.0,
                    "ballhandlerId": 10,
                    "primary_frame_eligible": True,
                    "candidateId": cid,
                    "candidate_rank": rank,
                    "assigned_defenderId": defenders[rank],
                    "matchupId": f"m{cid}",
                    "matchup_matched": True,
                    "matchup_ambiguous": False,
                    "candidate_x_event": -20.0 + rank,
                    "candidate_y_event": float(rank * 3),
                    "candidate_vx": 1.0,
                    "candidate_vy": 0.0,
                    "candidate_velocity_ok": True,
                    "candidate_predError": 0.2,
                    "candidate_tracked": True,
                }
            )
    # One ineligible frame (stride residue / exclusion) — must be filtered out.
    for rank, cid in enumerate(candidates):
        rows.append(
            {
                "gameId": 99,
                "touchId": "t1",
                "chanceId": "c1",
                "possessionId": "p1",
                "frameIdx": 101,
                "period": 1,
                "gameClock": 499.96,
                "shotClock": 14.0,
                "ballhandlerId": 10,
                "primary_frame_eligible": False,
                "candidateId": cid,
                "candidate_rank": rank,
                "assigned_defenderId": defenders[rank],
                "matchupId": f"m{cid}",
                "matchup_matched": True,
                "matchup_ambiguous": False,
                "candidate_x_event": -20.0 + rank,
                "candidate_y_event": float(rank * 3),
                "candidate_vx": 1.0,
                "candidate_vy": 0.0,
                "candidate_velocity_ok": True,
                "candidate_predError": 0.2,
                "candidate_tracked": True,
            }
        )
    candidate_states = pd.DataFrame(rows)

    touch_model_frames = pd.DataFrame(
        [
            {
                "touchId": "t1",
                "frameIdx": fi,
                "ballhandler_x": 0.0,
                "ballhandler_y": 0.0,
                "shotClock": 14.0,
                "gameClock": 500.0,
            }
            for fi in frames + [101]
        ]
    )

    player_rows = []
    for fi in frames + [101]:
        player_rows.append(
            {
                "frameIdx": fi,
                "playerId": 10,
                "x_event": 0.0,
                "y_event": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "velocity_ok": True,
                "predError": 0.1,
                "isDetected": True,
            }
        )
        for rank, cid in enumerate(candidates):
            player_rows.append(
                {
                    "frameIdx": fi,
                    "playerId": cid,
                    "x_event": -20.0 + rank,
                    "y_event": float(rank * 3),
                    "vx": 1.0,
                    "vy": 0.0,
                    "velocity_ok": True,
                    "predError": 0.2,
                    "isDetected": True,
                }
            )
        for did in defenders:
            player_rows.append(
                {
                    "frameIdx": fi,
                    "playerId": did,
                    "x_event": -10.0,
                    "y_event": float(did - 21),
                    "vx": 0.0,
                    "vy": 0.0,
                    "velocity_ok": True,
                    "predError": 0.3,
                    "isDetected": True,
                }
            )
    tracking_players = pd.DataFrame(player_rows)
    chances = pd.DataFrame(
        {
            "id": ["c1"],
            "defPlayerIds": [defenders],
            "offPlayerIds": [[10] + candidates],
        }
    )
    touches = pd.DataFrame({"id": ["t1"], "startFrame": [100], "endFrame": [120]})
    return {
        "candidate_states": candidate_states,
        "touch_model_frames": touch_model_frames,
        "tracking_players": tracking_players,
        "chances": chances,
        "touches": touches,
    }


def test_four_candidates_per_frame_and_primary_eligible_filter():
    tables = _synthetic_game_tables()
    feats = reconstruct_game_candidates(
        **tables,
        flight_bundle=_bundle(held_out=99, train=(1, 2)),
        primary_eligible_only=True,
        force_pass_releases=False,
    )
    assert not feats.empty
    assert set(feats["frameIdx"].unique()) == {100, 105}
    assert 101 not in set(feats["frameIdx"])
    sizes = feats.groupby(["touchId", "frameIdx"]).size()
    assert (sizes == 4).all()
    assert int(feats["model_hz"].iloc[0]) == 5
    assert int(feats["frame_stride"].iloc[0]) == 5
    assert feats["is_model_5hz_frame"].astype(bool).all()


def test_forced_release_frame_off_5hz_grid():
    """Release at frame 102 (not stride-5) must still produce 4 candidate rows."""
    tables = _synthetic_game_tables()
    # Ensure tracking exists at off-grid release frame 102.
    extra = []
    for pid, x, y in [(10, 0.0, 0.0), (11, -20.0, 0.0), (12, -19.0, 3.0), (13, -18.0, 6.0), (14, -17.0, 9.0)] + [
        (d, -10.0, float(d - 21)) for d in [21, 22, 23, 24, 25]
    ]:
        extra.append(
            {
                "frameIdx": 102,
                "playerId": pid,
                "x_event": x,
                "y_event": y,
                "vx": 1.0 if pid < 20 else 0.0,
                "vy": 0.0,
                "velocity_ok": True,
                "predError": 0.2,
                "isDetected": True,
            }
        )
    tables["tracking_players"] = pd.concat(
        [tables["tracking_players"], pd.DataFrame(extra)], ignore_index=True
    )
    touch_candidates = pd.DataFrame(
        [
            {
                "gameId": 99,
                "touchId": "t1",
                "chanceId": "c1",
                "ballhandlerId": 10,
                "candidateId": cid,
                "candidate_rank": r,
            }
            for r, cid in enumerate([11, 12, 13, 14])
        ]
    )
    matchups = pd.DataFrame(
        columns=["gameId", "chanceId", "offPlayerId", "startFrame", "endFrame", "defPlayerId", "id"]
    )
    passes = pd.DataFrame(
        [
            {
                "id": "pass_offgrid",
                "gameId": 99,
                "touchId": "t1",
                "chanceId": "c1",
                "startFrame": 102,
                "period": 1,
                "startGameClock": 499.0,
                "shotClock": 14.0,
                "passerId": 10,
                "receiverId": 12,
                "toReceiverId": np.nan,
                "complete": True,
                "pass_outcome_class": "complete",
            }
        ]
    )
    tables["touches"] = pd.DataFrame(
        {
            "id": ["t1"],
            "gameId": [99],
            "chanceId": ["c1"],
            "startFrame": [100],
            "endFrame": [120],
            "playerId": [10],
            "primary_touch_eligible": [True],
        }
    )
    feats = reconstruct_game_candidates(
        **tables,
        flight_bundle=_bundle(held_out=99, train=(1, 2)),
        primary_eligible_only=True,
        passes=passes,
        touch_candidates=touch_candidates,
        matchups=matchups,
        force_pass_releases=True,
    )
    assert 102 in set(feats["frameIdx"])
    rel = feats[feats["frameIdx"] == 102]
    assert len(rel) == 4
    assert rel["is_forced_release_frame"].astype(bool).all()
    # 5 Hz frames still present
    assert {100, 105}.issubset(set(feats["frameIdx"]))


def test_stage4_placeholders_are_nan_not_fabricated():
    tables = _synthetic_game_tables()
    feats = reconstruct_game_candidates(
        **tables,
        flight_bundle=_bundle(held_out=99, train=(1, 2)),
        force_pass_releases=False,
    )
    assert feats["q_passability_model"].isna().all()
    assert feats["V_catch_model"].isna().all()
    assert feats["NOV_model"].isna().all()
    assert feats["geometry_passability_score"].notna().any()
    assert "help_density_12ft" in feats.columns
    assert "intercepting_defender_vx" in feats.columns
    assert "min_temporal_margin_const35_s" in feats.columns


def test_empty_train_fold_raises():
    from passing_windows.flight_time import fit_fold_flight_bundles

    passes = pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(5)],
            "gameId": [10] * 5,
            "startFrame": [100] * 5,
            "endFrame": [110] * 5,
            "distance": np.linspace(5, 15, 5),
            "pass_outcome_class": ["complete"] * 5,
        }
    )
    eligibility = pd.DataFrame(
        [
            {
                "passId": f"p{i}",
                "gameId": 10,
                "cohort": "completed",
                "true_receiverId": 101,
                "interceptorId": pd.NA,
                "target_reliability_status": "known_receiver",
                "receiver_specific_eligible": True,
                "usable_as_receiver_completion_label": True,
                "usable_as_named_failed_target": False,
                "named_failed_target_requires_sensitivity_check": False,
                "usable_for_touch_level_turnover": False,
                "narrow_path_active": True,
                "target_inference_mode": "narrow_path",
                "human_audit_status": "deferred_waived",
                "label_gate_reason": "known_receiver",
                "pass_outcome_class": "complete",
                "passerId": 1,
                "decision": pd.NA,
                "decision_reason": pd.NA,
                "p_top": np.nan,
                "prob_margin": np.nan,
                "geom_margin_ft": np.nan,
                "inferred_targetId": pd.NA,
                "fold_id": "f",
            }
            for i in range(5)
        ]
    )
    folds = [{"fold_id": "h10", "held_out_gameId": 10, "train_gameIds": [20, 30]}]
    with pytest.raises(ValueError, match="zero completed training"):
        fit_fold_flight_bundles(passes, eligibility, folds)


# ---------------------------------------------------------------------------
# Narrow-path label join + interceptor never target
# ---------------------------------------------------------------------------


def _gate_row(pass_id: str, *, cohort: str = "completed", true_rx=12, interceptor=None) -> dict:
    known = cohort == "completed"
    return {
        "passId": pass_id,
        "gameId": 99,
        "fold_id": "h99",
        "cohort": cohort,
        "pass_outcome_class": "complete" if known else "incomplete_turnover",
        "passerId": 10,
        "interceptorId": interceptor if interceptor is not None else pd.NA,
        "true_receiverId": true_rx if known else pd.NA,
        "inferred_targetId": pd.NA,
        "decision": pd.NA,
        "decision_reason": pd.NA,
        "p_top": np.nan,
        "prob_margin": np.nan,
        "geom_margin_ft": np.nan,
        "narrow_path_active": True,
        "target_inference_mode": "narrow_path",
        "target_reliability_status": "known_receiver" if known else "ambiguous_unreviewed",
        "human_audit_status": "deferred_waived",
        "usable_as_receiver_completion_label": known,
        "usable_as_named_failed_target": False,
        "named_failed_target_requires_sensitivity_check": False,
        "usable_for_touch_level_turnover": not known,
        "receiver_specific_eligible": known,
        "label_gate_reason": "known_receiver" if known else "ambiguous",
    }


def test_narrow_path_label_join_completion_only():
    tables = _synthetic_game_tables()
    feats = reconstruct_game_candidates(
        **tables,
        flight_bundle=_bundle(held_out=99, train=(1, 2)),
        force_pass_releases=False,
    )
    passes = pd.DataFrame(
        [
            {
                "id": "pass_ok",
                "gameId": 99,
                "touchId": "t1",
                "startFrame": 105,
                "passerId": 10,
                "receiverId": 12,
                "toReceiverId": np.nan,
                "pass_outcome_class": "complete",
                "complete": True,
            },
            {
                "id": "pass_fail",
                "gameId": 99,
                "touchId": "t1",
                "startFrame": 100,
                "passerId": 10,
                "receiverId": np.nan,
                "toReceiverId": 21.0,  # interceptor (defender)
                "pass_outcome_class": "incomplete_turnover",
                "complete": False,
            },
        ]
    )
    eligibility = pd.DataFrame(
        [
            _gate_row("pass_ok", cohort="completed", true_rx=12),
            _gate_row("pass_fail", cohort="failed", true_rx=None, interceptor=21),
        ]
    )
    out = attach_narrow_path_labels(feats, eligibility, passes)
    from passing_windows.narrow_path import GATE_COLUMNS

    for col in GATE_COLUMNS:
        assert col in out.columns
    # Frame 105 = completed release → completion eligible; frame 100 = failed → not.
    at_105 = out[out["frameIdx"] == 105]
    at_100 = out[out["frameIdx"] == 100]
    assert at_105["completion_label_eligible"].all()
    assert not at_100["completion_label_eligible"].any()
    assert int(at_105["is_true_receiver"].sum()) == 1
    assert int(at_105.loc[at_105["is_true_receiver"], "candidateId"].iloc[0]) == 12
    # Interceptor is a defender; never equals an offensive candidate.
    assert_interceptor_never_target(out, name="test", target_col="true_receiverId")
    clash = (
        out["candidateId"].notna()
        & out["interceptorId"].notna()
        & (out["candidateId"].astype("Int64") == out["interceptorId"].astype("Int64"))
    )
    assert int(clash.sum()) == 0
    labels = receiver_completion_training_labels(eligibility)
    assert len(labels) == 1
    assert labels.iloc[0]["passId"] == "pass_ok"


def test_interceptor_never_used_as_target_assertion():
    bad = pd.DataFrame(
        {
            "candidateId": [11, 12],
            "true_receiverId": [21, 12],
            "interceptorId": [21, 21],
            "inferred_targetId": [pd.NA, pd.NA],
        }
    )
    with pytest.raises(Exception):
        assert_interceptor_never_target(bad, name="bad", target_col="true_receiverId")


# ---------------------------------------------------------------------------
# Rebuild / cache markers (unit: marker contract; integration: on-disk)
# ---------------------------------------------------------------------------


def test_pipeline_marker_constant():
    # Import the Stage 3 pipeline module by path (not installed as a package).
    import importlib.util

    path = ROOT / "pipelines" / "03_build_candidate_states.py"
    spec = importlib.util.spec_from_file_location("stage3_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.MARKER_NAME == "_DONE_STAGE3"
    assert mod.FEATURES_NAME == "stage3_candidate_features.parquet"
    assert mod.TRAJECTORY_RULE == TRAJECTORY_RULE


def test_summarize_candidates_shape():
    tables = _synthetic_game_tables()
    feats = reconstruct_game_candidates(
        **tables,
        flight_bundle=_bundle(held_out=99, train=(1, 2)),
        force_pass_releases=False,
    )
    s = summarize_candidates(feats)
    assert s["n_candidate_rows"] == 8
    assert s["candidates_per_frame_mode"] == 4
    assert s["n_frames"] == 2


@needs_stage3
def test_real_release_label_coverage():
    elig = pd.read_parquet(ELIGIBILITY)
    labels = receiver_completion_training_labels(elig)
    df = pd.read_parquet(
        FEATURES, columns=["passId", "is_pass_release_frame", "is_forced_release_frame", "is_true_receiver"]
    )
    release_ids = set(
        df.loc[df["is_pass_release_frame"].fillna(False), "passId"].dropna().astype(str)
    )
    n_hit = len(set(labels["passId"].astype(str)) & release_ids)
    assert n_hit >= int(0.9 * len(labels)), f"release coverage {n_hit}/{len(labels)}"
    assert int(df["is_true_receiver"].sum()) >= int(0.9 * len(labels))


@needs_stage3
def test_real_full_gate_columns():
    df = pd.read_parquet(FEATURES)
    from passing_windows.narrow_path import GATE_COLUMNS

    for col in GATE_COLUMNS:
        assert col in df.columns, col


@needs_stage3
def test_real_dual_sampling_flags():
    df = pd.read_parquet(
        FEATURES, columns=["is_model_5hz_frame", "is_forced_release_frame", "sampling_role"]
    )
    assert df["is_model_5hz_frame"].astype(bool).any()
    assert df["is_forced_release_frame"].astype(bool).any()


@needs_stage3
def test_real_four_candidates_per_frame():
    df = pd.read_parquet(FEATURES, columns=["gameId", "touchId", "frameIdx", "candidateId"])
    sizes = df.groupby(["gameId", "touchId", "frameIdx"]).size()
    assert (sizes == 4).all()


@needs_stage3
def test_real_trajectory_rule_and_both_families():
    cols = [
        "trajectory_choice",
        "trajectory_choice_rule",
        "receiver_velocity_ok",
        "direct_geometry_passability_score",
        "lead_geometry_passability_score",
        "geometry_passability_score",
    ]
    df = pd.read_parquet(FEATURES, columns=cols)
    assert (df["trajectory_choice_rule"] == TRAJECTORY_RULE).all()
    lead = df["receiver_velocity_ok"].astype(bool)
    assert (df.loc[lead, "trajectory_choice"] == TRAJECTORY_LEAD).all()
    assert (df.loc[~lead, "trajectory_choice"] == TRAJECTORY_DIRECT).all()


@needs_stage3
def test_real_null_velocity_not_zeroed():
    df = pd.read_parquet(
        FEATURES, columns=["receiver_velocity_ok", "receiver_vx", "receiver_vy"]
    )
    missing = ~df["receiver_velocity_ok"].astype(bool)
    if missing.any():
        assert df.loc[missing, "receiver_vx"].isna().all()
        assert df.loc[missing, "receiver_vy"].isna().all()


@needs_stage3
def test_real_support_rejection_flag_present():
    df = pd.read_parquet(FEATURES, columns=["rejected_outside_support", "pass_distance_ft"])
    assert set(df["rejected_outside_support"].dropna().unique()).issubset({True, False})


@needs_stage3
@needs_eligibility
def test_real_narrow_path_label_join():
    elig = pd.read_parquet(ELIGIBILITY)
    for col in GATE_COLUMNS:
        assert col in elig.columns
    labels = receiver_completion_training_labels(elig)
    assert (labels["cohort"] == "completed").all()
    assert labels["receiver_specific_eligible"].all()
    failed_eligible = (
        (elig["cohort"] == "failed") & elig["receiver_specific_eligible"].fillna(False)
    ).sum()
    assert int(failed_eligible) == 0

    cols = [
        "completion_label_eligible",
        "is_true_receiver",
        "true_receiverId",
        "candidateId",
        "cohort",
        "interceptorId",
        "receiver_specific_eligible",
        "passId",
    ]
    df = pd.read_parquet(FEATURES, columns=[c for c in cols if True])
    # Completion labels never from failed cohort.
    if "cohort" in df.columns and df["completion_label_eligible"].any():
        bad = df["completion_label_eligible"] & (df["cohort"] == "failed")
        assert int(bad.fillna(False).sum()) == 0
    assert_interceptor_never_target(df, name="stage3_real", target_col="true_receiverId")


@needs_stage3
def test_real_primary_eligible_only_and_5hz():
    df = pd.read_parquet(
        FEATURES,
        columns=[
            "primary_frame_eligible",
            "model_hz",
            "frameIdx",
            "gameId",
            "touchId",
            "is_model_5hz_frame",
        ],
    )
    assert int(df["model_hz"].iloc[0]) == 5
    hz = df[df["is_model_5hz_frame"].astype(bool)] if "is_model_5hz_frame" in df.columns else df
    assert hz["primary_frame_eligible"].astype(bool).all()
    diffs = []
    for _, g in hz.groupby(["gameId", "touchId"]):
        frames = sorted(g["frameIdx"].unique())
        if len(frames) >= 2:
            diffs.extend(np.diff(frames).tolist())
    assert all(int(d) % 5 == 0 for d in diffs)


@needs_stage3
def test_real_rebuild_cache_markers_and_manifest():
    assert len(DONE_MARKERS) >= 1
    for marker in DONE_MARKERS:
        feat = marker.parent / "stage3_candidate_features.parquet"
        assert feat.exists()
    assert MANIFEST.exists()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert man.get("stage") == 3
    assert man.get("self_checks_ok") is True
    assert man.get("trajectory_choice_rule") == TRAJECTORY_RULE
    lock = load_narrow_path_lock()
    assert man.get("narrow_path_lock_id") == lock.get("lock_id")


@needs_stage3
def test_real_fold_flight_models_no_leakage():
    path = ARTIFACTS / "stage3_fold_flight_models.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    for fold in payload["folds"]:
        assert int(fold["held_out_gameId"]) not in set(fold["train_gameIds"])
        assert fold["n_train_completed"] > 0
