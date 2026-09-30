"""Tests for Stage 2 failed-pass target inference.

Unit tests are synthetic and always run. Tests suffixed `_real` read the tables
produced by `pipelines/02_infer_failed_pass_targets.py` and skip when the
pipeline has not been run, so a fresh clone still gets a green suite.
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

from passing_windows.narrow_path import (
    AUDIT_COMPLETE,
    AUDIT_DEFERRED_WAIVED,
    COHORT_COMPLETED,
    COHORT_FAILED,
    GATE_COLUMNS,
    MODE_NARROW_PATH,
    RELIABILITY_AMBIGUOUS,
    RELIABILITY_INFERRED_PROVISIONAL,
    RELIABILITY_KNOWN_RECEIVER,
    RELIABILITY_NO_EVIDENCE,
    NarrowPathViolation,
    apply_label_gates,
    assert_audit_sheet_on_disk_is_blank,
    assert_interceptor_never_target,
    assert_named_failed_targets_are_accepted,
    assert_narrow_path_gates,
    assert_no_failed_precision_claim,
    assert_no_unreliable_completion_labels,
    assert_reviewer_columns_blank,
    build_label_eligibility,
    evaluate_narrow_path,
    load_narrow_path_lock,
    receiver_completion_training_labels,
)
from passing_windows.paths import ARTIFACTS, TABLES
from passing_windows.target_audit import (
    REVIEWER_COLUMNS,
    build_audit_sheet,
    check_assignment_in_candidate_set,
    check_flight_time_monotonic,
    check_interceptor_never_target,
    check_mirror_invariance,
    check_probability_simplex,
    check_target_not_passer,
    check_unresolved_has_no_label,
    failed_invariants,
    synthetic_flight_time_model,
)
from passing_windows.target_inference import (
    DECISION_AMBIGUOUS,
    DECISION_AUTO_ACCEPT,
    DECISION_UNRESOLVED,
    FLIGHT_MODE_ONE_SIDED,
    FLIGHT_MODE_SYMMETRIC,
    MAX_PROJECTION_SPEED_FPS,
    AutoAcceptRule,
    BallFlight,
    EvidenceScales,
    apply_auto_accept,
    attach_probabilities,
    candidate_evidence,
    candidate_set_for_pass,
    classify_decisions,
    completed_pass_agreement,
    evidence_from_states,
    exclusion_sensitivity_stub,
    extract_ball_flight,
    fit_evidence_scales,
    fit_flight_time_model,
    flight_from_row,
    flight_to_columns,
    probability_calibration,
    project_candidate,
    select_auto_accept_rule,
    summarize_pass_level,
    unresolved_summary,
)

MODEL = synthetic_flight_time_model()
SCALES = EvidenceScales(
    sigma_ball_ft=1.5, sigma_pos_ft=0.5, flight_offset_s=-0.12, sigma_flight_s=0.07, n_train_passes=100
)

OFF_LINEUP = [11, 12, 13, 14, 15]
DEF_LINEUP = [21, 22, 23, 24, 25]


def straight_flight(*, dir_x: float = 1.0, dir_y: float = 0.0, chord: float = 15.0) -> BallFlight:
    return BallFlight(
        available=True,
        reason="ok",
        release_frame=100,
        release_x=0.0,
        release_y=0.0,
        dir_x=dir_x,
        dir_y=dir_y,
        speed_fps=30.0,
        clean_end_frame=112,
        clean_displacement_ft=chord,
        max_turn_deg=0.0,
        t_clean_s=chord / 30.0,
        t_obs_s=chord / 30.0,
        n_ball_frames=13,
        n_clean_frames=13,
    )


def make_states(candidates: list[dict], *, pass_id: str = "p1", failed: bool = True) -> pd.DataFrame:
    flight = straight_flight()
    rows = []
    for rank, cand in enumerate(candidates):
        rows.append(
            {
                "passId": pass_id,
                "gameId": 1,
                "passerId": 11,
                "interceptorId": 21.0 if failed else np.nan,
                "is_failed_pass": failed,
                "pass_outcome_class": "incomplete_turnover" if failed else "complete",
                "true_receiverId": np.nan if failed else float(candidates[0]["id"]),
                "n_candidates_declared": len(candidates),
                "candidateId": cand["id"],
                "candidate_rank": rank,
                "candidate_x_event": cand.get("x", np.nan),
                "candidate_y_event": cand.get("y", np.nan),
                "candidate_vx": cand.get("vx", 0.0),
                "candidate_vy": cand.get("vy", 0.0),
                "candidate_velocity_ok": cand.get("velocity_ok", True),
                "is_true_receiver": (not failed) and rank == 0,
                **flight_to_columns(flight),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Candidate sets: the interceptor can never be a target
# ---------------------------------------------------------------------------


def test_candidate_set_is_four_teammates_without_the_passer():
    side, cands = candidate_set_for_pass(OFF_LINEUP, DEF_LINEUP, 13)
    assert side == "off"
    assert cands == [11, 12, 14, 15]
    assert 13 not in cands


def test_interceptor_is_never_in_the_candidate_set():
    # The interceptor plays for the other team, so a candidate set drawn from
    # the passer's own lineup structurally excludes them.
    _, cands = candidate_set_for_pass(OFF_LINEUP, DEF_LINEUP, 13)
    for interceptor in DEF_LINEUP:
        assert interceptor not in cands


def test_candidate_set_uses_the_lineup_that_holds_the_passer():
    side, cands = candidate_set_for_pass(OFF_LINEUP, DEF_LINEUP, 22)
    assert side == "def"
    assert cands == [21, 23, 24, 25]


def test_candidate_set_empty_when_passer_in_neither_lineup():
    side, cands = candidate_set_for_pass(OFF_LINEUP, DEF_LINEUP, 99)
    assert side == "none"
    assert cands == []


def test_candidate_set_deduplicates_and_handles_missing():
    _, cands = candidate_set_for_pass([11, 11, 12, 13, 14, 15], None, 11)
    assert cands == [12, 13, 14, 15]
    assert candidate_set_for_pass(None, None, 11) == ("none", [])
    assert candidate_set_for_pass(OFF_LINEUP, DEF_LINEUP, np.nan) == ("none", [])


# ---------------------------------------------------------------------------
# Ball flight extraction
# ---------------------------------------------------------------------------


def test_ball_flight_straight_line_direction_and_speed():
    frames = np.arange(100, 110)
    flight = extract_ball_flight(frames, np.linspace(0, 18, 10), np.zeros(10), 100, 109)
    assert flight.available
    assert flight.dir_x == pytest.approx(1.0)
    assert flight.dir_y == pytest.approx(0.0)
    assert flight.speed_fps == pytest.approx(18.0 / (9 / 25.0))
    assert not flight.deflected


def test_ball_flight_truncates_at_a_deflection():
    # Straight for 8 ft, then the ball turns hard: only the clean part counts.
    frames = np.arange(100, 108)
    x = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 8.5, 9.0, 9.5])
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 20.0, 28.0])
    flight = extract_ball_flight(frames, x, y, 100, 107)
    assert flight.available
    assert flight.deflected
    assert flight.clean_end_frame == 104
    assert flight.dir_y == pytest.approx(0.0)
    assert flight.t_obs_s > flight.t_clean_s


def test_ball_flight_requires_minimum_frames_and_travel():
    two = extract_ball_flight(np.arange(100, 102), np.array([0.0, 5.0]), np.zeros(2), 100, 101)
    assert not two.available
    assert two.reason == "too_few_ball_frames"

    still = extract_ball_flight(np.arange(100, 105), np.zeros(5), np.zeros(5), 100, 104)
    assert not still.available
    assert still.reason == "ball_did_not_travel"


def test_flight_round_trips_through_columns():
    flight = straight_flight(dir_x=0.6, dir_y=0.8)
    restored = flight_from_row(pd.Series(flight_to_columns(flight)))
    assert restored.available
    assert restored.dir_x == pytest.approx(0.6)
    assert restored.dir_y == pytest.approx(0.8)
    assert restored.speed_fps == pytest.approx(flight.speed_fps)


# ---------------------------------------------------------------------------
# Flight-time support and projection
# ---------------------------------------------------------------------------


def test_flight_time_model_is_monotonic_and_reports_support():
    result = check_flight_time_monotonic(MODEL)
    assert result["ok"]
    assert MODEL.predict(30.0) > MODEL.predict(10.0)
    assert MODEL.distance_support_max > 30.0


def test_flight_time_model_survives_empty_training_data():
    empty = fit_flight_time_model([], [])
    assert np.isnan(float(empty.predict(10.0)))


def test_projection_treats_missing_velocity_as_unknown_not_zero():
    x, y, clipped = project_candidate(10.0, 4.0, np.nan, np.nan, 0.4, velocity_ok=False)
    assert (x, y) == (10.0, 4.0)
    assert not clipped

    moved_x, moved_y, _ = project_candidate(10.0, 4.0, 5.0, 0.0, 0.4, velocity_ok=True)
    assert moved_x == pytest.approx(12.0)
    assert moved_y == pytest.approx(4.0)


def test_projection_caps_speed_and_court_bounds():
    x, _, clipped = project_candidate(0.0, 0.0, 400.0, 0.0, 1.0, velocity_ok=True)
    assert clipped
    assert x <= MAX_PROJECTION_SPEED_FPS + 1e-9

    _, y, clipped_court = project_candidate(0.0, 24.0, 0.0, 40.0, 1.0, velocity_ok=True)
    assert clipped_court
    assert y <= 25.0


def test_candidate_directly_down_the_lane_has_zero_residuals():
    flight = straight_flight(chord=15.0)
    evidence = candidate_evidence(
        flight, {"x": 15.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "velocity_ok": True}, MODEL
    )
    assert evidence["angle_deg"] == pytest.approx(0.0)
    assert evidence["lateral_ft"] == pytest.approx(0.0)
    assert evidence["flight_residual_s"] == pytest.approx(0.0)
    assert not evidence["behind_release"]


def test_candidate_behind_the_passer_is_flagged():
    evidence = candidate_evidence(
        straight_flight(), {"x": -12.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "velocity_ok": True}, MODEL
    )
    assert evidence["behind_release"]
    assert not evidence["flight_channel_ok"]


def test_untracked_candidate_produces_no_evidence():
    evidence = candidate_evidence(
        straight_flight(), {"x": np.nan, "y": np.nan, "vx": 1.0, "vy": 1.0, "velocity_ok": True}, MODEL
    )
    assert not evidence["candidate_tracked"]
    assert not evidence["lateral_channel_ok"]
    assert np.isnan(evidence["lateral_ft"])


def test_vectorised_evidence_matches_scalar():
    rng = np.random.default_rng(20250924)
    candidates = []
    for i in range(24):
        candidates.append(
            {
                "id": 100 + i,
                "x": float(rng.uniform(-30, 40)),
                "y": float(rng.uniform(-24, 24)),
                "vx": float(rng.uniform(-30, 30)),
                "vy": float(rng.uniform(-30, 30)),
                "velocity_ok": bool(i % 4),
            }
        )
    candidates.append({"id": 900, "x": np.nan, "y": np.nan, "velocity_ok": False})
    states = make_states(candidates)
    vectorised = evidence_from_states(states, MODEL)

    flight = flight_from_row(states.iloc[0])
    scalar = pd.DataFrame(
        [
            candidate_evidence(
                flight,
                {
                    "x": row.candidate_x_event,
                    "y": row.candidate_y_event,
                    "vx": row.candidate_vx,
                    "vy": row.candidate_vy,
                    "velocity_ok": row.candidate_velocity_ok,
                },
                MODEL,
            )
            for row in states.itertuples(index=False)
        ]
    )
    for column in scalar.columns:
        left, right = scalar[column], vectorised[column]
        if left.dtype == bool or right.dtype == bool:
            assert (left.astype(bool) == right.astype(bool)).all(), column
        else:
            assert np.allclose(
                left.astype(float), right.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True
            ), column


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------


def test_probabilities_form_a_simplex():
    states = make_states(
        [
            {"id": 12, "x": 15.0, "y": 0.5},
            {"id": 13, "x": 14.0, "y": 7.0},
            {"id": 14, "x": 8.0, "y": -12.0},
            {"id": 15, "x": -10.0, "y": 2.0},
        ]
    )
    scored = attach_probabilities(evidence_from_states(states, MODEL), SCALES)
    assert scored["target_probability"].sum() == pytest.approx(1.0)
    assert (scored["target_probability"] >= 0).all()
    assert check_probability_simplex(scored)["ok"]


def test_aligned_candidate_takes_most_of_the_mass():
    states = make_states(
        [
            {"id": 12, "x": 15.0, "y": 0.2},
            {"id": 13, "x": 12.0, "y": 14.0},
            {"id": 14, "x": 6.0, "y": -18.0},
            {"id": 15, "x": -8.0, "y": 3.0},
        ]
    )
    scored = attach_probabilities(evidence_from_states(states, MODEL), SCALES)
    best = scored.loc[scored["target_probability"].idxmax()]
    assert best["candidateId"] == 12
    assert best["target_probability"] > 0.9


def test_all_untracked_candidates_fall_back_to_uniform():
    states = make_states([{"id": 12 + i, "x": np.nan, "y": np.nan} for i in range(4)])
    scored = attach_probabilities(evidence_from_states(states, MODEL), SCALES)
    assert scored["target_probability"].tolist() == pytest.approx([0.25] * 4)


def test_heavy_tailed_likelihood_does_not_saturate():
    # A Gaussian at these scales returns p_top = 1.0 to machine precision and
    # then cannot be wrong; the Student-t tail keeps the runner-up alive.
    states = make_states(
        [
            {"id": 12, "x": 15.0, "y": 0.5},
            {"id": 13, "x": 15.0, "y": 4.0},
            {"id": 14, "x": 20.0, "y": -14.0},
            {"id": 15, "x": -6.0, "y": 9.0},
        ]
    )
    scored = attach_probabilities(evidence_from_states(states, MODEL), SCALES)
    p_top = scored["target_probability"].max()
    assert 0.5 < p_top < 0.9999


def test_flight_mode_changes_only_the_undershoot_side():
    # A candidate beyond where the ball stopped is expected for an interception
    # and must score better under the failed-pass mode than the symmetric one.
    states = make_states([{"id": 12, "x": 30.0, "y": 0.0}, {"id": 13, "x": 15.0, "y": 6.0}])
    evidence = evidence_from_states(states, MODEL)
    one_sided = attach_probabilities(evidence, SCALES, flight_mode=FLIGHT_MODE_ONE_SIDED)
    symmetric = attach_probabilities(evidence, SCALES, flight_mode=FLIGHT_MODE_SYMMETRIC)
    far = one_sided["candidateId"] == 12
    assert one_sided.loc[far, "target_probability"].iloc[0] > symmetric.loc[far, "target_probability"].iloc[0]


def test_unknown_flight_mode_raises():
    states = make_states([{"id": 12, "x": 15.0, "y": 0.0}])
    with pytest.raises(ValueError):
        attach_probabilities(evidence_from_states(states, MODEL), SCALES, flight_mode="nonsense")


def test_scales_fit_from_known_receivers():
    rng = np.random.default_rng(7)
    rows = pd.DataFrame(
        {
            "angle_deg": np.abs(rng.normal(0, 6, 400)),
            "lateral_ft": np.abs(rng.normal(0, 2, 400)),
            "along_ft": rng.uniform(5, 25, 400),
            "bearing_chord_ft": rng.uniform(8, 20, 400),
            "flight_residual_s": rng.normal(-0.1, 0.05, 400),
        }
    )
    scales = fit_evidence_scales(rows)
    assert scales.sigma_ball_ft > 0
    assert scales.sigma_pos_ft > 0
    assert scales.flight_offset_s == pytest.approx(-0.1, abs=0.02)
    assert scales.n_train_passes == 400
    # A wider tolerance applies farther down the lane and on short ball chords.
    near = scales.lateral_sigma(np.array([5.0]), np.array([15.0]))
    far = scales.lateral_sigma(np.array([25.0]), np.array([15.0]))
    short_chord = scales.lateral_sigma(np.array([25.0]), np.array([4.0]))
    assert far > near
    assert short_chord > far


def test_mirroring_does_not_change_probabilities():
    assert check_mirror_invariance(MODEL, SCALES)["ok"]


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def summary_row(**overrides) -> dict:
    base = {
        "passId": "p1",
        "p_top": 0.95,
        "p_second": 0.03,
        "prob_margin": 0.92,
        "geom_margin_ft": 8.0,
        "top_candidateId": 12,
        "second_candidateId": 13,
        "n_candidates": 4,
        "n_candidates_declared": 4,
        "n_candidates_tracked": 4,
        "ball_flight_available": True,
    }
    base.update(overrides)
    return base


RULE = AutoAcceptRule(p_min=0.7, prob_margin=0.3, geom_margin_ft=4.0, fold_id="test")


def test_auto_accept_requires_every_margin():
    frame = pd.DataFrame(
        [
            summary_row(passId="ok"),
            summary_row(passId="low_p", p_top=0.65, prob_margin=0.6),
            summary_row(passId="thin_prob_margin", prob_margin=0.1),
            summary_row(passId="thin_geometry", geom_margin_ft=1.0),
            summary_row(passId="nan_geometry", geom_margin_ft=np.nan),
        ]
    )
    accepted = apply_auto_accept(frame, RULE)
    assert accepted.tolist() == [True, False, False, False, False]


def test_classify_decisions_separates_unresolved_from_ambiguous():
    frame = pd.DataFrame(
        [
            summary_row(passId="accept"),
            summary_row(passId="ambiguous", p_top=0.4, prob_margin=0.05, geom_margin_ft=0.5),
            summary_row(passId="no_flight", ball_flight_available=False),
            summary_row(passId="no_tracking", n_candidates_tracked=1),
            summary_row(passId="bad_candidate_set", n_candidates_declared=3),
        ]
    )
    out = classify_decisions(frame, RULE)
    decisions = dict(zip(out["passId"], out["decision"], strict=True))
    assert decisions["accept"] == DECISION_AUTO_ACCEPT
    assert decisions["ambiguous"] == DECISION_AMBIGUOUS
    assert decisions["no_flight"] == DECISION_UNRESOLVED
    assert decisions["no_tracking"] == DECISION_UNRESOLVED
    assert decisions["bad_candidate_set"] == DECISION_UNRESOLVED

    reasons = dict(zip(out["passId"], out["decision_reason"], strict=True))
    assert reasons["no_flight"] == "no_usable_ball_flight"
    assert reasons["bad_candidate_set"] == "candidate_set_not_four"


def test_only_accepted_passes_carry_a_hard_label():
    frame = pd.DataFrame(
        [summary_row(passId="accept"), summary_row(passId="ambiguous", p_top=0.3, prob_margin=0.02)]
    )
    out = classify_decisions(frame, RULE)
    labelled = out.set_index("passId")["inferred_targetId"]
    assert labelled["accept"] == 12
    assert pd.isna(labelled["ambiguous"])
    assert check_unresolved_has_no_label(out)["ok"]


def test_rule_selection_prefers_coverage_among_rules_that_hit_precision():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(400):
        confident = i % 2 == 0
        p_top = 0.97 if confident else 0.55
        correct = confident or rng.random() < 0.4
        rows.append(
            summary_row(
                passId=f"p{i}",
                p_top=p_top,
                prob_margin=0.9 if confident else 0.1,
                geom_margin_ft=9.0 if confident else 1.0,
                top_candidateId=12,
                true_receiverId=12 if correct else 13,
            )
        )
    train = pd.DataFrame(rows)
    rule = select_auto_accept_rule(train, fold_id="f1", target_precision=0.95)
    assert rule.met_target
    assert rule.train_precision >= 0.95
    assert rule.provisional
    assert rule.fold_id == "f1"
    # The confident half must survive the rule it selected.
    assert rule.train_coverage == pytest.approx(0.5, abs=0.05)


def test_rule_selection_falls_back_to_strictest_without_training_signal():
    rule = select_auto_accept_rule(pd.DataFrame(), fold_id="empty")
    assert rule.p_min == pytest.approx(0.9)
    assert rule.geom_margin_ft == pytest.approx(6.0)
    assert not rule.met_target


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def test_unresolved_summary_tracks_the_ten_percent_kill_line():
    frame = pd.DataFrame({"decision": [DECISION_AUTO_ACCEPT] * 85 + [DECISION_AMBIGUOUS] * 3 + [DECISION_UNRESOLVED]})
    result = unresolved_summary(frame, kill_threshold_pct=10.0)
    assert result["n_failed"] == 89
    assert result["max_unresolved_allowed"] == 8
    assert result["unresolved_strict_pct"] == pytest.approx(100 / 89)
    assert not result["strict_breaches_kill"]
    assert not result["pre_audit_breaches_kill"]

    worse = pd.DataFrame({"decision": [DECISION_AUTO_ACCEPT] * 50 + [DECISION_AMBIGUOUS] * 39})
    assert unresolved_summary(worse)["pre_audit_breaches_kill"]
    assert not unresolved_summary(worse)["strict_breaches_kill"]


def test_completed_pass_agreement_counts_only_accepted_for_precision():
    frame = pd.DataFrame(
        {
            "passId": ["a", "b", "c", "d"],
            "top_candidateId": [1, 2, 3, 4],
            "true_receiverId": [1, 2, 9, 9],
            "decision": [DECISION_AUTO_ACCEPT, DECISION_AUTO_ACCEPT, DECISION_AUTO_ACCEPT, DECISION_AMBIGUOUS],
        }
    )
    result = completed_pass_agreement(frame)
    assert result["top1_accuracy"] == pytest.approx(0.5)
    assert result["auto_accept_precision"] == pytest.approx(2 / 3)
    assert result["auto_accept_rate"] == pytest.approx(0.75)


def test_probability_calibration_bins_and_gap():
    frame = pd.DataFrame(
        {
            "p_top": [0.99, 0.98, 0.97, 0.5, 0.45],
            "top_candidateId": [1, 1, 1, 1, 1],
            "true_receiverId": [1, 1, 2, 2, 1],
        }
    )
    table = probability_calibration(frame)
    assert set(table.columns) >= {"p_top_bin", "n", "mean_p_top", "empirical_accuracy", "calibration_gap"}
    assert int(table["n"].sum()) == 5


def test_exclusion_sensitivity_stub_counts_inferred_attempts():
    failed = pd.DataFrame(
        {
            "passId": ["f1", "f2"],
            "decision": [DECISION_AUTO_ACCEPT, DECISION_AMBIGUOUS],
            "top_candidateId": [12, 13],
        }
    )
    completed = pd.DataFrame({"passId": ["c1", "c2", "c3"], "true_receiverId": [12, 12, 14]})
    table = exclusion_sensitivity_stub(failed, completed).set_index("receiverId")
    assert table.loc[12, "n_inferred_failed"] == 1  # the ambiguous one is not counted
    assert table.loc[12, "n_completed"] == 2
    assert table.loc[12, "completion_rate_including_inferred"] == pytest.approx(2 / 3)
    assert table.loc[14, "n_inferred_failed"] == 0


# ---------------------------------------------------------------------------
# Audit export
# ---------------------------------------------------------------------------


def audit_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored_rows = []
    for pass_id, outcome in (("f1", "incomplete_turnover"), ("f2", "incomplete_turnover"), ("c1", "complete")):
        for rank, cid in enumerate([12, 13, 14, 15]):
            scored_rows.append(
                {
                    "passId": pass_id,
                    "gameId": 1,
                    "pass_outcome_class": outcome,
                    "passerId": 11,
                    "candidateId": cid,
                    "candidate_rank": rank,
                    "angle_deg": 5.0 * rank,
                    "lateral_ft": 1.0 * rank,
                    "target_probability": [0.7, 0.2, 0.07, 0.03][rank],
                }
            )
    scored = pd.DataFrame(scored_rows)
    failed = pd.DataFrame(
        {
            "passId": ["f1", "f2"],
            "gameId": [1, 1],
            "decision": [DECISION_AUTO_ACCEPT, DECISION_AMBIGUOUS],
            "decision_reason": ["margin_satisfied", "margin_not_satisfied"],
            "p_top": [0.95, 0.4],
            "prob_margin": [0.9, 0.05],
            "geom_margin_ft": [7.0, 0.5],
            "top_candidateId": [12, 12],
        }
    )
    completed = pd.DataFrame(
        {
            "passId": ["c1"],
            "gameId": [1],
            "decision": [DECISION_AUTO_ACCEPT],
            "decision_reason": ["margin_satisfied"],
            "p_top": [0.99],
            "prob_margin": [0.95],
            "geom_margin_ft": [9.0],
            "top_candidateId": [12],
        }
    )
    return scored, failed, completed


def test_audit_sheet_exports_every_failed_pass_with_blank_reviewer_columns():
    scored, failed, completed = audit_fixture()
    sheet = build_audit_sheet(scored, failed, completed, seed=20250924)
    assert set(sheet["passId"]) == {"f1", "f2", "c1"}
    assert set(sheet[sheet["passId"] == "f2"]["audit_stratum"]) == {"failed_ambiguous"}
    assert set(sheet[sheet["passId"] == "c1"]["audit_stratum"]) == {"completed_control"}
    # Four candidate rows per pass, so the reviewer sees the alternatives.
    assert len(sheet) == 12
    for column in ("reviewer_is_intended_target", "reviewer_decision", "reviewer_confidence", "reviewer_reason"):
        assert (sheet[column] == "").all()
    assert sheet["is_model_top_candidate"].sum() == 3


def test_audit_sheet_sampling_is_deterministic():
    scored, failed, completed = audit_fixture()
    first = build_audit_sheet(scored, failed, completed, seed=20250924)
    second = build_audit_sheet(scored, failed, completed, seed=20250924)
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Self-consistency checks catch what they claim to catch
# ---------------------------------------------------------------------------


def test_simplex_check_detects_broken_probabilities():
    broken = pd.DataFrame({"passId": ["p", "p"], "target_probability": [0.7, 0.7]})
    assert not check_probability_simplex(broken)["ok"]


def test_interceptor_check_detects_a_poisoned_assignment():
    scored = pd.DataFrame({"passId": ["p"], "candidateId": [21], "interceptorId": [21.0], "passerId": [11]})
    assignments = pd.DataFrame({"passId": ["p"], "inferred_targetId": [21.0], "interceptorId": [21.0]})
    result = check_interceptor_never_target(scored, assignments)
    assert not result["ok"]
    assert result["n_interceptor_in_candidate_set"] == 1
    assert result["n_interceptor_assigned_as_target"] == 1


def test_passer_check_detects_self_target():
    scored = pd.DataFrame({"passId": ["p"], "candidateId": [11], "passerId": [11]})
    assert not check_target_not_passer(scored)["ok"]


def test_assignment_outside_candidate_set_is_caught():
    scored = pd.DataFrame({"passId": ["p"] * 4, "candidateId": [12, 13, 14, 15]})
    assignments = pd.DataFrame({"passId": ["p"], "inferred_targetId": [99.0], "decision": [DECISION_AUTO_ACCEPT]})
    assert not check_assignment_in_candidate_set(scored, assignments)["ok"]


# ---------------------------------------------------------------------------
# Narrow-path lock: the gate that keeps unaudited targets out of Stage 3
# ---------------------------------------------------------------------------

LOCK = load_narrow_path_lock()
BREACHING_UNRESOLVED = {"unresolved_strict_pct": 2.2, "unresolved_pre_audit_pct": 32.6}
CLEAN_UNRESOLVED = {"unresolved_strict_pct": 1.0, "unresolved_pre_audit_pct": 4.0}


def narrow_state(unresolved: dict | None = None, **kwargs) -> dict:
    return evaluate_narrow_path(unresolved or BREACHING_UNRESOLVED, lock=LOCK, **kwargs)


def gated_failed(state: dict) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "passId": ["accept", "ambiguous", "unresolved"],
            "gameId": [1, 1, 1],
            "decision": [DECISION_AUTO_ACCEPT, DECISION_AMBIGUOUS, DECISION_UNRESOLVED],
            "inferred_targetId": [12.0, None, None],
            "true_receiverId": [np.nan, np.nan, np.nan],
            "interceptorId": [21.0, 21.0, np.nan],
            "top_candidateId": [12, 13, 14],
        }
    )
    return apply_label_gates(frame, cohort=COHORT_FAILED, state=state)


def gated_completed(state: dict) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "passId": ["c1", "c2"],
            "gameId": [1, 1],
            "decision": [DECISION_AUTO_ACCEPT, DECISION_AMBIGUOUS],
            "inferred_targetId": [12.0, None],
            "true_receiverId": [12.0, 13.0],
            "truth_in_candidate_set": [True, True],
            "interceptorId": [np.nan, np.nan],
            "top_candidateId": [12, 14],
        }
    )
    return apply_label_gates(frame, cohort=COHORT_COMPLETED, state=state)


def test_lock_freezes_narrow_path_mode_and_a_deferred_audit():
    assert LOCK["target_inference_mode"] == MODE_NARROW_PATH
    assert LOCK["human_audit"]["status"] == AUDIT_DEFERRED_WAIVED
    assert LOCK["human_audit"]["fabricated_reviewer_values_forbidden"] is True
    assert LOCK["completed_pass_proxy"]["role"] == "diagnostic_only"
    assert set(LOCK["gate_columns"]["required"]) == set(GATE_COLUMNS)


def test_pre_audit_breach_and_missing_audit_both_trigger_the_narrow_path():
    state = narrow_state()
    assert state["narrow_path_active"]
    assert "pre_audit_unresolved_rate_exceeds_frozen_threshold" in state["triggers"]
    assert "human_audit_status_is_not_complete" in state["triggers"]


def test_narrow_path_stays_active_even_if_the_unresolved_rate_clears():
    # Clearing the numeric kill line does not retire the gate on its own: the
    # audit that the frozen protocol demanded still does not exist.
    state = narrow_state(CLEAN_UNRESOLVED)
    assert state["narrow_path_active"]
    assert state["triggers"] == ["human_audit_status_is_not_complete"]


def test_gate_only_relaxes_when_a_real_audit_is_recorded_and_rates_clear():
    state = narrow_state(CLEAN_UNRESOLVED, human_audit_status=AUDIT_COMPLETE)
    # The lock itself still pins narrow mode, so relaxation needs a lock revision
    # as well as an audit — exactly one place to change, and it is reviewed.
    assert state["triggers"] == []
    assert state["narrow_path_active"]
    assert state["locked_mode"] == MODE_NARROW_PATH


def test_no_failed_pass_becomes_a_receiver_completion_label():
    failed = gated_failed(narrow_state())
    assert not failed["usable_as_receiver_completion_label"].any()
    assert not failed["receiver_specific_eligible"].any()
    # Nothing is discarded: every failed pass stays available at the touch level.
    assert failed["usable_for_touch_level_turnover"].all()


def test_reliability_status_tracks_the_decision():
    failed = gated_failed(narrow_state()).set_index("passId")
    assert failed.loc["accept", "target_reliability_status"] == RELIABILITY_INFERRED_PROVISIONAL
    assert failed.loc["ambiguous", "target_reliability_status"] == RELIABILITY_AMBIGUOUS
    assert failed.loc["unresolved", "target_reliability_status"] == RELIABILITY_NO_EVIDENCE

    completed = gated_completed(narrow_state())
    assert (completed["target_reliability_status"] == RELIABILITY_KNOWN_RECEIVER).all()


def test_auto_accepted_failures_are_opt_in_and_require_a_sensitivity_check():
    failed = gated_failed(narrow_state()).set_index("passId")
    assert failed.loc["accept", "usable_as_named_failed_target"]
    assert failed.loc["accept", "named_failed_target_requires_sensitivity_check"]
    # Opt-in only: the default Stage 3 filter still excludes them.
    assert not failed.loc["accept", "receiver_specific_eligible"]
    assert not failed.loc["ambiguous", "usable_as_named_failed_target"]


def test_completed_passes_with_known_receivers_are_the_only_eligible_labels():
    state = narrow_state()
    eligibility = build_label_eligibility(gated_failed(state), gated_completed(state))
    labels = receiver_completion_training_labels(eligibility)
    assert set(labels["passId"]) == {"c1", "c2"}
    assert (labels["cohort"] == COHORT_COMPLETED).all()
    assert (labels["target_reliability_status"] == RELIABILITY_KNOWN_RECEIVER).all()


def test_gate_columns_are_written_to_every_pass_table():
    state = narrow_state()
    for frame in (gated_failed(state), gated_completed(state)):
        assert set(GATE_COLUMNS) <= set(frame.columns)
    eligibility = build_label_eligibility(gated_failed(state), gated_completed(state))
    assert set(GATE_COLUMNS) <= set(eligibility.columns)
    assert eligibility["narrow_path_active"].all()


def test_full_gate_suite_passes_on_a_clean_run():
    state = narrow_state()
    failed, completed = gated_failed(state), gated_completed(state)
    summary = assert_narrow_path_gates(
        failed, completed, build_label_eligibility(failed, completed), state=state
    )
    assert summary["n_receiver_specific_eligible"] == 2
    assert summary["n_named_failed_targets_optional"] == 1
    assert summary["n_touch_level_only"] == 3


# --- the gate must actually bite ------------------------------------------


def test_promoting_an_ambiguous_pass_to_a_completion_label_is_fatal():
    state = narrow_state()
    failed = gated_failed(state)
    failed.loc[failed["passId"] == "ambiguous", "usable_as_receiver_completion_label"] = True
    with pytest.raises(NarrowPathViolation, match="unreliable target_reliability_status"):
        assert_no_unreliable_completion_labels(failed, name="failed")


def test_promoting_an_unresolved_pass_to_a_completion_label_is_fatal():
    state = narrow_state()
    failed, completed = gated_failed(state), gated_completed(state)
    failed.loc[failed["passId"] == "unresolved", "usable_as_receiver_completion_label"] = True
    eligibility = build_label_eligibility(failed, completed)
    with pytest.raises(NarrowPathViolation):
        assert_narrow_path_gates(failed, completed, eligibility, state=state)


def test_an_auto_accepted_failure_smuggled_in_as_a_completion_label_is_fatal():
    # The subtlest failure mode: a provisional label that looks good enough.
    state = narrow_state()
    failed = gated_failed(state)
    failed.loc[failed["passId"] == "accept", "usable_as_receiver_completion_label"] = True
    with pytest.raises(NarrowPathViolation):
        assert_no_unreliable_completion_labels(failed, name="failed")


def test_naming_a_failed_target_without_acceptance_is_fatal():
    state = narrow_state()
    failed = gated_failed(state)
    failed.loc[failed["passId"] == "ambiguous", "usable_as_named_failed_target"] = True
    with pytest.raises(NarrowPathViolation, match="not auto-accepted"):
        assert_named_failed_targets_are_accepted(failed, name="failed")


def test_missing_gate_columns_are_fatal():
    state = narrow_state()
    failed, completed = gated_failed(state), gated_completed(state)
    stripped = failed.drop(columns=["receiver_specific_eligible"])
    with pytest.raises(NarrowPathViolation, match="missing narrow-path gate columns"):
        assert_narrow_path_gates(
            stripped, completed, build_label_eligibility(stripped, completed), state=state
        )


def test_interceptor_as_inferred_target_is_fatal_not_merely_reported():
    state = narrow_state()
    failed = gated_failed(state)
    failed.loc[failed["passId"] == "accept", "inferred_targetId"] = 21.0
    with pytest.raises(NarrowPathViolation, match="interceptor"):
        assert_interceptor_never_target(failed, name="failed")


def test_fabricated_reviewer_values_are_fatal_while_the_audit_is_deferred():
    sheet = pd.DataFrame(
        {
            "passId": ["f1", "f2"],
            "reviewer_decision": ["", ""],
            "reviewer_confidence": ["", ""],
        }
    )
    assert_reviewer_columns_blank(
        sheet, ("reviewer_decision", "reviewer_confidence"), human_audit_status=AUDIT_DEFERRED_WAIVED
    )
    sheet.loc[0, "reviewer_decision"] = "agree"
    with pytest.raises(NarrowPathViolation, match="reviewer fields must stay"):
        assert_reviewer_columns_blank(
            sheet,
            ("reviewer_decision", "reviewer_confidence"),
            human_audit_status=AUDIT_DEFERRED_WAIVED,
        )


def test_a_hand_edited_audit_sheet_blocks_the_next_run(tmp_path):
    # Checking only the freshly built sheet would miss the file someone edited
    # between runs, which is the case that actually matters.
    path = tmp_path / "stage2_ambiguous_audit.csv"
    pd.DataFrame({"passId": ["f1"], "reviewer_decision": [""]}).to_csv(path, index=False)
    assert_audit_sheet_on_disk_is_blank(
        path, ("reviewer_decision",), human_audit_status=AUDIT_DEFERRED_WAIVED
    )
    pd.DataFrame({"passId": ["f1"], "reviewer_decision": ["target_is_12"]}).to_csv(path, index=False)
    with pytest.raises(NarrowPathViolation):
        assert_audit_sheet_on_disk_is_blank(
            path, ("reviewer_decision",), human_audit_status=AUDIT_DEFERRED_WAIVED
        )
    # Once a real review exists, the sheet is expected to be full.
    assert_audit_sheet_on_disk_is_blank(
        path, ("reviewer_decision",), human_audit_status=AUDIT_COMPLETE
    )


def test_reports_may_not_sell_the_completed_pass_proxy_as_failed_pass_precision():
    assert_no_failed_precision_claim(
        "Held-out completed-pass auto-accept precision (diagnostic) | 0.968", name="report"
    )
    with pytest.raises(NarrowPathViolation):
        assert_no_failed_precision_claim(
            "Stage 2 reaches an audited failed-pass precision of 0.968.", name="report"
        )


# ---------------------------------------------------------------------------
# Real pipeline outputs
# ---------------------------------------------------------------------------

PASSES = TABLES / "passes.parquet"
ASSIGNMENTS = TABLES / "stage2_failed_pass_assignments.parquet"
PROBABILITIES = TABLES / "stage2_target_probabilities.parquet"
VALIDATION = TABLES / "stage2_completed_pass_validation.parquet"
AUDIT_CSV = ARTIFACTS / "stage2_ambiguous_audit.csv"

needs_stage1 = pytest.mark.skipif(not PASSES.exists(), reason="Stage 1 tables not built")
needs_stage2 = pytest.mark.skipif(not ASSIGNMENTS.exists(), reason="Stage 2 pipeline not run")


@needs_stage1
@needs_stage2
def test_failed_cohort_is_the_tri_state_false_class_real():
    passes = pd.read_parquet(PASSES, columns=["id", "pass_outcome_class", "is_failed_pass"])
    assignments = pd.read_parquet(ASSIGNMENTS)

    failed_ids = set(passes.loc[passes["pass_outcome_class"] == "incomplete_turnover", "id"])
    unknown_ids = set(passes.loc[passes["pass_outcome_class"] == "unknown_outcome", "id"])
    scored_ids = set(assignments["passId"])

    assert len(failed_ids) == 89
    assert scored_ids == failed_ids
    # `complete` is tri-state: the null class is a different population and must
    # never be scored as a failed pass.
    assert not (scored_ids & unknown_ids)


@needs_stage2
def test_real_outputs_never_name_the_interceptor_as_target():
    assignments = pd.read_parquet(ASSIGNMENTS)
    scored = pd.read_parquet(PROBABILITIES)
    result = check_interceptor_never_target(scored, assignments)
    assert result["ok"], result
    assert result["n_interceptor_in_candidate_set"] == 0
    assert result["n_interceptor_assigned_as_target"] == 0


@needs_stage2
def test_real_probabilities_are_a_simplex_and_labels_are_clean():
    scored = pd.read_parquet(PROBABILITIES)
    assignments = pd.read_parquet(ASSIGNMENTS)
    assert check_probability_simplex(scored)["ok"]
    assert check_target_not_passer(scored)["ok"]
    assert check_assignment_in_candidate_set(scored, assignments)["ok"]
    assert check_unresolved_has_no_label(assignments)["ok"]
    assert not failed_invariants(
        {
            "probability_simplex": {**check_probability_simplex(scored), "severity": "invariant"},
            "target_not_passer": {**check_target_not_passer(scored), "severity": "invariant"},
        }
    )


@needs_stage2
def test_real_completed_pass_agreement_is_far_above_chance():
    validation = pd.read_parquet(VALIDATION)
    agreement = completed_pass_agreement(validation)
    # Four candidates, so chance is 0.25.
    assert agreement["top1_accuracy"] > 0.70
    assert agreement["auto_accept_precision"] > 0.90
    assert agreement["n"] > 4000


@needs_stage2
def test_real_unresolved_rate_is_reported_against_the_kill_line():
    assignments = pd.read_parquet(ASSIGNMENTS)
    result = unresolved_summary(assignments, kill_threshold_pct=10.0)
    assert result["n_failed"] == 89
    assert result["max_unresolved_allowed"] == 8
    assert (
        result["n_auto_accepted"] + result["n_ambiguous"] + result["n_unresolved_strict"]
        == result["n_failed"]
    )


@needs_stage2
def test_real_audit_export_contains_every_ambiguous_failure():
    assignments = pd.read_parquet(ASSIGNMENTS)
    sheet = pd.read_csv(AUDIT_CSV)
    ambiguous = set(assignments.loc[assignments["decision"] == DECISION_AMBIGUOUS, "passId"])
    assert ambiguous <= set(sheet["passId"])
    assert set(assignments["passId"]) <= set(sheet["passId"])
    for column in ("reviewer_is_intended_target", "reviewer_decision", "reviewer_confidence", "reviewer_reason"):
        assert sheet[column].isna().all() or (sheet[column].astype(str) == "").all()
    assert (sheet.groupby("passId").size() <= 5).all()


@needs_stage2
def test_real_fold_assignment_is_leave_one_game_out():
    scored = pd.read_parquet(PROBABILITIES)
    # A game's predictions may only come from the fold that held that game out.
    per_game = scored.groupby("gameId")["fold_id"].nunique()
    assert (per_game == 1).all()
    for game_id, fold_id in scored.groupby("gameId")["fold_id"].first().items():
        assert str(game_id) in fold_id


# ---------------------------------------------------------------------------
# Real outputs: the narrow-path gate is on disk, not just in the report prose
# ---------------------------------------------------------------------------

ELIGIBILITY = TABLES / "stage2_label_eligibility.parquet"
NARROW_STATE = ARTIFACTS / "stage2_narrow_path_state.json"

needs_gate = pytest.mark.skipif(not ELIGIBILITY.exists(), reason="Stage 2 pipeline not run")


@needs_gate
def test_real_tables_all_carry_the_gate_columns():
    for path in (ELIGIBILITY, ASSIGNMENTS, VALIDATION, PROBABILITIES):
        frame = pd.read_parquet(path)
        missing = [c for c in GATE_COLUMNS if c not in frame.columns]
        assert not missing, f"{path.name} missing {missing}"
        assert frame["narrow_path_active"].all()
        assert (frame["target_inference_mode"] == MODE_NARROW_PATH).all()


@needs_gate
def test_real_narrow_path_is_active_and_the_audit_is_recorded_as_deferred():
    state = json.loads(NARROW_STATE.read_text(encoding="utf-8"))
    assert state["narrow_path_active"]
    assert state["target_inference_mode"] == MODE_NARROW_PATH
    assert state["human_audit_status"] == AUDIT_DEFERRED_WAIVED
    assert "pre_audit_unresolved_rate_exceeds_frozen_threshold" in state["triggers"]
    assert "human_audit_status_is_not_complete" in state["triggers"]
    assert state["completed_pass_proxy_role"] == "diagnostic_only"


@needs_gate
def test_real_completion_labels_are_completed_passes_with_known_receivers_only():
    eligibility = pd.read_parquet(ELIGIBILITY)
    labels = receiver_completion_training_labels(eligibility)
    assert len(labels) > 4000
    assert (labels["cohort"] == COHORT_COMPLETED).all()
    assert (labels["target_reliability_status"] == RELIABILITY_KNOWN_RECEIVER).all()
    assert labels["true_receiverId"].notna().all()
    # No failed pass reaches the default Stage 3 training set.
    failed = eligibility[eligibility["cohort"] == COHORT_FAILED]
    assert len(failed) == 89
    assert not failed["receiver_specific_eligible"].any()
    assert not failed["usable_as_receiver_completion_label"].any()


@needs_gate
def test_real_failed_passes_stay_available_for_touch_level_turnover():
    eligibility = pd.read_parquet(ELIGIBILITY)
    failed = eligibility[eligibility["cohort"] == COHORT_FAILED]
    assert failed["usable_for_touch_level_turnover"].all()
    named = failed[failed["usable_as_named_failed_target"].astype(bool)]
    assert (named["decision"] == DECISION_AUTO_ACCEPT).all()
    assert named["inferred_targetId"].notna().all()
    assert named["named_failed_target_requires_sensitivity_check"].all()


@needs_gate
def test_real_gate_suite_holds_end_to_end():
    state = json.loads(NARROW_STATE.read_text(encoding="utf-8"))
    assert_narrow_path_gates(
        pd.read_parquet(ASSIGNMENTS),
        pd.read_parquet(VALIDATION),
        pd.read_parquet(ELIGIBILITY),
        state=state,
    )


@needs_stage2
def test_real_audit_sheet_has_no_fabricated_reviewer_decisions():
    sheet = pd.read_csv(AUDIT_CSV)
    assert_reviewer_columns_blank(
        sheet, REVIEWER_COLUMNS, human_audit_status=AUDIT_DEFERRED_WAIVED
    )


@needs_gate
def test_real_reports_do_not_sell_the_proxy_as_failed_pass_precision():
    for name in ("stage2_target_inference_report.md", "STAGE2_VERIFICATION.md"):
        assert_no_failed_precision_claim(
            (ARTIFACTS / name).read_text(encoding="utf-8"), name=name
        )
