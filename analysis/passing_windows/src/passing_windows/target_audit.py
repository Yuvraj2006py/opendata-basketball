"""Automated self-consistency checks for Stage 2 target inference.

These replace nothing that a human reviewer would do — they only prove that the
inference machinery is internally coherent (probabilities are a simplex, the
interceptor never becomes a target, geometry is mirror-invariant) and report the
held-out agreement that the provisional decision thresholds rest on.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .target_inference import (
    DECISION_AMBIGUOUS,
    DECISION_AUTO_ACCEPT,
    DECISION_UNRESOLVED,
    BallFlight,
    EvidenceScales,
    FlightTimeModel,
    attach_probabilities,
    candidate_evidence,
    fit_flight_time_model,
)

SIMPLEX_TOLERANCE = 1e-9

AUDIT_COMPLETED_PER_GAME = 6
REVIEWER_COLUMNS = (
    "reviewer_is_intended_target",
    "reviewer_decision",
    "reviewer_confidence",
    "reviewer_reason",
)
AUDIT_SHEET_COLUMNS = (
    "passId",
    "gameId",
    "period",
    "startFrame",
    "endFrame",
    "startGameClock",
    "shotClock",
    "pass_outcome_class",
    "passerId",
    "interceptorId",
    "true_receiverId",
    "candidate_side",
    "passer_x_event",
    "passer_y_event",
    "ball_release_x_event",
    "ball_release_y_event",
    "ball_dir_x",
    "ball_dir_y",
    "ball_speed_fps",
    "ball_t_clean_s",
    "ball_t_obs_s",
    "ball_deflected",
    "n_ball_frames",
    "candidateId",
    "candidate_rank",
    "candidate_x_event",
    "candidate_y_event",
    "candidate_vx",
    "candidate_vy",
    "candidate_velocity_ok",
    "candidate_predError",
    "proj_x_event",
    "proj_y_event",
    "t_pred_flight_s",
    "angle_deg",
    "along_ft",
    "lateral_ft",
    "flight_residual_s",
    "behind_release",
    "outside_distance_support",
    "target_probability",
    "fold_id",
)
DECISION_SHEET_COLUMNS = (
    "passId",
    "decision",
    "decision_reason",
    "p_top",
    "prob_margin",
    "geom_margin_ft",
    "top_candidateId",
)


def build_audit_sheet(
    scored: pd.DataFrame,
    failed: pd.DataFrame,
    completed: pd.DataFrame,
    *,
    seed: int,
    completed_per_game: int = AUDIT_COMPLETED_PER_GAME,
) -> pd.DataFrame:
    """Every failed pass plus a seeded completed-pass control, one row per candidate.

    Ambiguous failures are the cases a human must settle, but a sheet holding
    only those cannot measure reviewer accuracy, so auto-accepted failures and a
    blind sample of completed passes (where the answer is known but not shown in
    the reviewer's workflow) travel with them. Reviewer columns are left empty
    by design: Stage 2 produces the sheet, a human fills it, Stage 4 reads the
    agreement back and re-selects the margin against it.
    """
    rng = np.random.default_rng(seed)
    strata: list[pd.DataFrame] = []
    if not failed.empty:
        strata.append(
            pd.DataFrame(
                {
                    "passId": failed["passId"].to_numpy(),
                    "audit_stratum": "failed_" + failed["decision"].astype(str),
                }
            )
        )
    if not completed.empty and completed_per_game > 0:
        picks: list[Any] = []
        for _, group in completed.sort_values("passId").groupby("gameId"):
            ids = group["passId"].to_numpy()
            picks.extend(rng.choice(ids, size=min(completed_per_game, len(ids)), replace=False).tolist())
        chosen = completed[completed["passId"].isin(picks)]
        strata.append(
            pd.DataFrame({"passId": chosen["passId"].to_numpy(), "audit_stratum": "completed_control"})
        )
    if not strata:
        return pd.DataFrame()

    selected = pd.concat(strata, ignore_index=True)
    sheet = scored[[c for c in AUDIT_SHEET_COLUMNS if c in scored.columns]].merge(
        selected, on="passId", how="inner"
    )

    decision_frames = [
        frame[[c for c in DECISION_SHEET_COLUMNS if c in frame.columns]]
        for frame in (failed, completed)
        if not frame.empty
    ]
    if decision_frames:
        sheet = sheet.merge(pd.concat(decision_frames, ignore_index=True), on="passId", how="left")
    if "top_candidateId" in sheet.columns:
        sheet["is_model_top_candidate"] = sheet["candidateId"] == sheet["top_candidateId"]
    for column in REVIEWER_COLUMNS:
        sheet[column] = ""
    sort_cols = [c for c in ("audit_stratum", "passId", "target_probability") if c in sheet.columns]
    return sheet.sort_values(
        sort_cols, ascending=[c != "target_probability" for c in sort_cols]
    ).reset_index(drop=True)


def check_probability_simplex(
    scored: pd.DataFrame,
    *,
    pass_key: str = "passId",
    prob_col: str = "target_probability",
) -> dict[str, Any]:
    """Every pass's candidate probabilities are non-negative and sum to one."""
    if scored.empty:
        return {"ok": True, "n_passes": 0, "max_abs_sum_error": 0.0, "n_negative": 0}
    sums = scored.groupby(pass_key)[prob_col].sum()
    max_err = float(np.max(np.abs(sums.to_numpy(dtype=float) - 1.0))) if len(sums) else 0.0
    n_neg = int((scored[prob_col] < 0).sum())
    n_nan = int(scored[prob_col].isna().sum())
    return {
        "ok": bool(max_err <= 1e-6 and n_neg == 0 and n_nan == 0),
        "n_passes": int(len(sums)),
        "max_abs_sum_error": max_err,
        "n_negative": n_neg,
        "n_nan": n_nan,
    }


def check_interceptor_never_target(
    scored: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, Any]:
    """The frozen rule: `toReceiverId` is the interceptor and never a target."""
    in_candidates = 0
    if not scored.empty and {"candidateId", "interceptorId"}.issubset(scored.columns):
        both = scored["interceptorId"].notna()
        in_candidates = int(
            (scored.loc[both, "candidateId"] == scored.loc[both, "interceptorId"]).sum()
        )
    assigned = 0
    if not assignments.empty and {"inferred_targetId", "interceptorId"}.issubset(assignments.columns):
        both = assignments["inferred_targetId"].notna() & assignments["interceptorId"].notna()
        assigned = int(
            (
                assignments.loc[both, "inferred_targetId"].astype("float64")
                == assignments.loc[both, "interceptorId"].astype("float64")
            ).sum()
        )
    return {
        "ok": bool(in_candidates == 0 and assigned == 0),
        "n_interceptor_in_candidate_set": in_candidates,
        "n_interceptor_assigned_as_target": assigned,
    }


def check_target_not_passer(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty or not {"candidateId", "passerId"}.issubset(scored.columns):
        return {"ok": True, "n_passer_as_candidate": 0}
    n = int((scored["candidateId"] == scored["passerId"]).sum())
    return {"ok": n == 0, "n_passer_as_candidate": n}


def check_assignment_in_candidate_set(
    scored: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    pass_key: str = "passId",
) -> dict[str, Any]:
    """Accepted labels name a member of that pass's own candidate set."""
    if assignments.empty:
        return {"ok": True, "n_checked": 0, "n_outside_candidate_set": 0}
    sets = scored.groupby(pass_key)["candidateId"].apply(set)
    accepted = assignments[assignments["inferred_targetId"].notna()]
    bad = 0
    for row in accepted.itertuples(index=False):
        members = sets.get(getattr(row, pass_key), set())
        if float(row.inferred_targetId) not in {float(m) for m in members}:
            bad += 1
    return {"ok": bad == 0, "n_checked": int(len(accepted)), "n_outside_candidate_set": bad}


def check_unresolved_has_no_label(assignments: pd.DataFrame) -> dict[str, Any]:
    if assignments.empty:
        return {"ok": True, "n_labelled_non_accepted": 0}
    non_accepted = assignments["decision"] != DECISION_AUTO_ACCEPT
    n = int(assignments.loc[non_accepted, "inferred_targetId"].notna().sum())
    return {"ok": n == 0, "n_labelled_non_accepted": n}


def check_mirror_invariance(flight_time_model: FlightTimeModel, scales: EvidenceScales) -> dict[str, Any]:
    """Mirroring the court in y must not change any candidate probability.

    Stage 1 requires features to mirror and probabilities not to. The same must
    hold for target inference, otherwise a left-side pass would be scored
    differently from its right-side twin.
    """
    flight = BallFlight(
        available=True,
        reason="ok",
        release_frame=0,
        release_x=0.0,
        release_y=0.0,
        dir_x=0.9,
        dir_y=np.sqrt(1.0 - 0.81),
        speed_fps=30.0,
        clean_end_frame=10,
        clean_displacement_ft=12.0,
        max_turn_deg=1.0,
        t_clean_s=0.4,
        t_obs_s=0.4,
        n_ball_frames=10,
        n_clean_frames=10,
    )
    mirrored_flight = replace(flight, dir_y=-flight.dir_y)
    candidates = [
        {"x": 18.0, "y": 3.0, "vx": 1.0, "vy": 2.0, "velocity_ok": True},
        {"x": 12.0, "y": -9.0, "vx": -2.0, "vy": 0.5, "velocity_ok": True},
        {"x": 25.0, "y": 11.0, "vx": 0.0, "vy": 0.0, "velocity_ok": True},
        {"x": -6.0, "y": 2.0, "vx": 0.0, "vy": -1.0, "velocity_ok": True},
    ]

    def _score(fl: BallFlight, cands: list[dict[str, Any]]) -> np.ndarray:
        rows = [candidate_evidence(fl, c, flight_time_model) for c in cands]
        frame = pd.DataFrame(rows)
        frame["passId"] = "synthetic"
        frame["candidateId"] = np.arange(len(cands))
        scored = attach_probabilities(frame, scales)
        return scored["target_probability"].to_numpy(dtype=float)

    raw = _score(flight, candidates)
    mirrored = _score(
        mirrored_flight,
        [{**c, "y": -c["y"], "vy": -c["vy"]} for c in candidates],
    )
    max_diff = float(np.max(np.abs(raw - mirrored))) if raw.size else 0.0
    return {
        "ok": bool(max_diff <= 1e-9),
        "max_abs_probability_difference": max_diff,
        "probabilities": [round(float(p), 6) for p in raw],
    }


def check_flight_time_monotonic(flight_time_model: FlightTimeModel) -> dict[str, Any]:
    grid = np.arange(2.0, 45.0, 1.0)
    pred = flight_time_model.predict(grid)
    diffs = np.diff(pred)
    return {
        "ok": bool(np.all(diffs >= -1e-9)) and bool(np.all(pred > 0)),
        "n_bins": int(flight_time_model.centers.size),
        "n_train": int(flight_time_model.n_train),
        "distance_support_max_ft": float(flight_time_model.distance_support_max),
        "t_at_5ft": float(flight_time_model.predict(5.0)),
        "t_at_15ft": float(flight_time_model.predict(15.0)),
        "t_at_30ft": float(flight_time_model.predict(30.0)),
        "min_diff": float(np.min(diffs)) if diffs.size else 0.0,
    }


def check_release_point_matches_passer(states: pd.DataFrame) -> dict[str, Any]:
    """The tracked ball at release should sit on the passer, not elsewhere."""
    cols = {"ball_release_x_event", "ball_release_y_event", "passer_x_event", "passer_y_event"}
    if states.empty or not cols.issubset(states.columns):
        return {"ok": True, "n": 0}
    per_pass = states.drop_duplicates(subset=["passId"])
    d = np.hypot(
        per_pass["ball_release_x_event"] - per_pass["passer_x_event"],
        per_pass["ball_release_y_event"] - per_pass["passer_y_event"],
    ).to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"ok": True, "n": 0}
    median = float(np.median(d))
    p95 = float(np.quantile(d, 0.95))
    return {
        # A ball on the passer's hands is a few feet from their tracked centre;
        # a systematic offset would mean the release anchor is wrong.
        "ok": bool(median <= 6.0),
        "n": int(d.size),
        "median_ft": median,
        "p95_ft": p95,
        "share_within_6ft": float(np.mean(d <= 6.0)),
    }


def check_failed_residuals_within_completed_support(
    failed_assignments: pd.DataFrame,
    completed_validation: pd.DataFrame,
) -> dict[str, Any]:
    """Accepted failed-pass geometry should look like known-receiver geometry."""
    if failed_assignments.empty or completed_validation.empty:
        return {"ok": True, "n_accepted": 0}
    accepted = failed_assignments[failed_assignments["decision"] == DECISION_AUTO_ACCEPT]
    correct = completed_validation[
        completed_validation["top_candidateId"] == completed_validation["true_receiverId"]
    ]
    if accepted.empty or correct.empty:
        return {"ok": True, "n_accepted": int(len(accepted))}
    out: dict[str, Any] = {"n_accepted": int(len(accepted)), "n_reference": int(len(correct))}
    ok = True
    for col in ("top_angle_deg", "top_lateral_ft"):
        ref_p99 = float(np.nanquantile(correct[col].to_numpy(dtype=float), 0.99))
        share = float(np.nanmean(accepted[col].to_numpy(dtype=float) > ref_p99))
        out[f"{col}_reference_p99"] = ref_p99
        out[f"{col}_share_beyond_reference_p99"] = share
        ok = ok and share <= 0.10
    out["ok"] = ok
    return out


def decision_mix(summary: pd.DataFrame) -> dict[str, int]:
    counts = summary["decision"].value_counts() if not summary.empty else pd.Series(dtype=int)
    return {
        DECISION_AUTO_ACCEPT: int(counts.get(DECISION_AUTO_ACCEPT, 0)),
        DECISION_AMBIGUOUS: int(counts.get(DECISION_AMBIGUOUS, 0)),
        DECISION_UNRESOLVED: int(counts.get(DECISION_UNRESOLVED, 0)),
    }


# Invariants must hold for the output to be valid at all. Diagnostics describe
# the data — a failed pass genuinely has harder geometry than a completed one —
# so they are reported rather than treated as build failures.
INVARIANT_CHECKS = (
    "probability_simplex",
    "interceptor_never_target",
    "target_not_passer",
    "assignment_in_candidate_set",
    "unresolved_has_no_label",
    "mirror_invariance",
    "flight_time_monotonic",
)


def run_self_consistency_suite(
    scored: pd.DataFrame,
    failed_assignments: pd.DataFrame,
    completed_validation: pd.DataFrame,
    states: pd.DataFrame,
    flight_time_model: FlightTimeModel,
    scales: EvidenceScales,
) -> dict[str, dict[str, Any]]:
    """All automated Stage 2 geometry checks in one dict, keyed by check name."""
    checks = {
        "probability_simplex": check_probability_simplex(scored),
        "interceptor_never_target": check_interceptor_never_target(scored, failed_assignments),
        "target_not_passer": check_target_not_passer(scored),
        "assignment_in_candidate_set": check_assignment_in_candidate_set(scored, failed_assignments),
        "unresolved_has_no_label": check_unresolved_has_no_label(failed_assignments),
        "mirror_invariance": check_mirror_invariance(flight_time_model, scales),
        "flight_time_monotonic": check_flight_time_monotonic(flight_time_model),
        "release_point_matches_passer": check_release_point_matches_passer(states),
        "failed_residuals_within_completed_support": check_failed_residuals_within_completed_support(
            failed_assignments, completed_validation
        ),
    }
    for name, result in checks.items():
        result["severity"] = "invariant" if name in INVARIANT_CHECKS else "diagnostic"
    return checks


def failed_invariants(checks: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, res in checks.items() if res.get("severity") == "invariant" and not res.get("ok")]


def failed_diagnostics(checks: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, res in checks.items() if res.get("severity") == "diagnostic" and not res.get("ok")]


def synthetic_flight_time_model() -> FlightTimeModel:
    """Deterministic stand-in model for unit tests and offline checks."""
    rng = np.random.default_rng(20250924)
    distance = rng.uniform(2.0, 40.0, size=2000)
    time_s = 0.08 + distance / 40.0
    return fit_flight_time_model(distance, time_s)
