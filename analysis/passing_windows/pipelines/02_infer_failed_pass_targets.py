#!/usr/bin/env python
"""Stage 2: infer intended targets for failed passes with explicit uncertainty.

Restartable: the expensive step is reading retained tracking per game to build
release states and ball flights, and that is cached under
`tables/by_game/{gameId}/stage2_pass_states.parquet` behind a `_DONE_STAGE2`
marker. Fold fitting, scoring, and reporting are cheap and always rerun.

    python pipelines/02_infer_failed_pass_targets.py            # build + report
    python pipelines/02_infer_failed_pass_targets.py --force    # rebuild caches
    python pipelines/02_infer_failed_pass_targets.py --games 114243
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

# Allow running without install: add src to path
STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from passing_windows.ingest import game_ids_from_freeze, load_freeze  # noqa: E402
from passing_windows.io_utils import (  # noqa: E402
    build_output_manifest,
    verify_output_manifest,
    write_json,
    write_markdown,
    write_table,
)
from passing_windows.narrow_path import (  # noqa: E402
    COHORT_COMPLETED,
    COHORT_FAILED,
    GATE_COLUMNS,
    NarrowPathViolation,
    apply_label_gates,
    assert_audit_sheet_on_disk_is_blank,
    assert_narrow_path_gates,
    assert_no_failed_precision_claim,
    assert_reviewer_columns_blank,
    build_label_eligibility,
    evaluate_narrow_path,
    load_narrow_path_lock,
)
from passing_windows.paths import ARTIFACTS, PACKAGE_ROOT, TABLES  # noqa: E402
from passing_windows.target_audit import (  # noqa: E402
    AUDIT_COMPLETED_PER_GAME,
    REVIEWER_COLUMNS,
    build_audit_sheet,
    decision_mix,
    failed_diagnostics,
    failed_invariants,
    run_self_consistency_suite,
)
from passing_windows.target_inference import (  # noqa: E402
    AUTO_ACCEPT_TARGET_PRECISION,
    DECISION_AMBIGUOUS,
    DECISION_AUTO_ACCEPT,
    DECISION_UNRESOLVED,
    FLIGHT_MODE_ONE_SIDED,
    FLIGHT_MODE_SYMMETRIC,
    FREEZE_ID,
    AutoAcceptRule,
    attach_probabilities,
    candidate_set_for_pass,
    classify_decisions,
    completed_pass_agreement,
    evidence_from_states,
    exclusion_sensitivity_stub,
    extract_ball_flight,
    fit_evidence_scales,
    fit_flight_time_model,
    flight_to_columns,
    probability_calibration,
    select_auto_accept_rule,
    summarize_pass_level,
    unresolved_summary,
)
from passing_windows.velocities import FRAME_RATE_HZ  # noqa: E402

STAGE = 2
MARKER_NAME = "_DONE_STAGE2"
STATES_NAME = "stage2_pass_states.parquet"
UNRESOLVED_KILL_PCT = 10.0


def _game_dir(game_id: int) -> Path:
    return TABLES / "by_game" / str(game_id)


# ---------------------------------------------------------------------------
# Per-game release states (cached)
# ---------------------------------------------------------------------------


def _select_pass_cohort(passes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split passes into the Stage 2 cohorts, keeping the tri-state honest.

    `complete` is tri-state: the 68 `unknown_outcome` passes have no end frame,
    no receiver, and no interceptor. They are never target-inference inputs and
    are only counted.
    """
    failed = passes[passes["failed_pass_inference_eligible"].astype(bool)].copy()
    completed = passes[
        (passes["pass_outcome_class"] == "complete")
        & passes["receiverId"].notna()
        & passes["endFrame"].notna()
        & ~passes["is_self_pass"].astype(bool)
    ].copy()
    unknown = passes[passes["pass_outcome_class"] == "unknown_outcome"].copy()
    return {"failed": failed, "completed": completed, "unknown": unknown}


def build_game_states(game_id: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Release-time candidate states and ball flights for one game's passes."""
    gdir = _game_dir(game_id)
    passes = pd.read_parquet(gdir / "passes.parquet")
    chances = pd.read_parquet(
        gdir / "chances.parquet", columns=["id", "gameId", "offPlayerIds", "defPlayerIds", "transition", "usable_flag"]
    ).rename(columns={"id": "chanceId", "transition": "chance_transition", "usable_flag": "chance_usable"})
    touches = pd.read_parquet(
        gdir / "touches.parquet", columns=["id", "gameId", "primary_touch_eligible", "exclusion_reason"]
    ).rename(columns={"id": "touchId", "exclusion_reason": "touch_exclusion_reason"})

    cohort = _select_pass_cohort(passes)
    target = pd.concat([cohort["failed"], cohort["completed"]], ignore_index=True)
    target = target.merge(chances, on=["gameId", "chanceId"], how="left").merge(
        touches, on=["gameId", "touchId"], how="left"
    )

    frame_cols = ["frameIdx", "ball_x_event", "ball_y_event", "ball_z", "ball_isDetected", "ball_predError"]
    tframes = pd.read_parquet(gdir / "tracking_frames.parquet", columns=frame_cols).sort_values("frameIdx")
    tplayers = pd.read_parquet(
        gdir / "tracking_players.parquet",
        columns=["frameIdx", "playerId", "x_event", "y_event", "vx", "vy", "velocity_ok", "predError", "isDetected"],
    )

    release_frames = set(int(f) for f in target["startFrame"].dropna().tolist())
    release = tplayers[tplayers["frameIdx"].isin(release_frames)]
    release_lookup: dict[tuple[int, int], dict[str, Any]] = {
        (int(r.frameIdx), int(r.playerId)): {
            "x": float(r.x_event) if pd.notna(r.x_event) else np.nan,
            "y": float(r.y_event) if pd.notna(r.y_event) else np.nan,
            "vx": float(r.vx) if pd.notna(r.vx) else np.nan,
            "vy": float(r.vy) if pd.notna(r.vy) else np.nan,
            "velocity_ok": bool(r.velocity_ok) if pd.notna(r.velocity_ok) else False,
            "predError": float(r.predError) if pd.notna(r.predError) else np.nan,
            "isDetected": r.isDetected,
        }
        for r in release.itertuples(index=False)
    }

    frame_idx = tframes["frameIdx"].to_numpy()
    ball_x = tframes["ball_x_event"].to_numpy(dtype=float)
    ball_y = tframes["ball_y_event"].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for prow in target.itertuples(index=False):
        f0 = int(prow.startFrame)
        f1 = int(prow.endFrame) if pd.notna(prow.endFrame) else f0
        side, candidates = candidate_set_for_pass(prow.offPlayerIds, prow.defPlayerIds, prow.passerId)
        is_failed = bool(prow.is_failed_pass)
        true_receiver = int(prow.receiverId) if pd.notna(prow.receiverId) else None
        truth_in_set = true_receiver in candidates if true_receiver is not None else False

        if not is_failed and (len(candidates) != 4 or not truth_in_set):
            # A completed pass can only validate the method when the known
            # receiver is actually one of the four candidates.
            skipped.append(
                {
                    "gameId": game_id,
                    "passId": prow.id,
                    "pass_outcome_class": prow.pass_outcome_class,
                    "reason": "candidate_set_not_four" if len(candidates) != 4 else "receiver_outside_candidate_set",
                    "n_candidates": len(candidates),
                }
            )
            continue

        lo = np.searchsorted(frame_idx, f0, side="left")
        hi = np.searchsorted(frame_idx, f1, side="right")
        flight = extract_ball_flight(frame_idx[lo:hi], ball_x[lo:hi], ball_y[lo:hi], f0, f1)
        flight_cols = flight_to_columns(flight)

        passer_state = release_lookup.get((f0, int(prow.passerId)), {})
        base = {
            "passId": prow.id,
            "gameId": int(prow.gameId),
            "period": prow.period,
            "chanceId": prow.chanceId,
            "possessionId": prow.possessionId,
            "touchId": prow.touchId,
            "startFrame": f0,
            "endFrame": f1,
            "startGameClock": prow.startGameClock,
            "shotClock": prow.shotClock,
            "passerId": int(prow.passerId),
            "interceptorId": float(prow.interceptorId) if pd.notna(prow.interceptorId) else np.nan,
            "pass_outcome_class": prow.pass_outcome_class,
            "is_failed_pass": is_failed,
            "turnover": bool(prow.turnover),
            "inbounds": bool(prow.inbounds),
            "backcourt": bool(prow.backcourt),
            "chance_transition": prow.chance_transition,
            "chance_usable": prow.chance_usable,
            "primary_touch_eligible": prow.primary_touch_eligible,
            "touch_exclusion_reason": prow.touch_exclusion_reason,
            "true_receiverId": float(true_receiver) if true_receiver is not None else np.nan,
            "truth_in_candidate_set": truth_in_set,
            "candidate_side": side,
            "n_candidates_declared": len(candidates),
            "observed_flight_time_s": float((f1 - f0) / FRAME_RATE_HZ),
            "passer_x_event": passer_state.get("x", np.nan),
            "passer_y_event": passer_state.get("y", np.nan),
            "passer_tracked": bool(np.isfinite(passer_state.get("x", np.nan))),
            **flight_cols,
        }

        if not candidates:
            # Keep the pass in the denominator with an explicit empty candidate
            # row rather than dropping it from the unresolved accounting.
            rows.append(
                {
                    **base,
                    "candidateId": -1,
                    "candidate_rank": 0,
                    "candidate_x_event": np.nan,
                    "candidate_y_event": np.nan,
                    "candidate_vx": np.nan,
                    "candidate_vy": np.nan,
                    "candidate_velocity_ok": False,
                    "candidate_predError": np.nan,
                    "candidate_isDetected": np.nan,
                    "candidate_tracked": False,
                    "state_release_distance_ft": np.nan,
                    "is_true_receiver": False,
                }
            )
            continue

        for rank, cid in enumerate(candidates):
            state = release_lookup.get((f0, int(cid)), {})
            cx, cy = state.get("x", np.nan), state.get("y", np.nan)
            d_release = (
                float(np.hypot(cx - flight.release_x, cy - flight.release_y))
                if flight.available and np.isfinite(cx) and np.isfinite(cy)
                else np.nan
            )
            rows.append(
                {
                    **base,
                    "candidateId": int(cid),
                    "candidate_rank": rank,
                    "candidate_x_event": cx,
                    "candidate_y_event": cy,
                    "candidate_vx": state.get("vx", np.nan),
                    "candidate_vy": state.get("vy", np.nan),
                    "candidate_velocity_ok": bool(state.get("velocity_ok", False)),
                    "candidate_predError": state.get("predError", np.nan),
                    "candidate_isDetected": state.get("isDetected", np.nan),
                    "candidate_tracked": bool(np.isfinite(cx) and np.isfinite(cy)),
                    "state_release_distance_ft": d_release,
                    "is_true_receiver": true_receiver is not None and int(cid) == true_receiver,
                }
            )

    states = pd.DataFrame(rows)
    summary = {
        "gameId": game_id,
        "n_passes_total": int(len(passes)),
        "n_failed": int(len(cohort["failed"])),
        "n_completed_usable": int(len(cohort["completed"])),
        "n_unknown_outcome": int(len(cohort["unknown"])),
        "n_states": int(len(states)),
        "n_skipped_completed": len(skipped),
        "skipped": skipped,
        "n_failed_with_ball_flight": int(
            states[states["is_failed_pass"]].drop_duplicates("passId")["ball_flight_available"].sum()
        )
        if len(states)
        else 0,
    }
    return states, summary


def load_states(game_ids: list[int], force: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for gid in game_ids:
        gdir = _game_dir(gid)
        marker = gdir / MARKER_NAME
        path = gdir / STATES_NAME
        if marker.exists() and path.exists() and not force:
            print(f"[{gid}] stage2 states cached")
            parts.append(pd.read_parquet(path))
            summaries.append({"gameId": gid, "cached": True})
            continue
        print(f"[{gid}] building stage2 release states...")
        states, summary = build_game_states(gid)
        write_table(states, path)
        write_json(summary, gdir / "stage2_game_summary.json")
        marker.write_text("ok\n", encoding="utf-8")
        parts.append(states)
        summaries.append(summary)
    if not parts:
        return pd.DataFrame(), summaries
    return pd.concat(parts, ignore_index=True), summaries


# ---------------------------------------------------------------------------
# Leave-one-game-out fitting and scoring
# ---------------------------------------------------------------------------


def run_folds(states: pd.DataFrame, freeze: dict[str, Any]) -> dict[str, Any]:
    """Fit flight time, likelihood scales, and the accept rule per outer fold."""
    folds = freeze["validation"]["outer_folds"]
    scored_parts: list[pd.DataFrame] = []
    failed_parts: list[pd.DataFrame] = []
    completed_parts: list[pd.DataFrame] = []
    train_diag_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    last_model = None
    last_scales = None

    for fold in folds:
        fold_id = fold["fold_id"]
        held_out = int(fold["held_out_gameId"])
        train_ids = [int(g) for g in fold["train_gameIds"]]

        train_states = states[states["gameId"].isin(train_ids)]
        test_states = states[states["gameId"] == held_out]
        if test_states.empty:
            continue

        # 1. Empirical flight-time support from training completed passes only.
        truth_rows = train_states[train_states["is_true_receiver"]]
        model = fit_flight_time_model(
            truth_rows["state_release_distance_ft"], truth_rows["observed_flight_time_s"]
        )

        # 2. Residual scales from the same known receivers, scored under that model.
        train_completed = evidence_from_states(
            train_states[~train_states["is_failed_pass"]], model
        )
        scales = fit_evidence_scales(train_completed[train_completed["is_true_receiver"]])

        # 3. Decision thresholds: completed passes stand in for the audit labels
        #    that Stage 4 will provide, and only training games are used.
        train_scored = attach_probabilities(train_completed, scales, flight_mode=FLIGHT_MODE_ONE_SIDED)
        train_summary = summarize_pass_level(train_scored).merge(
            train_scored.drop_duplicates("passId")[
                ["passId", "gameId", "true_receiverId", "n_candidates_declared", "ball_flight_available"]
            ],
            on="passId",
            how="left",
        )
        rule = select_auto_accept_rule(train_summary, fold_id=fold_id)

        # 4. Score the held-out game only.
        test_evidence = evidence_from_states(test_states, model)
        test_failed = attach_probabilities(
            test_evidence[test_evidence["is_failed_pass"]], scales, flight_mode=FLIGHT_MODE_ONE_SIDED
        )
        test_completed = attach_probabilities(
            test_evidence[~test_evidence["is_failed_pass"]], scales, flight_mode=FLIGHT_MODE_ONE_SIDED
        )
        test_completed_symmetric = attach_probabilities(
            test_evidence[~test_evidence["is_failed_pass"]],
            scales,
            flight_mode=FLIGHT_MODE_SYMMETRIC,
            prob_col="target_probability_symmetric",
        )
        test_completed["target_probability_symmetric"] = test_completed_symmetric[
            "target_probability_symmetric"
        ]

        for frame, mode in ((test_failed, FLIGHT_MODE_ONE_SIDED), (test_completed, FLIGHT_MODE_ONE_SIDED)):
            frame["fold_id"] = fold_id
            frame["flight_mode"] = mode
        scored_parts.extend([test_failed, test_completed])

        failed_summary = _summarize(test_failed, rule)
        completed_summary = _summarize(test_completed, rule)
        failed_parts.append(failed_summary)
        completed_parts.append(completed_summary)
        train_diag_parts.append(train_summary.assign(fold_id=fold_id))

        agreement = completed_pass_agreement(completed_summary)
        fold_records.append(
            {
                **rule.as_dict(),
                "held_out_gameId": held_out,
                "n_train_games": len(train_ids),
                "flight_time_bins": int(model.centers.size),
                "flight_time_n_train": int(model.n_train),
                "distance_support_max_ft": float(model.distance_support_max),
                "scale_sigma_ball_ft": scales.sigma_ball_ft,
                "scale_sigma_pos_ft": scales.sigma_pos_ft,
                "scale_flight_offset_s": scales.flight_offset_s,
                "scale_sigma_flight_s": scales.sigma_flight_s,
                "scale_nu": scales.nu,
                "scale_undershoot": scales.undershoot_scale,
                "scale_n_train_passes": scales.n_train_passes,
                "n_failed_heldout": int(len(failed_summary)),
                "n_completed_heldout": int(len(completed_summary)),
                "heldout_completed_top1_accuracy": agreement["top1_accuracy"],
                "heldout_completed_auto_accept_rate": agreement["auto_accept_rate"],
                "heldout_completed_auto_accept_precision": agreement["auto_accept_precision"],
                **{f"failed_{k}": v for k, v in decision_mix(failed_summary).items()},
            }
        )
        last_model, last_scales = model, scales
        print(
            f"[{fold_id}] failed={len(failed_summary)} "
            f"accept={decision_mix(failed_summary)[DECISION_AUTO_ACCEPT]} "
            f"completed_top1={agreement['top1_accuracy']:.3f}"
        )

    return {
        "scored": pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame(),
        "failed": pd.concat(failed_parts, ignore_index=True) if failed_parts else pd.DataFrame(),
        "completed": pd.concat(completed_parts, ignore_index=True) if completed_parts else pd.DataFrame(),
        "train_diagnostics": pd.concat(train_diag_parts, ignore_index=True)
        if train_diag_parts
        else pd.DataFrame(),
        "folds": pd.DataFrame(fold_records),
        "last_model": last_model,
        "last_scales": last_scales,
    }


PASS_META_COLUMNS = [
    "passId",
    "gameId",
    "period",
    "chanceId",
    "touchId",
    "startFrame",
    "endFrame",
    "startGameClock",
    "shotClock",
    "passerId",
    "interceptorId",
    "pass_outcome_class",
    "is_failed_pass",
    "turnover",
    "inbounds",
    "backcourt",
    "chance_transition",
    "chance_usable",
    "primary_touch_eligible",
    "touch_exclusion_reason",
    "true_receiverId",
    "truth_in_candidate_set",
    "candidate_side",
    "n_candidates_declared",
    "observed_flight_time_s",
    "passer_x_event",
    "passer_y_event",
    "ball_flight_available",
    "ball_flight_reason",
    "ball_release_x_event",
    "ball_release_y_event",
    "ball_dir_x",
    "ball_dir_y",
    "ball_speed_fps",
    "ball_t_clean_s",
    "ball_t_obs_s",
    "ball_deflected",
    "n_ball_frames",
    "n_clean_ball_frames",
    "fold_id",
]


def _summarize(scored: pd.DataFrame, rule: AutoAcceptRule) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    meta = scored.drop_duplicates("passId")[[c for c in PASS_META_COLUMNS if c in scored.columns]]
    summary = summarize_pass_level(scored).merge(meta, on="passId", how="left")
    return classify_decisions(summary, rule)


def _attach_gates_to_candidates(
    scored: pd.DataFrame,
    eligibility: pd.DataFrame,
    state: dict[str, Any],
) -> pd.DataFrame:
    """Broadcast the full pass-level gate onto the candidate-row probability table.

    A consumer reading probabilities directly must see the same gate as one
    reading the assignment / eligibility tables, or the gate is only as strong
    as the table someone happens to open. Matches
    ``configs/stage2_narrow_path.yaml`` ``gate_columns.written_to``.
    """
    if scored.empty:
        return scored
    out = scored.drop(columns=[c for c in GATE_COLUMNS if c in scored.columns])
    if not eligibility.empty:
        out = out.merge(eligibility[["passId", *GATE_COLUMNS]], on="passId", how="left")
    else:
        # Defensive: keep mode flags even if eligibility is empty.
        out["narrow_path_active"] = bool(state["narrow_path_active"])
        out["target_inference_mode"] = state["target_inference_mode"]
        out["human_audit_status"] = state["human_audit_status"]
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(
    failed: pd.DataFrame,
    completed: pd.DataFrame,
    folds: pd.DataFrame,
    unresolved: dict[str, Any],
    agreement: dict[str, Any],
    cohort_counts: dict[str, int],
    sensitivity: pd.DataFrame,
    audit_sheet: pd.DataFrame,
    checks: dict[str, dict[str, Any]],
    calibration: pd.DataFrame,
    gates: dict[str, Any],
) -> str:
    mix = decision_mix(failed)
    narrow = bool(gates["narrow_path_active"])
    lines = [
        "# Stage 2 failed-pass target inference report",
        "",
        f"**Freeze ID:** `{FREEZE_ID}`",
        "**Stage:** 2 — resolve failed-pass targets with explicit uncertainty",
        "**Design:** leave-one-game-out; every fold fits flight time, likelihood scales,",
        "and decision thresholds on nine games and scores only the tenth.",
        f"**Acceptance mode:** `{gates['target_inference_mode']}` "
        f"(lock `{gates['lock_id']}`, `configs/stage2_narrow_path.yaml`)",
        f"**Human audit:** `{gates['human_audit_status']}` — no blinded human review of",
        "failed-pass targets exists. The reviewer columns of the audit sheet are blank and",
        "stay blank; nothing in this report is a human judgement.",
        "",
        "## 1. Cohorts",
        "",
        "`complete` is tri-state in the raw feed and the three classes carry different evidence.",
        "",
        "| Pass class | N | Enters target inference |",
        "|---|---:|---|",
        f"| `complete` (usable for validation) | {cohort_counts['completed_scored']} | as a known-receiver control |",
        f"| `incomplete_turnover` (failed) | {cohort_counts['failed']} | **yes — the Stage 2 population** |",
        f"| `unknown_outcome` (`complete` null) | {cohort_counts['unknown']} | no: no end frame, receiver, or interceptor |",
        f"| completed passes excluded from control | {cohort_counts['completed_skipped']} | receiver outside the four-candidate set |",
        "",
        "`toReceiverId` is the interceptor. It is never a candidate and never a target:",
        "candidates come from the lineup containing the passer, so an opponent cannot appear.",
        "",
        "## 2. Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Failed passes scored | {unresolved['n_failed']} |",
        f"| Auto-accepted | {mix[DECISION_AUTO_ACCEPT]} ({_fmt(unresolved['auto_accept_pct'], 1)}%) |",
        f"| Ambiguous (audit sheet) | {mix[DECISION_AMBIGUOUS]} ({_fmt(unresolved['ambiguous_pct'], 1)}%) |",
        f"| Unresolved, no usable evidence | {mix[DECISION_UNRESOLVED]} ({_fmt(unresolved['unresolved_strict_pct'], 1)}%) |",
        f"| Unresolved before human audit (ambiguous + no evidence) | {mix[DECISION_AMBIGUOUS] + mix[DECISION_UNRESOLVED]} ({_fmt(unresolved['unresolved_pre_audit_pct'], 1)}%) |",
        f"| Held-out completed-pass top-1 agreement (diagnostic) | {_fmt(agreement['top1_accuracy'])} (n={agreement['n']}) |",
        f"| Held-out completed-pass auto-accept precision (diagnostic) | {_fmt(agreement['auto_accept_precision'])} |",
        f"| Held-out completed-pass auto-accept rate (diagnostic) | {_fmt(agreement['auto_accept_rate'])} |",
        "",
        "The last three rows are measured on **completed** passes, where the receiver is known.",
        "They are diagnostics of the scoring machinery. They are **not** failed-pass precision",
        "and may not be cited as evidence that any inferred failed target is correct; see",
        "section 3.1.",
        "",
        "## 3. Kill criterion: unresolved failed-pass targets > 10%",
        "",
        f"- Frozen budget: at most **{unresolved['max_unresolved_allowed']}** of {unresolved['n_failed']} failed passes.",
        f"- Strict unresolved (no candidate set, no ball flight, or no tracking): "
        f"**{unresolved['n_unresolved_strict']}** = {_fmt(unresolved['unresolved_strict_pct'], 1)}% — "
        f"{'BREACH' if unresolved['strict_breaches_kill'] else 'within budget'}.",
        f"- Pre-audit unresolved (everything without an accepted label): "
        f"**{unresolved['n_ambiguous'] + unresolved['n_unresolved_strict']}** = "
        f"{_fmt(unresolved['unresolved_pre_audit_pct'], 1)}% — "
        f"{'BREACH' if narrow else 'within budget'}.",
        "",
        "Both numbers are reported because they answer different questions. The strict figure is",
        "the share where the method has nothing to work with. The pre-audit figure is the honest",
        "share while the audit sheet is still blank, and it is the one to read against the kill line",
        "until a reviewer fills it in.",
        "",
    ]

    if narrow:
        lines += [
            "### Narrow path — ACTIVE and binding",
            "",
            f"Triggers recomputed from this run: `{'`, `'.join(gates['triggers'])}`.",
            "",
            "Stage 0's `on_inadequate_reliability` clause has fired, so the narrow path is the",
            "Stage 2 acceptance mode rather than a caveat. It is locked in",
            "`configs/stage2_narrow_path.yaml` and enforced in code by",
            "`src/passing_windows/narrow_path.py`:",
            "",
            "1. Failed passes are retained for **touch-level turnover and pass-attempt",
            "   features only**. No failed pass is a named-receiver completion label.",
            "2. Receiver-specific completion labels come from **completed passes with a known",
            "   receiver, and nothing else**. That is the default Stage 3+ training set.",
            "3. Auto-accepted inferred failures remain available as an *optional* named target",
            "   (`usable_as_named_failed_target`) but are excluded from",
            "   `receiver_specific_eligible`. A downstream stage may opt in only if it also",
            "   publishes the same result with them excluded.",
            "4. Ambiguous and unresolved failures may never be attributed to a named receiver.",
            "5. The interceptor is never substituted for the target under any path.",
            "",
            "Machine-readable gate columns are written to every Stage 2 pass table and",
            f"consolidated in `tables/stage2_label_eligibility.parquet` ({gates['n_passes']} passes):",
            "",
            "| Gate | Passes |",
            "|---|---:|",
            f"| `receiver_specific_eligible` (Stage 3+ default) | {gates['n_receiver_specific_eligible']} |",
            f"| `usable_as_receiver_completion_label` | {gates['n_receiver_completion_labels']} |",
            f"| `usable_as_named_failed_target` (opt-in, sensitivity required) | {gates['n_named_failed_targets_optional']} |",
            f"| failed passes restricted to touch level | {gates['n_touch_level_only']} |",
            "",
            f"Reliability mix: `{gates['reliability_mix']}`.",
            "",
            "### 3.1 Why the 0.968 figure does not close this gate",
            "",
            "The auto-accept precision above is measured on held-out **completed** passes. A",
            "completed pass reaches its receiver and is geometrically easier than an intercepted",
            "one, so that number is an optimistic upper bound on failed-pass precision, not an",
            "estimate of it. The project's own diagnostic",
            "`failed_residuals_within_completed_support` fails: a material share of accepted",
            "failures sit beyond the completed-correct residual support, which is direct evidence",
            "that the two populations differ. The completed-pass proxy is therefore recorded as",
            "**diagnostic only** in the lock, and citing it as failed-pass precision is a",
            "forbidden use.",
            "",
            "### 3.2 Human audit: DEFERRED / WAIVED",
            "",
            "The frozen protocol required a blinded human review of every ambiguous failure, a",
            "random auto-accepted sample, and completed controls, with the auto-accept margin",
            "re-selected against those labels. That review has **not** been performed. The",
            "reviewer columns in `artifacts/stage2_ambiguous_audit.csv` are blank and the",
            "pipeline refuses to run if anything fills them while the audit is deferred, because",
            "an invented reviewer decision would be indistinguishable downstream from a real one.",
            "Stage 2 is accepted under the narrow path *instead of* that gate, not as if it had",
            "passed it.",
            "",
        ]
    else:
        lines += [
            "### Narrow path — not active",
            "",
            "No trigger fired and the lock does not force narrow mode. Receiver-specific",
            "completion calibration may use auto-accepted inferred failures, still flagged",
            "provisional until the audit sheet is returned.",
            "",
        ]

    lines += [
        "## 4. Per-fold thresholds and held-out agreement",
        "",
        "Thresholds are **provisional and selected under a substituted protocol**. Stage 0 froze",
        "the procedure as `failed_pass_auto_accept_geometric_margin: TBD_TRAINING_FOLD`, chosen",
        "inside training folds to maximise *audit* agreement. No audit labels exist, so training",
        "folds' completed-pass receiver labels were used instead at a target precision of",
        f"{AUTO_ACCEPT_TARGET_PRECISION:.2f}. That is a documented protocol substitution, not the",
        "frozen rule, and it is the second reason the narrow-path lock is binding: the thresholds",
        "below are not qualified to license receiver-specific failed-pass claims.",
        "",
    ]
    if not folds.empty:
        show = folds[
            [
                "fold_id",
                "p_min",
                "prob_margin",
                "geom_margin_ft",
                "train_precision",
                "met_target",
                "scale_sigma_ball_ft",
                "scale_sigma_pos_ft",
                "scale_sigma_flight_s",
                "n_failed_heldout",
                "heldout_completed_top1_accuracy",
                "heldout_completed_auto_accept_precision",
                f"failed_{DECISION_AUTO_ACCEPT}",
                f"failed_{DECISION_AMBIGUOUS}",
                f"failed_{DECISION_UNRESOLVED}",
            ]
        ].round(4)
        lines.append(show.to_markdown(index=False))
    lines += ["", "## 5. Evidence channels", ""]
    lines += [
        "| Channel | Residual | Availability on failed passes |",
        "|---|---|---:|",
    ]
    if not failed.empty:
        lines += [
            f"| Lane alignment | perpendicular miss from the observed ball bearing to the projected "
            f"catch point, scaled by a tolerance that grows with distance and shrinks with the ball "
            f"chord | {int(failed['ball_flight_available'].sum())}/{len(failed)} |",
            f"| Flight-time compatibility | clean flight duration vs ball travel time to the "
            f"candidate | {int(failed['top_flight_residual_s'].notna().sum())}/{len(failed)} |",
            f"| Direction sign | candidates behind the release bearing are penalised | "
            f"{len(failed)}/{len(failed)} |",
        ]
    lines += [
        "",
        "Both residuals use a Student-t likelihood (nu = 3). A Gaussian fitted to the sharp peak of",
        "known-receiver residuals treats the genuine tail — a pass tipped at release, a receiver who",
        "cut after the ball left — as impossible, and then returns certainty on the passes it gets",
        "wrong. The heavy tail is what makes section 6's calibration hold.",
        "",
        "The flight-time channel is asymmetric for failures: an intercepted ball stops short, so a",
        "candidate farther away than the ball reached is expected and only weakly penalised, while a",
        "candidate the ball had already flown past is evidence against.",
        "",
        "## 6. Is `p_top` a probability?",
        "",
        "Held-out completed passes only, so the receiver is known and the fold never saw the game.",
        "",
    ]
    if not calibration.empty:
        lines.append(calibration.round(3).to_markdown(index=False))
    lines += [
        "",
        "## 7. Automated geometry self-consistency",
        "",
        "Invariants must hold for the output to be valid; diagnostics describe the data and are",
        "reported rather than enforced.",
        "",
        "| Check | Severity | Result | Detail |",
        "|---|---|---|---|",
    ]
    for name, res in checks.items():
        status = "PASS" if res.get("ok") else "FAIL"
        detail = {k: v for k, v in res.items() if k not in ("ok", "severity")}
        lines.append(f"| `{name}` | {res.get('severity', '')} | **{status}** | `{detail}` |")

    lines += [
        "",
        "## 8. Sensitivity: excluding all inferred failures",
        "",
        "Stub for Stage 4. Dropping every inferred failure removes the only failure evidence",
        "attributable to a receiver, which forces receiver-specific completion rates to 1.0 by",
        "construction — the reason the frozen narrow path restricts that calibration rather than",
        "silently accepting it.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Receivers with at least one completed pass | {int((sensitivity['n_completed'] > 0).sum()) if not sensitivity.empty else 0} |",
        f"| Receivers gaining an inferred failure | {int((sensitivity['n_inferred_failed'] > 0).sum()) if not sensitivity.empty else 0} |",
        f"| Attempts added by inferred failures | {int(sensitivity['n_inferred_failed'].sum()) if not sensitivity.empty else 0} |",
        f"| Max single-receiver share of attempts that is inferred | "
        f"{_fmt(float(sensitivity['share_of_attempts_inferred'].max()) if not sensitivity.empty else np.nan)} |",
        "",
        "Full table: `tables/stage2_failure_sensitivity.parquet`.",
        "",
        "## 9. Manual audit export",
        "",
        f"- `artifacts/stage2_ambiguous_audit.csv`: {len(audit_sheet)} candidate rows across "
        f"{audit_sheet['passId'].nunique() if not audit_sheet.empty else 0} passes.",
        "- Contains every failed pass (auto-accepted, ambiguous, and unresolved) plus a seeded",
        f"  sample of {AUDIT_COMPLETED_PER_GAME} completed passes per game as a blind control.",
        "- One row per candidate, carrying release coordinates, projected catch point, ball release",
        "  point and bearing, and the per-channel residuals — the tabular court-frame sync that",
        "  stands in until the Stage 9 interactive viewer exists.",
        f"- Reviewer columns (`{'`, `'.join(REVIEWER_COLUMNS)}`) are **blank and must stay blank**",
        f"  while `human_audit_status = {gates['human_audit_status']}`. The pipeline asserts this",
        "  and exits non-zero if any value appears, so a fabricated reviewer decision cannot enter",
        "  the study by accident. Mark `reviewer_is_intended_target` on the candidate row believed",
        "  to be the target, then record `reviewer_decision`, `reviewer_confidence`, and",
        "  `reviewer_reason` once per pass — only if a real blinded review is actually performed.",
        "- The sheet also carries tabular release geometry rather than synchronized court-frame",
        "  playback. That is another reason the audit is recorded as deferred rather than merely",
        "  outstanding: the review material the frozen protocol asked for does not exist yet.",
        "",
        "## 10. Known limitations",
        "",
        "1. Completed passes are a **proxy** for audit labels and are recorded in the lock as",
        "   diagnostic only. A completed pass reaches its receiver, so its geometry is easier than",
        "   an intercepted pass; the held-out precision reported here is an optimistic bound on",
        "   failed-pass precision and may not be cited as failed-pass precision.",
        "2. Candidate motion is projected from release-time velocity only. Cuts that start after",
        "   release are invisible to the projection by design (no future leakage).",
        "3. A deflection ends the clean flight segment, so heavily deflected passes contribute a",
        "   short bearing and a weak flight-time channel.",
        "4. `matchup_matched == False` states and null velocities are carried as missing, never as",
        "   zero; a candidate with no tracking receives no likelihood rather than a default share.",
        "5. **No failed-pass target in this study has been verified by a human.** Every",
        "   `inferred_targetId` is an unaudited geometric inference, which is exactly what the",
        "   narrow-path gate columns encode.",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_verification(
    unresolved: dict[str, Any],
    agreement: dict[str, Any],
    cohort_counts: dict[str, int],
    checks: dict[str, dict[str, Any]],
    folds: pd.DataFrame,
    gates: dict[str, Any],
) -> str:
    mix_accept = unresolved["n_auto_accepted"]
    logo_clean = not folds.empty and bool((folds["n_train_games"] == len(folds) - 1).all())
    lines = [
        "# Stage 2 USER VERIFICATION CHECKLIST",
        "",
        f"**Freeze ID:** `{FREEZE_ID}`",
        f"**Narrow-path lock:** `{gates['lock_id']}` (`configs/stage2_narrow_path.yaml`)",
        f"**Acceptance mode:** `{gates['target_inference_mode']}`",
        f"**Human audit:** `{gates['human_audit_status']}` — NOT complete, NOT simulated",
        "**Purpose:** approve failed-pass target inference *under the narrow path* before Stage 3.",
        "",
        "This checklist is **not** a claim that the frozen human-audit gate was passed. It was",
        "not. Stage 2 is offered for acceptance under Stage 0's `on_inadequate_reliability`",
        "fallback, which the run below triggers and which the code now enforces. See",
        "[`STAGE2_NARROW_PATH_LOCK.md`](STAGE2_NARROW_PATH_LOCK.md) for the decision and",
        "[`STAGE2_AUDIT_REPORT.md`](STAGE2_AUDIT_REPORT.md) for the adversarial audit that",
        "forced it.",
        "",
        "Mark each item **APPROVE** / **REJECT** / **CHANGE REQUEST**.",
        "",
        "## A. Sample sizes",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Failed passes (`complete == False`) | {cohort_counts['failed']} |",
        f"| `unknown_outcome` passes held out of inference | {cohort_counts['unknown']} |",
        f"| Completed passes used as known-receiver controls | {cohort_counts['completed_scored']} |",
        f"| Auto-accepted failed targets | {mix_accept} |",
        f"| Ambiguous failed targets | {unresolved['n_ambiguous']} |",
        f"| Unresolved (no evidence) failed targets | {unresolved['n_unresolved_strict']} |",
        "",
        "## B. Frozen rules",
        "",
        "| # | Rule | Pipeline result | Verify |",
        "|---|---|---|---|",
        f"| B1 | `toReceiverId` never used as intended target | "
        f"{'PASS' if checks['interceptor_never_target']['ok'] else 'FAIL'} | ☐ |",
        f"| B2 | Probabilities form a simplex per pass | "
        f"{'PASS' if checks['probability_simplex']['ok'] else 'FAIL'} | ☐ |",
        f"| B3 | Passer never a candidate for their own pass | "
        f"{'PASS' if checks['target_not_passer']['ok'] else 'FAIL'} | ☐ |",
        f"| B4 | Accepted labels lie inside the candidate set | "
        f"{'PASS' if checks['assignment_in_candidate_set']['ok'] else 'FAIL'} | ☐ |",
        f"| B5 | Non-accepted passes carry no hard label | "
        f"{'PASS' if checks['unresolved_has_no_label']['ok'] else 'FAIL'} | ☐ |",
        f"| B6 | Mirroring does not change probabilities | "
        f"{'PASS' if checks['mirror_invariance']['ok'] else 'FAIL'} | ☐ |",
        f"| B7 | Flight-time model monotonic in distance | "
        f"{'PASS' if checks['flight_time_monotonic']['ok'] else 'FAIL'} | ☐ |",
        f"| B8 | Ball release anchored on the passer | "
        f"{'PASS' if checks['release_point_matches_passer']['ok'] else 'FAIL'} | ☐ |",
        f"| B9 | Tri-state `complete`: unknown-outcome passes excluded from the failed cohort | "
        f"{cohort_counts['unknown']} held out | ☐ |",
        f"| B10 | Thresholds selected inside training folds only (leave-one-game-out) | "
        f"{'PASS' if logo_clean else 'FAIL'} | ☐ |",
        "",
        "## C. Kill criteria",
        "",
        "| # | Criterion | Value | Verdict | Verify |",
        "|---|---|---:|---|---|",
        f"| C1 | Unresolved (strict) > 10% | {_fmt(unresolved['unresolved_strict_pct'], 1)}% | "
        f"{'BREACH' if unresolved['strict_breaches_kill'] else 'OK'} | ☐ |",
        f"| C2 | Unresolved pre-audit > 10% | {_fmt(unresolved['unresolved_pre_audit_pct'], 1)}% | "
        f"{'BREACH — narrow path' if unresolved['pre_audit_breaches_kill'] else 'OK'} | ☐ |",
        f"| C3 | Held-out completed-pass agreement (**diagnostic only**) | {_fmt(agreement['top1_accuracy'])} | — | ☐ |",
        f"| C4 | Held-out auto-accept precision (**diagnostic only, NOT failed-pass precision**) | "
        f"{_fmt(agreement['auto_accept_precision'])} | — | ☐ |",
        "",
        "C3 and C4 are measured on completed passes. They describe the scoring machinery and",
        "carry no information about whether any inferred failed target is correct.",
        "",
        "## D. Narrow-path acceptance criteria (replaces the human-audit gate)",
        "",
        "Stage 2 is accepted only if every row below holds. These are machine-checked; the",
        "pipeline exits non-zero on any violation.",
        "",
        "| # | Criterion | Pipeline result | Verify |",
        "|---|---|---|---|",
        f"| D1 | `target_inference_mode` locked to `narrow_path` | "
        f"{'PASS' if gates['target_inference_mode'] == 'narrow_path' else 'FAIL'} | ☐ |",
        f"| D2 | Narrow path active, triggers recomputed from this run | "
        f"{'PASS' if gates['narrow_path_active'] else 'FAIL'} (`{'`, `'.join(gates['triggers'])}`) | ☐ |",
        f"| D3 | Gate columns present on every Stage 2 pass table | PASS ({len(GATE_COLUMNS)} columns) | ☐ |",
        f"| D4 | Receiver-specific eligible = completed passes with known receivers only | "
        f"{gates['n_receiver_specific_eligible']} of {gates['n_passes']} | ☐ |",
        f"| D5 | No failed pass carries `usable_as_receiver_completion_label` | "
        f"{gates['n_receiver_completion_labels'] - gates['n_receiver_specific_eligible'] == 0} | ☐ |",
        f"| D6 | Failed passes restricted to touch-level turnover use | {gates['n_touch_level_only']} | ☐ |",
        f"| D7 | Auto-accepted failures opt-in only, sensitivity check required | "
        f"{gates['n_named_failed_targets_optional']} flagged | ☐ |",
        "| D8 | Interceptor never a target — asserted, not merely reported | PASS | ☐ |",
        f"| D9 | Reviewer columns blank; no fabricated reviewer decisions | PASS "
        f"(`human_audit_status = {gates['human_audit_status']}`) | ☐ |",
        "",
        "## E. Human audit status: DEFERRED / WAIVED",
        "",
        "| # | Item | Status |",
        "|---|---|---|",
        "| E1 | Blinded human review of ambiguous failures | **NOT PERFORMED** |",
        "| E2 | Blinded review of random auto-accepted sample | **NOT PERFORMED** |",
        "| E3 | Completed-pass control review | **NOT PERFORMED** |",
        "| E4 | Margin re-selected against audit labels | **NOT PERFORMED** (completed-pass proxy used instead) |",
        "| E5 | Synchronized court-frame review material | **NOT BUILT** (tabular sheet only) |",
        "| E6 | Reviewer columns in `stage2_ambiguous_audit.csv` | **BLANK BY DESIGN — do not fill with model output** |",
        "",
        "The user waived human checklist sign-off and directed autonomous execution. Waiving the",
        "sign-off does not create the audit, so the audit is recorded as deferred and the narrow",
        "path carries the reliability burden instead. Reopening the lock requires the real review",
        "listed above, a threshold re-selection against those labels, and a new lock revision.",
        "",
        "## F. Artifacts present",
        "",
        "| # | Artifact | Verify |",
        "|---|---|---|",
        "| F1 | `artifacts/stage2_target_inference_report.md` | ☐ |",
        "| F2 | `artifacts/stage2_ambiguous_audit.csv` (reviewer columns blank) | ☐ |",
        "| F3 | `artifacts/stage2_geometry_selfcheck.md` | ☐ |",
        "| F4 | `artifacts/stage2_fold_thresholds.json` | ☐ |",
        "| F5 | `artifacts/stage2_narrow_path_state.json` | ☐ |",
        "| F6 | `artifacts/STAGE2_NARROW_PATH_LOCK.md` | ☐ |",
        "| F7 | `configs/stage2_narrow_path.yaml` | ☐ |",
        "| F8 | `tables/stage2_label_eligibility.parquet` | ☐ |",
        "| F9 | `artifacts/stage2_output_manifest.json` verifies clean | ☐ |",
        "",
        "## Sign-off",
        "",
        "| Role | Name | Date | Overall |",
        "|---|---|---|---|",
        "| User / analyst | _(waived: autonomous execution directed by user)_ | | APPROVE / REJECT |",
        "| Human failed-pass target reviewer | _(none — audit deferred)_ | | N/A |",
        "",
        "**If APPROVE:** Stage 3 may begin **under the narrow path only**. It must join",
        "`tables/stage2_label_eligibility.parquet`, train completion labels on",
        "`receiver_specific_eligible == True` only, and treat every failed pass as touch-level",
        "turnover evidence. No Stage 3+ output may attribute a failed pass to a named receiver",
        "without opting in via `usable_as_named_failed_target` and publishing the exclusion",
        "sensitivity alongside it.",
        "",
    ]
    if not folds.empty:
        lines += [
            "## G. Fold thresholds actually applied",
            "",
            folds[["fold_id", "p_min", "prob_margin", "geom_margin_ft", "train_precision", "met_target"]]
            .round(4)
            .to_markdown(index=False),
            "",
        ]
    return "\n".join(lines) + "\n"


def render_selfcheck(checks: dict[str, dict[str, Any]], folds: pd.DataFrame) -> str:
    lines = [
        "# Stage 2 geometry self-consistency report",
        "",
        f"**Freeze ID:** `{FREEZE_ID}`",
        "",
        "Automated checks only. These prove the inference machinery is internally coherent;",
        "they do not substitute for the human audit that `stage2_ambiguous_audit.csv` collects.",
        "An invariant failure fails the build; a diagnostic failure is a statement about the data.",
        "",
        "| Check | Severity | Result |",
        "|---|---|---|",
    ]
    for name, res in checks.items():
        lines.append(
            f"| `{name}` | {res.get('severity', '')} | {'PASS' if res.get('ok') else 'FAIL'} |"
        )
    lines += ["", "## Detail", ""]
    for name, res in checks.items():
        lines += [f"### `{name}`", "", "```", str(res), "```", ""]
    if not folds.empty:
        lines += [
            "## Per-fold likelihood scales",
            "",
            folds[
                [
                    "fold_id",
                    "scale_sigma_ball_ft",
                    "scale_sigma_pos_ft",
                    "scale_flight_offset_s",
                    "scale_sigma_flight_s",
                    "flight_time_bins",
                    "distance_support_max_ft",
                ]
            ]
            .round(4)
            .to_markdown(index=False),
            "",
            "Scales are robust (1.4826 x MAD) residual widths of the **known** receiver in the nine",
            "training games of each fold, so the likelihood width is data-driven rather than assumed.",
            "`sigma_ball_ft` sets the bearing error (divided by the observed ball chord) and",
            "`sigma_pos_ft` the receiver position/projection error; the lateral tolerance at a",
            "candidate combines them and therefore widens with distance and with short ball chords.",
            "",
            "Checks that need a fitted model (`mirror_invariance`, `flight_time_monotonic`) use the",
            "last fold's model; every fold's scales are listed above so the choice is inspectable.",
            "",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 failed-pass target inference")
    parser.add_argument("--force", action="store_true", help="Rebuild cached per-game states")
    parser.add_argument("--games", nargs="*", type=int, default=None, help="Subset of game IDs")
    parser.add_argument("--verify-manifest", action="store_true", help="Verify the Stage 2 manifest and exit")
    args = parser.parse_args()

    manifest_path = ARTIFACTS / "stage2_output_manifest.json"
    if args.verify_manifest:
        result = verify_output_manifest(manifest_path, root=PACKAGE_ROOT)
        print(result)
        return 0 if result.get("ok") else 2

    t0 = time.time()
    freeze = load_freeze()
    game_ids = args.games or game_ids_from_freeze(freeze)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    try:
        states, game_summaries = load_states(game_ids, force=args.force)
    except Exception:
        traceback.print_exc()
        return 1
    if states.empty:
        print("No Stage 2 states built; nothing to do.")
        return 1

    print(f"States: {len(states)} rows across {states['passId'].nunique()} passes")
    results = run_folds(states, freeze)
    scored = results["scored"]
    failed = results["failed"]
    completed = results["completed"]
    folds = results["folds"]

    cohort_counts = _cohort_counts(game_ids, states, failed, completed)
    unresolved = unresolved_summary(failed, UNRESOLVED_KILL_PCT)
    agreement = completed_pass_agreement(completed)
    sensitivity = exclusion_sensitivity_stub(failed, completed)
    calibration = probability_calibration(completed)
    audit_sheet = build_audit_sheet(scored, failed, completed, seed=int(freeze["rng"]["master_seed"]))
    checks = run_self_consistency_suite(
        scored, failed, completed, states, results["last_model"], results["last_scales"]
    )

    # The narrow-path lock is applied before anything is written, so no Stage 2
    # table can reach disk without its gate columns.
    lock = load_narrow_path_lock()
    state = evaluate_narrow_path(unresolved, lock=lock)
    failed = apply_label_gates(failed, cohort=COHORT_FAILED, state=state)
    completed = apply_label_gates(completed, cohort=COHORT_COMPLETED, state=state)
    eligibility = build_label_eligibility(failed, completed)
    scored = _attach_gates_to_candidates(scored, eligibility, state)
    try:
        gates = assert_narrow_path_gates(failed, completed, eligibility, state=state)
        assert_reviewer_columns_blank(
            audit_sheet, REVIEWER_COLUMNS, human_audit_status=state["human_audit_status"]
        )
        assert_audit_sheet_on_disk_is_blank(
            ARTIFACTS / "stage2_ambiguous_audit.csv",
            REVIEWER_COLUMNS,
            human_audit_status=state["human_audit_status"],
        )
    except NarrowPathViolation as exc:
        print("NARROW-PATH GATE VIOLATION:", exc)
        return 2

    write_table(states, TABLES / "stage2_pass_candidate_states.parquet")
    write_table(scored, TABLES / "stage2_target_probabilities.parquet")
    write_table(failed, TABLES / "stage2_failed_pass_assignments.parquet")
    write_table(completed, TABLES / "stage2_completed_pass_validation.parquet")
    write_table(folds, TABLES / "stage2_fold_thresholds.parquet")
    write_table(sensitivity, TABLES / "stage2_failure_sensitivity.parquet")
    write_table(calibration, TABLES / "stage2_probability_calibration.parquet")
    write_table(eligibility, TABLES / "stage2_label_eligibility.parquet")
    write_table(audit_sheet, ARTIFACTS / "stage2_ambiguous_audit.csv")

    write_json({**gates, "lock": lock}, ARTIFACTS / "stage2_narrow_path_state.json")

    write_json(
        {
            "freeze_id": FREEZE_ID,
            "stage": STAGE,
            "provisional": True,
            "narrow_path": gates,
            "selection_procedure": (
                "Grid search inside each outer-fold training set over (p_min, prob_margin, "
                "geom_margin_ft); keep rules reaching the completed-pass precision target, "
                "pick maximum coverage. This substitutes completed-pass receiver labels for "
                "the human audit labels Stage 0 specifies, which is a protocol substitution, "
                "not the frozen procedure: the resulting thresholds are provisional and the "
                "narrow-path lock bars their output from receiver-specific use."
            ),
            "completed_pass_proxy_role": "diagnostic_only",
            "target_precision": AUTO_ACCEPT_TARGET_PRECISION,
            "folds": folds.to_dict("records"),
            "unresolved": unresolved,
            "completed_agreement": agreement,
            "cohorts": cohort_counts,
        },
        ARTIFACTS / "stage2_fold_thresholds.json",
    )
    report_md = render_report(
        failed,
        completed,
        folds,
        unresolved,
        agreement,
        cohort_counts,
        sensitivity,
        audit_sheet,
        checks,
        calibration,
        gates,
    )
    verification_md = render_verification(
        unresolved, agreement, cohort_counts, checks, folds, gates
    )
    try:
        assert_no_failed_precision_claim(report_md, name="stage2_target_inference_report.md")
        assert_no_failed_precision_claim(verification_md, name="STAGE2_VERIFICATION.md")
    except NarrowPathViolation as exc:
        print("NARROW-PATH GATE VIOLATION:", exc)
        return 2

    write_markdown(report_md, ARTIFACTS / "stage2_target_inference_report.md")
    write_markdown(render_selfcheck(checks, folds), ARTIFACTS / "stage2_geometry_selfcheck.md")
    write_markdown(verification_md, ARTIFACTS / "STAGE2_VERIFICATION.md")
    write_json({"gameId_summaries": game_summaries}, ARTIFACTS / "stage2_game_summaries.json")

    outputs = [
        TABLES / "stage2_pass_candidate_states.parquet",
        TABLES / "stage2_target_probabilities.parquet",
        TABLES / "stage2_failed_pass_assignments.parquet",
        TABLES / "stage2_completed_pass_validation.parquet",
        TABLES / "stage2_fold_thresholds.parquet",
        TABLES / "stage2_failure_sensitivity.parquet",
        TABLES / "stage2_probability_calibration.parquet",
        TABLES / "stage2_label_eligibility.parquet",
        ARTIFACTS / "stage2_ambiguous_audit.csv",
        ARTIFACTS / "stage2_target_inference_report.md",
        ARTIFACTS / "stage2_geometry_selfcheck.md",
        ARTIFACTS / "stage2_fold_thresholds.json",
        ARTIFACTS / "stage2_narrow_path_state.json",
        ARTIFACTS / "stage2_game_summaries.json",
        PACKAGE_ROOT / "configs" / "stage2_narrow_path.yaml",
    ]
    manifest = build_output_manifest(
        outputs,
        root=PACKAGE_ROOT,
        manifest_path=manifest_path,
        sign_off_paths=[ARTIFACTS / "STAGE2_VERIFICATION.md"],
    )
    manifest["freeze_id"] = FREEZE_ID
    manifest["stage"] = STAGE
    manifest["narrow_path_lock_id"] = gates["lock_id"]
    manifest["target_inference_mode"] = gates["target_inference_mode"]
    manifest["human_audit_status"] = gates["human_audit_status"]
    manifest["elapsed_seconds"] = round(time.time() - t0, 1)
    write_json(manifest, manifest_path)

    verification = verify_output_manifest(manifest_path, root=PACKAGE_ROOT)
    mix = decision_mix(failed)
    print(f"Stage 2 complete in {time.time() - t0:.1f}s")
    print(
        f"Failed passes: {unresolved['n_failed']} | auto-accept {mix[DECISION_AUTO_ACCEPT]} "
        f"({unresolved['auto_accept_pct']:.1f}%) | ambiguous {mix[DECISION_AMBIGUOUS]} "
        f"({unresolved['ambiguous_pct']:.1f}%) | unresolved {mix[DECISION_UNRESOLVED]} "
        f"({unresolved['unresolved_strict_pct']:.1f}%)"
    )
    print(
        f"Held-out COMPLETED-pass top-1 agreement: {agreement['top1_accuracy']:.3f} "
        f"(auto-accept precision {agreement['auto_accept_precision']:.3f}) "
        "[diagnostic only; not failed-pass precision]"
    )
    print(
        f"Narrow path: {'ACTIVE' if gates['narrow_path_active'] else 'inactive'} "
        f"({gates['lock_id']}) | human audit: {gates['human_audit_status']} | "
        f"receiver-specific eligible {gates['n_receiver_specific_eligible']}/{gates['n_passes']} "
        f"| touch-level only {gates['n_touch_level_only']}"
    )
    diagnostics = failed_diagnostics(checks)
    if diagnostics:
        print("SELF-CHECK DIAGNOSTICS FLAGGED (reported, not fatal):", diagnostics)
    invariants = failed_invariants(checks)
    if invariants:
        print("SELF-CHECK INVARIANT FAILURES:", invariants)
        return 2
    if not verification.get("ok"):
        print("MANIFEST VERIFICATION FAILED:", verification)
        return 2
    return 0


def _cohort_counts(
    game_ids: list[int],
    states: pd.DataFrame,
    failed: pd.DataFrame,
    completed: pd.DataFrame,
) -> dict[str, int]:
    skipped = 0
    unknown = 0
    for gid in game_ids:
        path = _game_dir(gid) / "stage2_game_summary.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            skipped += int(meta.get("n_skipped_completed", 0))
            unknown += int(meta.get("n_unknown_outcome", 0))
    return {
        "failed": int(len(failed)),
        "completed_scored": int(len(completed)),
        "completed_skipped": skipped,
        "unknown": unknown,
        "states_rows": int(len(states)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
