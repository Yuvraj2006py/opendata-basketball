"""Stage 3: reconstruct four candidate receiver options per sampled frame.

For each eligible touch model frame and each of four teammates:

1. Fit fold-safe flight time from distance (never using the held-out game).
2. Project the catch point from causal velocity (null velocity ≠ zero).
3. Evaluate **direct** and **lead** trajectories; keep both feature families and
   an explicit ``trajectory_choice`` that is *not* score-maximizing.
4. Derive lane clearance, temporal interception margin, separation, pressure,
   recovery time, and tracking-confidence features.
5. Reject candidates beyond fold attempted-pass distance support.
6. Join narrow-path label eligibility so completion-training targets stay
   restricted to ``receiver_specific_eligible`` completed known receivers.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .flight_time import (
    FoldFlightBundle,
    outside_distance_support,
    predict_flight_time,
)
from .joins import build_candidate_states
from .lane_geometry import (
    DEFAULT_DEFENDER_SPEED_FPS,
    DEFAULT_N_PATH_SAMPLES,
    DEFAULT_PASS_SPEED_FPS,
    geometry_only_passability_score,
    help_defender_density,
    mean_pred_error,
    min_lane_clearance,
    min_temporal_interception_margin,
    motion_toward_rim,
    passer_pressure,
    path_length,
    pass_angle_deg,
    project_receiver_catch,
    receiver_separation_at_catch,
    rim_relative_features,
    sideline_baseline_proximity,
    target_defender_recovery_time,
)
from .narrow_path import (
    BOOLEAN_GATE_COLUMNS,
    GATE_COLUMNS,
    assert_gate_columns_present,
    assert_interceptor_never_target,
    receiver_completion_training_labels,
)
from .velocities import FRAME_RATE_HZ

TRAJECTORY_DIRECT = "direct"
TRAJECTORY_LEAD = "lead"
TRAJECTORY_CHOICES = (TRAJECTORY_DIRECT, TRAJECTORY_LEAD)

# Explicit, non-score rule: use lead when causal velocity is usable, else direct.
# Never pick whichever trajectory yields the higher geometry score.
TRAJECTORY_RULE = "lead_if_velocity_ok_else_direct"

FREEZE_MODEL_HZ = 5
FREEZE_FRAME_STRIDE = 5
SAMPLING_MODEL_5HZ = "model_5hz"
SAMPLING_PASS_RELEASE = "pass_release"


def choose_trajectory_explicit(*, receiver_velocity_ok: bool) -> str:
    """Frozen explicit trajectory rule — independent of geometry scores."""
    return TRAJECTORY_LEAD if bool(receiver_velocity_ok) else TRAJECTORY_DIRECT


def _finite_or_nan(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _player_state(row: dict[str, Any] | pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {
            "playerId": None,
            "x": float("nan"),
            "y": float("nan"),
            "vx": float("nan"),
            "vy": float("nan"),
            "velocity_ok": False,
            "predError": float("nan"),
            "isDetected": False,
        }
    if isinstance(row, pd.Series):
        row = row.to_dict()
    vok = row.get("velocity_ok", False)
    if vok is None or (isinstance(vok, float) and np.isnan(vok)):
        vok = False
    return {
        "playerId": int(row["playerId"]) if row.get("playerId") is not None and pd.notna(row.get("playerId")) else None,
        "x": _finite_or_nan(row.get("x_event", row.get("x"))),
        "y": _finite_or_nan(row.get("y_event", row.get("y"))),
        "vx": _finite_or_nan(row.get("vx")),
        "vy": _finite_or_nan(row.get("vy")),
        "velocity_ok": bool(vok) and np.isfinite(_finite_or_nan(row.get("vx"))) and np.isfinite(_finite_or_nan(row.get("vy"))),
        "predError": _finite_or_nan(row.get("predError")),
        "isDetected": bool(row.get("isDetected")) if row.get("isDetected") is not None and pd.notna(row.get("isDetected")) else False,
    }


def trajectory_catch_point(
    *,
    passer: dict[str, Any],
    receiver: dict[str, Any],
    flight_time_s: float,
    trajectory: str,
) -> tuple[float, float, bool, bool]:
    """Catch point under an explicit trajectory assumption.

    * ``direct``: aim at the receiver's *current* position (no lead).
    * ``lead``: aim at the causally projected position over ``flight_time_s``.
    """
    if trajectory == TRAJECTORY_DIRECT:
        return (
            float(receiver["x"]),
            float(receiver["y"]),
            False,
            False,
        )
    if trajectory == TRAJECTORY_LEAD:
        return project_receiver_catch(
            float(receiver["x"]),
            float(receiver["y"]),
            float(receiver["vx"]),
            float(receiver["vy"]),
            float(flight_time_s) if np.isfinite(flight_time_s) else 0.0,
            velocity_ok=bool(receiver["velocity_ok"]),
        )
    raise ValueError(f"unknown trajectory: {trajectory}")


def _features_for_trajectory(
    *,
    passer: dict[str, Any],
    receiver: dict[str, Any],
    defenders: Sequence[dict[str, Any]],
    assigned_defender: dict[str, Any] | None,
    catch_x: float,
    catch_y: float,
    flight_time_s: float,
    distance_support_reject: bool,
    prefix: str,
) -> dict[str, Any]:
    passer_xy = (float(passer["x"]), float(passer["y"]))
    catch_xy = (float(catch_x), float(catch_y))
    def_xys = [(float(d["x"]), float(d["y"])) for d in defenders]

    dist = path_length(passer_xy[0], passer_xy[1], catch_xy[0], catch_xy[1])
    clearance = min_lane_clearance(passer_xy, catch_xy, def_xys)

    # Align ball arrival with fold flight-time model: speed = distance / t_hat.
    # Keep a constant-35-fps ablation column for transparency.
    if np.isfinite(dist) and dist > 0 and np.isfinite(flight_time_s) and flight_time_s > 1e-6:
        pass_speed = float(dist / flight_time_s)
    else:
        pass_speed = float(DEFAULT_PASS_SPEED_FPS)
    margin, interceptor_id = min_temporal_interception_margin(
        passer_xy,
        catch_xy,
        list(defenders),
        pass_speed_fps=pass_speed,
        defender_speed_fps=DEFAULT_DEFENDER_SPEED_FPS,
        n_path_samples=DEFAULT_N_PATH_SAMPLES,
    )
    margin_const, _ = min_temporal_interception_margin(
        passer_xy,
        catch_xy,
        list(defenders),
        pass_speed_fps=DEFAULT_PASS_SPEED_FPS,
        defender_speed_fps=DEFAULT_DEFENDER_SPEED_FPS,
        n_path_samples=DEFAULT_N_PATH_SAMPLES,
    )

    interceptor_vx = interceptor_vy = float("nan")
    interceptor_vok = False
    if interceptor_id is not None:
        for d in defenders:
            if d.get("playerId") is not None and int(d["playerId"]) == int(interceptor_id):
                interceptor_vok = bool(d.get("velocity_ok", False))
                interceptor_vx = float(d["vx"]) if interceptor_vok else float("nan")
                interceptor_vy = float(d["vy"]) if interceptor_vok else float("nan")
                break

    sep = receiver_separation_at_catch(catch_xy, def_xys)
    pressure = passer_pressure(passer_xy, def_xys)
    recovery = target_defender_recovery_time(catch_xy, assigned_defender)
    help_n = help_defender_density(catch_xy, def_xys)
    sideline, baseline = sideline_baseline_proximity(catch_xy[0], catch_xy[1])
    rim = rim_relative_features(catch_xy[0], catch_xy[1])
    toward = motion_toward_rim(
        float(receiver["x"]),
        float(receiver["y"]),
        float(receiver["vx"]),
        float(receiver["vy"]),
        velocity_ok=bool(receiver["velocity_ok"]),
    )
    lane_pred = mean_pred_error(
        [passer.get("predError"), receiver.get("predError")]
        + [d.get("predError") for d in defenders]
    )
    geom_score = geometry_only_passability_score(
        distance_ft=dist,
        min_clearance_ft=clearance,
        min_temporal_margin_s=margin,
        receiver_sep_ft=sep,
        outside_support=bool(distance_support_reject),
    )
    return {
        f"{prefix}catch_x_event": catch_xy[0],
        f"{prefix}catch_y_event": catch_xy[1],
        f"{prefix}pass_distance_ft": dist,
        f"{prefix}pass_angle_deg": pass_angle_deg(passer_xy, catch_xy),
        f"{prefix}trajectory_length_ft": dist,
        f"{prefix}flight_time_s": float(flight_time_s) if np.isfinite(flight_time_s) else float("nan"),
        f"{prefix}pass_speed_fps": pass_speed,
        f"{prefix}min_lane_clearance_ft": clearance,
        f"{prefix}min_temporal_margin_s": margin,
        f"{prefix}min_temporal_margin_const35_s": margin_const,
        f"{prefix}intercepting_defenderId": interceptor_id if interceptor_id is not None else pd.NA,
        f"{prefix}intercepting_defender_vx": interceptor_vx,
        f"{prefix}intercepting_defender_vy": interceptor_vy,
        f"{prefix}intercepting_defender_velocity_ok": interceptor_vok,
        f"{prefix}receiver_sep_ft": sep,
        f"{prefix}passer_pressure_ft": pressure,
        f"{prefix}target_defender_recovery_s": recovery,
        f"{prefix}help_density_12ft": help_n,
        f"{prefix}sideline_prox_ft": sideline,
        f"{prefix}baseline_prox_ft": baseline,
        f"{prefix}dist_to_rim_ft": rim["dist_to_rim_ft"],
        f"{prefix}angle_to_rim_deg": rim["angle_to_rim_deg"],
        f"{prefix}catch_x_norm": rim["x_norm"],
        f"{prefix}catch_y_norm": rim["y_norm"],
        f"{prefix}court_region": rim["court_region"],
        f"{prefix}receiver_toward_rim_fps": toward,
        f"{prefix}lane_predError_mean": lane_pred,
        f"{prefix}geometry_passability_score": geom_score,
        f"{prefix}rejected_outside_support": bool(distance_support_reject),
    }


def reconstruct_one_candidate(
    *,
    passer: dict[str, Any],
    receiver: dict[str, Any],
    defenders: Sequence[dict[str, Any]],
    assigned_defender: dict[str, Any] | None,
    matchup_matched: bool,
    flight_bundle: FoldFlightBundle,
) -> dict[str, Any]:
    """Full Stage 3 feature dict for one (frame, candidate) pair."""
    tracked = np.isfinite(receiver["x"]) and np.isfinite(receiver["y"])
    passer_ok = np.isfinite(passer["x"]) and np.isfinite(passer["y"])

    # Initial distance for flight-time lookup uses current separation (causal).
    d0 = (
        path_length(passer["x"], passer["y"], receiver["x"], receiver["y"])
        if tracked and passer_ok
        else float("nan")
    )
    t_hat = float(predict_flight_time(flight_bundle.flight_model, d0)[()]) if np.isfinite(d0) else float("nan")

    # Iterate once so lead distance / flight time are consistent.
    lead_x, lead_y, lead_used_vel, lead_clipped = trajectory_catch_point(
        passer=passer, receiver=receiver, flight_time_s=t_hat, trajectory=TRAJECTORY_LEAD
    )
    d_lead = path_length(passer["x"], passer["y"], lead_x, lead_y) if passer_ok else float("nan")
    t_lead = float(predict_flight_time(flight_bundle.flight_model, d_lead)[()]) if np.isfinite(d_lead) else t_hat
    lead_x, lead_y, lead_used_vel, lead_clipped = trajectory_catch_point(
        passer=passer, receiver=receiver, flight_time_s=t_lead, trajectory=TRAJECTORY_LEAD
    )
    d_lead = path_length(passer["x"], passer["y"], lead_x, lead_y) if passer_ok else float("nan")

    direct_x, direct_y, _, _ = trajectory_catch_point(
        passer=passer, receiver=receiver, flight_time_s=t_hat, trajectory=TRAJECTORY_DIRECT
    )
    d_direct = path_length(passer["x"], passer["y"], direct_x, direct_y) if passer_ok else float("nan")
    t_direct = float(predict_flight_time(flight_bundle.flight_model, d_direct)[()]) if np.isfinite(d_direct) else t_hat

    reject_direct = bool(outside_distance_support(d_direct, flight_bundle.distance_support)) if np.isfinite(d_direct) else True
    reject_lead = bool(outside_distance_support(d_lead, flight_bundle.distance_support)) if np.isfinite(d_lead) else True

    # Assigned defender only when the matchup join matched; else missing.
    assigned = assigned_defender if matchup_matched and assigned_defender is not None else None

    direct_feats = _features_for_trajectory(
        passer=passer,
        receiver=receiver,
        defenders=defenders,
        assigned_defender=assigned,
        catch_x=direct_x,
        catch_y=direct_y,
        flight_time_s=t_direct,
        distance_support_reject=reject_direct or not tracked or not passer_ok,
        prefix="direct_",
    )
    lead_feats = _features_for_trajectory(
        passer=passer,
        receiver=receiver,
        defenders=defenders,
        assigned_defender=assigned,
        catch_x=lead_x,
        catch_y=lead_y,
        flight_time_s=t_lead,
        distance_support_reject=reject_lead or not tracked or not passer_ok,
        prefix="lead_",
    )

    choice = choose_trajectory_explicit(receiver_velocity_ok=bool(receiver["velocity_ok"]))
    primary_reject = reject_lead if choice == TRAJECTORY_LEAD else reject_direct
    primary_prefix = "lead_" if choice == TRAJECTORY_LEAD else "direct_"

    # Primary columns mirror the chosen trajectory (both families also retained).
    chosen = lead_feats if choice == TRAJECTORY_LEAD else direct_feats
    pfx = primary_prefix
    primary = {
        "trajectory_choice": choice,
        "trajectory_choice_rule": TRAJECTORY_RULE,
        "catch_x_event": chosen[f"{pfx}catch_x_event"],
        "catch_y_event": chosen[f"{pfx}catch_y_event"],
        "pass_distance_ft": chosen[f"{pfx}pass_distance_ft"],
        "pass_angle_deg": chosen[f"{pfx}pass_angle_deg"],
        "trajectory_length_ft": chosen[f"{pfx}trajectory_length_ft"],
        "flight_time_s": chosen[f"{pfx}flight_time_s"],
        "pass_speed_fps": chosen[f"{pfx}pass_speed_fps"],
        "min_lane_clearance_ft": chosen[f"{pfx}min_lane_clearance_ft"],
        "min_temporal_margin_s": chosen[f"{pfx}min_temporal_margin_s"],
        "min_temporal_margin_const35_s": chosen[f"{pfx}min_temporal_margin_const35_s"],
        "receiver_sep_ft": chosen[f"{pfx}receiver_sep_ft"],
        "passer_pressure_ft": chosen[f"{pfx}passer_pressure_ft"],
        "target_defender_recovery_s": chosen[f"{pfx}target_defender_recovery_s"],
        "help_density_12ft": chosen[f"{pfx}help_density_12ft"],
        "sideline_prox_ft": chosen[f"{pfx}sideline_prox_ft"],
        "baseline_prox_ft": chosen[f"{pfx}baseline_prox_ft"],
        "dist_to_rim_ft": chosen[f"{pfx}dist_to_rim_ft"],
        "angle_to_rim_deg": chosen[f"{pfx}angle_to_rim_deg"],
        "catch_x_norm": chosen[f"{pfx}catch_x_norm"],
        "catch_y_norm": chosen[f"{pfx}catch_y_norm"],
        "receiver_toward_rim_fps": chosen[f"{pfx}receiver_toward_rim_fps"],
        "geometry_passability_score": chosen[f"{pfx}geometry_passability_score"],
        "rejected_outside_support": bool(primary_reject or not tracked or not passer_ok),
        "intercepting_defenderId": chosen[f"{pfx}intercepting_defenderId"],
        "intercepting_defender_vx": chosen[f"{pfx}intercepting_defender_vx"],
        "intercepting_defender_vy": chosen[f"{pfx}intercepting_defender_vy"],
        "intercepting_defender_velocity_ok": chosen[f"{pfx}intercepting_defender_velocity_ok"],
        "court_region": chosen[f"{pfx}court_region"],
        "lane_predError_mean": chosen[f"{pfx}lane_predError_mean"],
    }

    out = {
        "passer_x_event": passer["x"],
        "passer_y_event": passer["y"],
        "passer_vx": passer["vx"] if passer["velocity_ok"] else float("nan"),
        "passer_vy": passer["vy"] if passer["velocity_ok"] else float("nan"),
        "passer_velocity_ok": bool(passer["velocity_ok"]),
        "passer_predError": passer["predError"],
        "receiver_x_event": receiver["x"],
        "receiver_y_event": receiver["y"],
        "receiver_vx": receiver["vx"] if receiver["velocity_ok"] else float("nan"),
        "receiver_vy": receiver["vy"] if receiver["velocity_ok"] else float("nan"),
        "receiver_velocity_ok": bool(receiver["velocity_ok"]),
        "receiver_predError": receiver["predError"],
        "receiver_tracked": bool(tracked),
        "matchup_matched": bool(matchup_matched),
        "assigned_defender_x_event": assigned["x"] if assigned else float("nan"),
        "assigned_defender_y_event": assigned["y"] if assigned else float("nan"),
        "assigned_defender_vx": assigned["vx"] if assigned and assigned["velocity_ok"] else float("nan"),
        "assigned_defender_vy": assigned["vy"] if assigned and assigned["velocity_ok"] else float("nan"),
        "assigned_defender_velocity_ok": bool(assigned["velocity_ok"]) if assigned else False,
        "lead_projection_used_velocity": bool(lead_used_vel),
        "lead_projection_clipped": bool(lead_clipped),
        "fold_id": flight_bundle.fold_id,
        "distance_support_max_ft": float(flight_bundle.distance_support.distance_max_ft),
        "distance_support_min_ft": float(flight_bundle.distance_support.distance_min_ft),
        "model_hz": FREEZE_MODEL_HZ,
        "frame_stride": FREEZE_FRAME_STRIDE,
        # Placeholders for Stage 4 statistical models (geometry-only retained above).
        "q_passability_model": float("nan"),
        "V_catch_model": float("nan"),
        "NOV_model": float("nan"),
        **direct_feats,
        **lead_feats,
        **primary,
        "_primary_prefix": primary_prefix,
    }
    return out


def _row_to_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "_asdict"):
        return row._asdict()
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def build_pass_release_model_frames(
    passes: pd.DataFrame,
    touches: pd.DataFrame,
) -> pd.DataFrame:
    """One model-frame row per pass release that falls inside its touch interval."""
    if passes.empty or touches.empty:
        return pd.DataFrame()
    p = passes.copy()
    if "id" in p.columns and "passId" not in p.columns:
        p = p.rename(columns={"id": "passId"})
    t = touches.copy()
    touch_id_col = "id" if "id" in t.columns else "touchId"
    t = t.rename(columns={touch_id_col: "touchId"})
    t = t[
        [
            c
            for c in [
                "touchId",
                "gameId",
                "chanceId",
                "possessionId",
                "startFrame",
                "endFrame",
                "playerId",
                "primary_touch_eligible",
            ]
            if c in t.columns
        ]
    ].rename(columns={"startFrame": "touch_startFrame", "endFrame": "touch_endFrame", "playerId": "ballhandlerId"})

    cols = [
        c
        for c in [
            "passId",
            "gameId",
            "touchId",
            "chanceId",
            "possessionId",
            "startFrame",
            "period",
            "startGameClock",
            "shotClock",
            "passerId",
        ]
        if c in p.columns
    ]
    rel = p[cols].dropna(subset=["touchId", "startFrame"]).copy()
    rel["frameIdx"] = rel["startFrame"].astype(int)
    if "startGameClock" in rel.columns:
        rel = rel.rename(columns={"startGameClock": "gameClock"})
    if "passerId" in rel.columns and "ballhandlerId" not in rel.columns:
        rel["ballhandlerId"] = rel["passerId"]

    merged = rel.merge(t, on=["gameId", "touchId"], how="inner", suffixes=("", "_touch"))
    if "chanceId_touch" in merged.columns:
        merged["chanceId"] = merged["chanceId"].where(merged["chanceId"].notna(), merged["chanceId_touch"])
    if "possessionId_touch" in merged.columns and "possessionId" in merged.columns:
        merged["possessionId"] = merged["possessionId"].where(
            merged["possessionId"].notna(), merged["possessionId_touch"]
        )
    if "ballhandlerId_touch" in merged.columns:
        merged["ballhandlerId"] = merged["ballhandlerId"].where(
            merged["ballhandlerId"].notna(), merged["ballhandlerId_touch"]
        )
    in_touch = (merged["frameIdx"] >= merged["touch_startFrame"]) & (
        merged["frameIdx"] <= merged["touch_endFrame"]
    )
    merged = merged.loc[in_touch].copy()
    # Release rows are kept for labeling even when the touch fails primary filters.
    merged["primary_frame_eligible"] = True
    keep = [
        c
        for c in [
            "gameId",
            "touchId",
            "chanceId",
            "possessionId",
            "frameIdx",
            "period",
            "gameClock",
            "shotClock",
            "ballhandlerId",
            "primary_frame_eligible",
            "passId",
        ]
        if c in merged.columns
    ]
    out = merged[keep].drop_duplicates(subset=["gameId", "touchId", "frameIdx"], keep="first")
    return out.reset_index(drop=True)


def build_forced_release_candidate_states(
    *,
    passes: pd.DataFrame,
    touch_candidates: pd.DataFrame,
    matchups: pd.DataFrame,
    tracking_players: pd.DataFrame,
    touches: pd.DataFrame,
) -> pd.DataFrame:
    """Synthesize Stage-1-shaped candidate states at every pass release frame.

    Stage 1 ``touch_candidates`` only covers primary-eligible touches. Completion
    labels also live on non-primary touches, so we expand candidates via
    ``build_touch_candidates(..., strict=False)`` for any release touch missing
    from the Stage 1 table.
    """
    from .joins import build_touch_candidates

    model_frames = build_pass_release_model_frames(passes, touches)
    if model_frames.empty:
        return pd.DataFrame()

    needed_touches = set(model_frames["touchId"].unique())
    existing = set(touch_candidates["touchId"].unique()) if not touch_candidates.empty else set()
    missing = needed_touches - existing
    cand = touch_candidates
    if missing:
        t = touches.copy()
        tid = "id" if "id" in t.columns else "touchId"
        t = t.rename(columns={tid: "touchId"}) if tid != "touchId" else t
        # build_touch_candidates expects column ``id``.
        extra_touches = t[t["touchId"].isin(missing)].copy()
        if "id" not in extra_touches.columns:
            extra_touches = extra_touches.rename(columns={"touchId": "id"})
        try:
            extra_cand = build_touch_candidates(extra_touches, strict=False)
        except Exception:
            extra_cand = pd.DataFrame()
        if not extra_cand.empty:
            cand = pd.concat([cand, extra_cand], ignore_index=True) if not cand.empty else extra_cand

    if cand.empty:
        return pd.DataFrame()

    # Only keep release frames whose touch now has exactly 4 candidates.
    ok_touches = (
        cand.groupby("touchId").size().loc[lambda s: s == 4].index
        if not cand.empty
        else []
    )
    model_frames = model_frames[model_frames["touchId"].isin(ok_touches)].copy()
    if model_frames.empty:
        return pd.DataFrame()

    states = build_candidate_states(
        model_frames.drop(columns=["passId"], errors="ignore"),
        cand,
        matchups,
        players=tracking_players,
    )
    if states.empty:
        return states
    states = states.copy()
    states["is_model_5hz_frame"] = False
    states["is_forced_release_frame"] = True
    states["sampling_role"] = SAMPLING_PASS_RELEASE
    return states


def merge_decision_states(
    primary_5hz: pd.DataFrame,
    release_states: pd.DataFrame,
) -> pd.DataFrame:
    """Union 5 Hz primary grid with forced release frames; dual-flag overlaps."""
    if primary_5hz.empty and release_states.empty:
        return pd.DataFrame()
    hz = primary_5hz.copy()
    if not hz.empty:
        hz["is_model_5hz_frame"] = True
        hz["is_forced_release_frame"] = False
        hz["sampling_role"] = SAMPLING_MODEL_5HZ
    if release_states.empty:
        return hz
    rel = release_states.copy()
    if hz.empty:
        return rel

    hz_keys = set(zip(hz["touchId"].tolist(), hz["frameIdx"].astype(int).tolist()))
    rel_keys = list(zip(rel["touchId"].tolist(), rel["frameIdx"].astype(int).tolist()))
    overlap_mask = [(t, int(f)) in hz_keys for t, f in rel_keys]
    # Mark overlapping 5 Hz rows as also release frames.
    if any(overlap_mask):
        overlap_set = {(t, int(f)) for (t, f), ov in zip(rel_keys, overlap_mask) if ov}
        key_series = list(zip(hz["touchId"].tolist(), hz["frameIdx"].astype(int).tolist()))
        mark = [(t, int(f)) in overlap_set for t, f in key_series]
        hz.loc[mark, "is_forced_release_frame"] = True
        hz.loc[mark, "sampling_role"] = "model_5hz+pass_release"
    # Append release-only frames.
    only_rel = rel.loc[[not ov for ov in overlap_mask]].copy()
    if only_rel.empty:
        return hz.reset_index(drop=True)
    return pd.concat([hz, only_rel], ignore_index=True)


def reconstruct_game_candidates(
    *,
    candidate_states: pd.DataFrame,
    touch_model_frames: pd.DataFrame,
    tracking_players: pd.DataFrame,
    chances: pd.DataFrame,
    touches: pd.DataFrame,
    flight_bundle: FoldFlightBundle,
    primary_eligible_only: bool = True,
    passes: pd.DataFrame | None = None,
    touch_candidates: pd.DataFrame | None = None,
    matchups: pd.DataFrame | None = None,
    force_pass_releases: bool = True,
) -> pd.DataFrame:
    """Build Stage 3 feature rows for one game from Stage 1 tables.

    Dual sampling:
    - ``model_5hz``: primary-eligible stride-5 frames (window time series)
    - ``pass_release``: forced pass ``startFrame`` rows for Stage 4 labels
    """
    if candidate_states.empty and not force_pass_releases:
        return pd.DataFrame()

    hz = candidate_states
    if primary_eligible_only and not hz.empty and "primary_frame_eligible" in hz.columns:
        hz = hz[hz["primary_frame_eligible"].astype(bool)].copy()

    release_states = pd.DataFrame()
    if force_pass_releases and passes is not None and touch_candidates is not None and matchups is not None:
        release_states = build_forced_release_candidate_states(
            passes=passes,
            touch_candidates=touch_candidates,
            matchups=matchups,
            tracking_players=tracking_players,
            touches=touches,
        )

    states = merge_decision_states(hz, release_states)
    if states.empty:
        return pd.DataFrame()

    # Ball-handler velocities from tracking (touch_model_frames has XY only).
    tplayers = tracking_players.copy()
    needed_cols = [
        "frameIdx",
        "playerId",
        "x_event",
        "y_event",
        "vx",
        "vy",
        "velocity_ok",
        "predError",
        "isDetected",
    ]
    tplayers = tplayers[[c for c in needed_cols if c in tplayers.columns]]

    chance_lookup = chances.set_index("id") if "id" in chances.columns else chances.set_index("chanceId")
    touch_lookup = touches.set_index("id") if "id" in touches.columns else touches.set_index("touchId")

    # Index players by frame for O(1) frame slices.
    players_by_frame: dict[int, pd.DataFrame] = {
        int(fi): g for fi, g in tplayers.groupby("frameIdx", sort=False)
    }

    # Merge passer XY from touch_model_frames when available (5 Hz rows).
    tm_cols = [
        c
        for c in ["touchId", "frameIdx", "ballhandler_x", "ballhandler_y", "shotClock", "gameClock"]
        if c in touch_model_frames.columns
    ]
    if tm_cols and not touch_model_frames.empty:
        tm = touch_model_frames[tm_cols].drop_duplicates(subset=["touchId", "frameIdx"])
        states = states.merge(tm, on=["touchId", "frameIdx"], how="left", suffixes=("", "_tm"))

    rows: list[dict[str, Any]] = []
    # Group by frame to reuse defender lookups.
    for (touch_id, frame_idx), group in states.groupby(["touchId", "frameIdx"], sort=False):
        fi = int(frame_idx)
        frame_players = players_by_frame.get(fi)
        if frame_players is None or frame_players.empty:
            continue

        first = group.iloc[0]
        bh_id = int(first["ballhandlerId"]) if pd.notna(first["ballhandlerId"]) else None
        chance_id = first["chanceId"]

        def_ids: list[int] = []
        if chance_id in chance_lookup.index:
            ch = chance_lookup.loc[chance_id]
            if isinstance(ch, pd.DataFrame):
                ch = ch.iloc[0]
            raw = ch["defPlayerIds"] if "defPlayerIds" in ch.index else []
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                def_ids = [int(x) for x in raw if x is not None and not (isinstance(x, float) and np.isnan(x))]

        defenders = []
        by_id = {int(r.playerId): _row_to_dict(r) for r in frame_players.itertuples(index=False)}
        for pid in def_ids:
            defenders.append(_player_state(by_id.get(pid, {"playerId": pid})))

        passer_row = by_id.get(bh_id) if bh_id is not None else None
        passer = _player_state(passer_row)
        if passer_row is None and pd.notna(first.get("ballhandler_x")):
            passer = _player_state(
                {
                    "playerId": bh_id,
                    "x_event": first.get("ballhandler_x"),
                    "y_event": first.get("ballhandler_y"),
                    "vx": np.nan,
                    "vy": np.nan,
                    "velocity_ok": False,
                    "predError": np.nan,
                }
            )

        touch_start = None
        if touch_id in touch_lookup.index:
            trow = touch_lookup.loc[touch_id]
            if isinstance(trow, pd.DataFrame):
                trow = trow.iloc[0]
            touch_start = int(trow["startFrame"]) if pd.notna(trow.get("startFrame")) else None

        touch_age_s = (
            float((fi - touch_start) / FRAME_RATE_HZ) if touch_start is not None else float("nan")
        )

        is_5hz = bool(first.get("is_model_5hz_frame", False))
        is_release = bool(first.get("is_forced_release_frame", False))
        sampling_role = first.get("sampling_role", SAMPLING_MODEL_5HZ if is_5hz else SAMPLING_PASS_RELEASE)

        for crow in group.itertuples(index=False):
            cand_id = int(crow.candidateId)
            receiver = _player_state(by_id.get(cand_id, {"playerId": cand_id}))
            if not np.isfinite(receiver["x"]) and pd.notna(getattr(crow, "candidate_x_event", np.nan)):
                receiver = _player_state(
                    {
                        "playerId": cand_id,
                        "x_event": crow.candidate_x_event,
                        "y_event": crow.candidate_y_event,
                        "vx": crow.candidate_vx,
                        "vy": crow.candidate_vy,
                        "velocity_ok": crow.candidate_velocity_ok,
                        "predError": crow.candidate_predError,
                    }
                )

            matchup_matched = bool(getattr(crow, "matchup_matched", False))
            assigned = None
            aid = getattr(crow, "assigned_defenderId", np.nan)
            if matchup_matched and pd.notna(aid):
                assigned = _player_state(by_id.get(int(aid), {"playerId": int(aid)}))

            feats = reconstruct_one_candidate(
                passer=passer,
                receiver=receiver,
                defenders=defenders,
                assigned_defender=assigned,
                matchup_matched=matchup_matched,
                flight_bundle=flight_bundle,
            )
            feats.update(
                {
                    "gameId": int(first["gameId"]),
                    "touchId": touch_id,
                    "chanceId": chance_id,
                    "possessionId": first.get("possessionId"),
                    "frameIdx": fi,
                    "period": first.get("period"),
                    "gameClock": first.get("gameClock"),
                    "shotClock": first.get("shotClock"),
                    "ballhandlerId": bh_id,
                    "candidateId": cand_id,
                    "candidate_rank": int(crow.candidate_rank),
                    "assigned_defenderId": int(aid) if pd.notna(aid) else pd.NA,
                    "matchupId": getattr(crow, "matchupId", pd.NA),
                    "primary_frame_eligible": bool(getattr(crow, "primary_frame_eligible", False)),
                    "is_model_5hz_frame": is_5hz,
                    "is_forced_release_frame": is_release,
                    "sampling_role": sampling_role,
                    "touch_age_s": touch_age_s,
                    "n_defenders_resolved": int(sum(1 for d in defenders if np.isfinite(d["x"]))),
                }
            )
            feats.pop("_primary_prefix", None)
            rows.append(feats)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def attach_narrow_path_labels(
    candidates: pd.DataFrame,
    eligibility: pd.DataFrame,
    passes: pd.DataFrame,
) -> pd.DataFrame:
    """Join Stage 2 label gates and mark completion-training eligibility.

    Completion labels are attached only for release-time rows whose pass is in
    ``receiver_completion_training_labels`` (completed known receivers). Failed
    passes remain touch-level only: we never promote inferred targets as named
    receivers here.
    """
    assert_gate_columns_present(eligibility, name="stage2_label_eligibility")
    out = candidates.copy()
    empty_gate_defaults = (
        "passId",
        "is_pass_release_frame",
        "completion_label_eligible",
        "true_receiverId",
        "is_true_receiver",
        "pass_outcome_class",
        "interceptorId",
        "cohort",
        "label_gate_reason",
        *GATE_COLUMNS,
    )
    if out.empty:
        for col in empty_gate_defaults:
            out[col] = pd.Series(dtype=object)
        return out

    labels = receiver_completion_training_labels(eligibility)
    # Map touch → pass release for primary pass on that touch.
    p = passes.copy()
    if "id" in p.columns and "passId" not in p.columns:
        p = p.rename(columns={"id": "passId"})
    release = p[
        [
            c
            for c in [
                "passId",
                "gameId",
                "touchId",
                "startFrame",
                "passerId",
                "receiverId",
                "toReceiverId",
                "pass_outcome_class",
                "complete",
            ]
            if c in p.columns
        ]
    ].copy()
    release = release.rename(columns={"startFrame": "frameIdx", "toReceiverId": "interceptorId"})

    # One pass per (touch, release frame); multi-pass touches are rare and audited.
    release = release.drop_duplicates(subset=["gameId", "touchId", "frameIdx"], keep="first")

    out = out.merge(
        release,
        on=["gameId", "touchId", "frameIdx"],
        how="left",
        suffixes=("", "_pass"),
    )
    out["is_pass_release_frame"] = out["passId"].notna()
    # Align forced-release flag with successful pass join when present.
    if "is_forced_release_frame" in out.columns:
        out["is_forced_release_frame"] = (
            out["is_forced_release_frame"].fillna(False).astype(bool) | out["is_pass_release_frame"]
        )

    gate_cols = [
        "passId",
        "true_receiverId",
        "interceptorId",
        "pass_outcome_class",
        "cohort",
        *GATE_COLUMNS,
    ]
    # Preserve order, drop accidental duplicates (e.g. label_gate_reason in GATE_COLUMNS).
    seen: set[str] = set()
    gate_cols_unique: list[str] = []
    for c in gate_cols:
        if c not in seen and c in eligibility.columns:
            seen.add(c)
            gate_cols_unique.append(c)
    gate = eligibility[gate_cols_unique].drop_duplicates("passId")
    out = out.merge(gate, on="passId", how="left", suffixes=("", "_gate"))

    # Prefer eligibility interceptor / outcome when present.
    if "interceptorId_gate" in out.columns:
        out["interceptorId"] = out["interceptorId_gate"].where(out["interceptorId_gate"].notna(), out.get("interceptorId"))
        out = out.drop(columns=["interceptorId_gate"])
    if "pass_outcome_class_gate" in out.columns:
        out["pass_outcome_class"] = out["pass_outcome_class_gate"].where(
            out["pass_outcome_class_gate"].notna(), out.get("pass_outcome_class")
        )
        out = out.drop(columns=["pass_outcome_class_gate"])

    # Ensure every GATE_COLUMNS name exists even if eligibility slice missed one.
    for col in GATE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    # Completion-training label: release frame + receiver_specific_eligible.
    # is_true_receiver marks the known completed receiver among those four.
    eligible_pass_ids = set(labels["passId"].tolist()) if not labels.empty else set()
    recv_elig = out["receiver_specific_eligible"]
    if recv_elig.dtype == object:
        recv_elig = recv_elig.where(recv_elig.notna(), False).astype(bool)
    else:
        recv_elig = recv_elig.fillna(False).astype(bool)
    out["completion_label_eligible"] = (
        out["is_pass_release_frame"].fillna(False).astype(bool)
        & out["passId"].isin(eligible_pass_ids)
        & recv_elig
    )
    true_rx = out["true_receiverId"]
    if true_rx.isna().all() and "receiverId" in out.columns:
        true_rx = out["receiverId"]
        out["true_receiverId"] = true_rx
    out["is_true_receiver"] = (
        out["completion_label_eligible"]
        & out["candidateId"].notna()
        & true_rx.notna()
        & (out["candidateId"].astype("Int64") == true_rx.astype("Int64"))
    )

    # Fill gate defaults for non-release rows.
    def _bool_fill(series: pd.Series, default: bool) -> pd.Series:
        if series.dtype == object:
            return series.where(series.notna(), default).astype(bool)
        return series.fillna(default).astype(bool)

    out["narrow_path_active"] = _bool_fill(out["narrow_path_active"], True)
    out["target_inference_mode"] = out["target_inference_mode"].where(
        out["target_inference_mode"].notna(), "narrow_path"
    )
    out["human_audit_status"] = out["human_audit_status"].where(
        out["human_audit_status"].notna(), "deferred_waived"
    )
    out["receiver_specific_eligible"] = _bool_fill(out["receiver_specific_eligible"], False)
    out["usable_as_receiver_completion_label"] = _bool_fill(
        out["usable_as_receiver_completion_label"], False
    )
    out["usable_as_named_failed_target"] = _bool_fill(out["usable_as_named_failed_target"], False)
    out["named_failed_target_requires_sensitivity_check"] = _bool_fill(
        out["named_failed_target_requires_sensitivity_check"], False
    )
    out["usable_for_touch_level_turnover"] = _bool_fill(
        out["usable_for_touch_level_turnover"], False
    )
    if "target_reliability_status" not in out.columns:
        out["target_reliability_status"] = pd.NA
    out["target_reliability_status"] = out["target_reliability_status"].where(
        out["target_reliability_status"].notna(), pd.NA
    )

    # Guard: never treat interceptor as target / true receiver on feature rows.
    assert_interceptor_never_target(out, name="stage3_candidates", target_col="true_receiverId")
    if "candidateId" in out.columns and "interceptorId" in out.columns:
        clash = (
            out["candidateId"].notna()
            & out["interceptorId"].notna()
            & (out["candidateId"].astype("Int64") == out["interceptorId"].astype("Int64"))
        )
        if int(clash.sum()):
            raise AssertionError(
                f"stage3_candidates: {int(clash.sum())} row(s) have candidateId == interceptorId"
            )

    # Assert completion labels never come from failed cohort.
    if out["completion_label_eligible"].any():
        bad = out["completion_label_eligible"] & (out.get("cohort") == "failed")
        if "cohort" in out.columns and int(bad.fillna(False).sum()):
            raise AssertionError("completion_label_eligible rows drawn from failed cohort")

    # Parquet forbids duplicate column names (can arise from merge key collisions).
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()

    return out


def summarize_candidates(df: pd.DataFrame) -> dict[str, Any]:
    """Self-QC metrics for the Stage 3 candidate table."""
    if df.empty:
        return {"n_candidate_rows": 0}
    n = len(df)
    return {
        "n_candidate_rows": n,
        "n_games": int(df["gameId"].nunique()) if "gameId" in df.columns else 0,
        "n_touches": int(df["touchId"].nunique()) if "touchId" in df.columns else 0,
        "n_frames": int(df.groupby(["touchId", "frameIdx"]).ngroups) if "frameIdx" in df.columns else 0,
        "candidates_per_frame_mode": int(df.groupby(["touchId", "frameIdx"]).size().mode().iloc[0])
        if n
        else 0,
        "pct_rejected_outside_support": float(100.0 * df["rejected_outside_support"].mean())
        if "rejected_outside_support" in df.columns
        else float("nan"),
        "pct_missing_matchup": float(100.0 * (~df["matchup_matched"].astype(bool)).mean())
        if "matchup_matched" in df.columns
        else float("nan"),
        "pct_receiver_velocity_missing": float(100.0 * (~df["receiver_velocity_ok"].astype(bool)).mean())
        if "receiver_velocity_ok" in df.columns
        else float("nan"),
        "trajectory_mix": {
            str(k): int(v) for k, v in df["trajectory_choice"].value_counts().items()
        }
        if "trajectory_choice" in df.columns
        else {},
        "n_completion_label_eligible_rows": int(df["completion_label_eligible"].sum())
        if "completion_label_eligible" in df.columns
        else 0,
        "n_true_receiver_rows": int(df["is_true_receiver"].sum()) if "is_true_receiver" in df.columns else 0,
        "n_forced_release_frames": (
            int(df.loc[df["is_forced_release_frame"].astype(bool)].groupby(["touchId", "frameIdx"]).ngroups)
            if "is_forced_release_frame" in df.columns
            else 0
        ),
        "n_model_5hz_frames": (
            int(df.loc[df["is_model_5hz_frame"].astype(bool)].groupby(["touchId", "frameIdx"]).ngroups)
            if "is_model_5hz_frame" in df.columns
            else 0
        ),
        "court_region_mix": {
            str(k): int(v) for k, v in df["court_region"].value_counts().items()
        }
        if "court_region" in df.columns
        else {},
    }
