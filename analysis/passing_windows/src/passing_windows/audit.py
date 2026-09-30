"""Join validation and calculation double-checks for Stage 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .coords import distance_xy, mirror_xy, normalize_attack_xy, symmetry_feature_check
from .geometry import euclidean_distance, interception_margin, lane_clearance, point_to_segment_distance
from .io_utils import verify_output_manifest
from .joins import EXPECTED_CANDIDATES_PER_TOUCH, ballhandler_in_offense, clock_agreement
from .velocities import (
    MAX_CAUSAL_FRAME_GAP,
    MAX_PLAUSIBLE_PLAYER_SPEED_FPS,
    assert_no_future_leakage,
    causal_velocity_from_positions,
    orientation_segment_ids,
)


def audit_event_tracking_clocks(
    events: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    event_frame_col: str = "startFrame",
    event_clock_col: str = "startGameClock",
    event_wall_col: str = "startWallClock",
    sample_n: int = 200,
    seed: int = 20250924,
) -> dict[str, Any]:
    """Spot-check event clocks vs tracking at joined (gameId, frameIdx)."""
    if events.empty or frames.empty:
        return {"n": 0, "pass_rate": np.nan, "ok": False}
    if "gameId" not in events.columns or "gameId" not in frames.columns:
        return {"n": 0, "pass_rate": np.nan, "ok": False, "error": "missing gameId"}
    fr = frames.set_index(["gameId", "frameIdx"], drop=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(events), size=min(sample_n, len(events)), replace=False)
    sample = events.iloc[idx]
    results = []
    for row in sample.itertuples(index=False):
        ef = int(getattr(row, event_frame_col))
        key = (int(row.gameId), ef)
        if key not in fr.index:
            results.append({"ok": False, "reason": "missing_frame"})
            continue
        tr = fr.loc[key]
        if isinstance(tr, pd.DataFrame):
            tr = tr.iloc[0]
        ca = clock_agreement(
            ef,
            int(tr["frameIdx"]),
            getattr(row, event_clock_col, None),
            tr.get("gameClock"),
            getattr(row, event_wall_col, None),
            tr.get("wallClock"),
        )
        results.append(ca)
    ok = sum(1 for r in results if r.get("ok"))
    return {
        "n": len(results),
        "n_ok": ok,
        "pass_rate": ok / max(len(results), 1),
        "ok": (ok / max(len(results), 1)) >= 0.99,
        "missing_frames": sum(1 for r in results if r.get("reason") == "missing_frame"),
    }


def audit_pass_release_vs_touch_end(passes_joined: pd.DataFrame) -> dict[str, Any]:
    """Ordinary passes: release at/near touch end."""
    p = passes_joined[
        passes_joined["touchId"].notna()
        & passes_joined.get("is_self_pass", False) != True  # noqa: E712
    ].copy()
    if "is_self_pass" in passes_joined.columns:
        p = passes_joined[
            passes_joined["touchId"].notna() & ~passes_joined["is_self_pass"].astype(bool)
        ].copy()
    # Ordinary: complete or incomplete but not inbound
    if "inbounds" in p.columns:
        p = p[~p["inbounds"].astype(bool)]
    if p.empty:
        return {"n": 0, "frac_near_end": np.nan, "ok": False}
    near = p["release_near_touch_end"].astype(bool)
    within = p["pass_within_touch"].astype(bool)
    return {
        "n": len(p),
        "frac_within_touch": float(within.mean()),
        "frac_near_end": float(near.mean()),
        "median_touch_end_minus_release": float((p["touch_endFrame"] - p["startFrame"]).median()),
        "ok": float(within.mean()) >= 0.95,
    }


def audit_receiver_loc_vs_tracking(
    passes: pd.DataFrame,
    players: pd.DataFrame,
    *,
    max_dist_ft: float = 3.0,
) -> dict[str, Any]:
    """Completed-pass receiverLoc at release vs tracking position."""
    completed = passes[
        passes["complete"].astype(bool)
        & passes["receiverId"].notna()
        & passes["receiverLoc"].notna()
    ].copy()
    if completed.empty or players.empty:
        return {"n": 0, "frac_agree": np.nan, "ok": False}

    pl = players.sort_values(["gameId", "frameIdx", "playerId"]).set_index(
        ["gameId", "frameIdx", "playerId"]
    )
    dists = []
    for row in completed.itertuples(index=False):
        key = (int(row.gameId), int(row.startFrame), int(row.receiverId))
        if key not in pl.index:
            dists.append(np.nan)
            continue
        tr = pl.loc[key]
        if isinstance(tr, pd.DataFrame):
            tr = tr.iloc[0]
        rl = row.receiverLoc
        if hasattr(rl, "tolist"):
            rl = rl.tolist()
        # Compare against event-aligned tracking when available
        if "x_event" in tr.index and pd.notna(tr.get("x_event")):
            tx, ty = float(tr["x_event"]), float(tr["y_event"])
        else:
            tx, ty = float(tr["x"]), float(tr["y"])
        dists.append(distance_xy((float(rl[0]), float(rl[1])), (tx, ty)))
    arr = np.asarray(dists, dtype=float)
    valid = ~np.isnan(arr)
    agree = (arr[valid] <= max_dist_ft) if valid.any() else np.array([])
    return {
        "n": int(valid.sum()),
        "n_missing_tracking": int((~valid).sum()),
        "median_dist_ft": float(np.nanmedian(arr)),
        "frac_within_3ft": float(agree.mean()) if len(agree) else np.nan,
        "ok": (float(agree.mean()) >= 0.90) if len(agree) else False,
    }


def audit_mirroring_symmetry() -> dict[str, Any]:
    """Unit-level geometry symmetry: mirror flips signed y features, not distances."""
    cases = []
    for passer, receiver, defender in [
        ((-20.0, 5.0), (-10.0, -8.0), (-15.0, 0.0)),
        ((-30.0, 12.0), (-5.0, 3.0), (-18.0, 6.0)),
    ]:
        d = euclidean_distance(*passer, *receiver)
        pm, rm = mirror_xy(*passer), mirror_xy(*receiver)
        dm = euclidean_distance(*pm, *rm)
        y_signed = passer[1] - receiver[1]
        y_signed_m = pm[1] - rm[1]
        clear = lane_clearance(passer, receiver, defender)
        clear_m = lane_clearance(pm, rm, mirror_xy(*defender))
        cases.append(
            {
                "distance_symmetric": symmetry_feature_check(d, dm, expect_flip=False, tol=1e-9),
                "y_delta_flips": symmetry_feature_check(y_signed, y_signed_m, expect_flip=True),
                "clearance_symmetric": symmetry_feature_check(clear, clear_m, expect_flip=False, tol=1e-9),
            }
        )
    ok = all(all(c.values()) for c in cases)
    return {"n_cases": len(cases), "cases": cases, "ok": ok}


def audit_causal_velocity() -> dict[str, Any]:
    """Synthetic series: velocity uses past only; future change must not affect v[t]."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    y = np.zeros(5)
    frames = np.arange(5)
    vx, vy, speed = causal_velocity_from_positions(x, y, frames)
    # At t=3, v should be based on frames 2->3 only (=1*25 fps = 25)
    v3 = vx[3]
    # Mutate future
    x2 = x.copy()
    x2[4] = 100.0
    vx2, _, _ = causal_velocity_from_positions(x2, y, frames)
    assert_no_future_leakage(frames[:4], prediction_time=3)
    leaked = False
    try:
        assert_no_future_leakage(frames, prediction_time=3)
        leaked = True  # should have raised
    except AssertionError:
        leaked = False
    # Segment reset: a coordinate discontinuity at a possession/orientation
    # boundary must yield NaN, not an enormous velocity.
    seg_x = np.array([-40.0, -39.0, 39.0, 40.0])
    seg_frames = np.array([0, 1, 2, 3])
    seg_ids = np.array([0, 0, 1, 1])
    _, _, seg_speed = causal_velocity_from_positions(
        seg_x, np.zeros(4), seg_frames, segment_id=seg_ids
    )
    _, _, no_seg_speed = causal_velocity_from_positions(seg_x, np.zeros(4), seg_frames)
    gap_x = np.array([0.0, 1.0, 400.0])
    _, _, gap_speed = causal_velocity_from_positions(
        gap_x, np.zeros(3), np.array([0, 1, 500]), max_frame_gap=MAX_CAUSAL_FRAME_GAP
    )

    checks = {
        "future_mutate_invariant": abs(float(v3) - float(vx2[3])) < 1e-12,
        "leakage_detector_works": not leaked,
        "segment_boundary_resets": bool(np.isnan(seg_speed[2])),
        "unsegmented_boundary_is_impossible": float(no_seg_speed[2]) > MAX_PLAUSIBLE_PLAYER_SPEED_FPS,
        "wide_gap_resets": bool(np.isnan(gap_speed[2])),
    }
    return {
        "v_at_3": float(v3),
        "v_at_3_after_future_mutate": float(vx2[3]),
        "speed_at_unsegmented_boundary": float(no_seg_speed[2]),
        "checks": checks,
        **checks,
        "ok": all(checks.values()),
    }


def audit_causal_velocity_boundaries(
    players: pd.DataFrame,
    *,
    max_speed_fps: float = MAX_PLAUSIBLE_PLAYER_SPEED_FPS,
) -> dict[str, Any]:
    """Real-data check that no derivative crosses an orientation/gap boundary.

    Differencing event-aligned coordinates across an attacking-hoop change used
    to produce speeds up to ~1169 ft/s. Velocities are now taken on continuous
    broadcast coordinates with resets at orientation changes and wide frame
    gaps, so every retained speed must be physically plausible.
    """
    if players.empty or "speed_causal" not in players.columns:
        return {"n": 0, "ok": False, "error": "no velocity columns"}

    s = players["speed_causal"]
    finite = s[np.isfinite(s)]
    n_over = int((finite > max_speed_fps).sum())
    checks = {
        "no_speed_over_cap": n_over == 0,
        "no_implausible_flagged_retained": (
            int((players.get("speed_implausible", pd.Series(dtype=bool)).astype(bool) & s.notna()).sum()) == 0
            if "speed_implausible" in players.columns
            else True
        ),
    }

    # Orientation boundaries must carry no velocity at all.
    boundary_speeds = np.nan
    n_boundary_with_speed = None
    if {"gameId", "playerId", "frameIdx", "leftHoop"}.issubset(players.columns):
        pl = players.sort_values(["gameId", "playerId", "frameIdx"])
        seg = pl.groupby(["gameId", "playerId"], sort=False)["leftHoop"].transform(
            lambda col: orientation_segment_ids(col.tolist())
        )
        first_of_segment = seg.groupby([pl["gameId"], pl["playerId"], seg], sort=False).cumcount() == 0
        n_boundary_with_speed = int(pl.loc[first_of_segment, "speed_causal"].notna().sum())
        checks["no_velocity_at_orientation_boundary"] = n_boundary_with_speed == 0
        boundary_speeds = float(pl.loc[first_of_segment, "speed_causal"].max(skipna=True))

    gap_violation = None
    if "velocity_frame_gap" in players.columns:
        gap_violation = int(
            (
                (players["velocity_frame_gap"] > MAX_CAUSAL_FRAME_GAP)
                & players["speed_causal"].notna()
            ).sum()
        )
        checks["no_velocity_across_wide_gap"] = gap_violation == 0

    return {
        "n_player_rows": len(players),
        "n_speed_finite": int(len(finite)),
        "max_speed_fps": float(finite.max()) if len(finite) else np.nan,
        "p99_9_speed_fps": float(finite.quantile(0.999)) if len(finite) else np.nan,
        "n_speed_over_cap": n_over,
        "speed_cap_fps": max_speed_fps,
        "max_frame_gap_frames": MAX_CAUSAL_FRAME_GAP,
        "n_rows_at_orientation_boundary_with_speed": n_boundary_with_speed,
        "max_speed_at_orientation_boundary": boundary_speeds,
        "n_velocity_across_wide_gap": gap_violation,
        "n_speed_null": int(s.isna().sum()),
        "checks": checks,
        "ok": all(checks.values()),
    }


def audit_candidate_sets(
    candidates: pd.DataFrame,
    touches: pd.DataFrame,
) -> dict[str, Any]:
    """Every primary touch must contribute exactly four unique teammates."""
    primary = touches[touches["primary_touch_eligible"]]
    n_primary = len(primary)
    if candidates.empty:
        return {"n_primary_touches": n_primary, "ok": n_primary == 0}

    per_touch = candidates.groupby(["gameId", "touchId"])["candidateId"].agg(["size", "nunique"])
    counts = per_touch["size"].value_counts().to_dict()
    bad_count = per_touch[per_touch["size"] != EXPECTED_CANDIDATES_PER_TOUCH]
    dupes = per_touch[per_touch["size"] != per_touch["nunique"]]

    covered = set(zip(candidates["gameId"], candidates["touchId"], strict=True))
    primary_keys = set(zip(primary["gameId"], primary["id"], strict=True))
    uncovered = primary_keys - covered
    extra = covered - primary_keys

    # No candidate may be the ball-handler, and every candidate must sit in the
    # chance's offensive lineup.
    off_lookup = {
        (int(g), t): set(int(x) for x in (o if o is not None else []))
        for g, t, o in zip(primary["gameId"], primary["id"], primary["offPlayerIds"], strict=True)
    }
    n_self = int((candidates["candidateId"] == candidates["ballhandlerId"]).sum())
    n_outside_lineup = 0
    for g, t, c in zip(
        candidates["gameId"], candidates["touchId"], candidates["candidateId"], strict=True
    ):
        lineup = off_lookup.get((int(g), t))
        if lineup is not None and int(c) not in lineup:
            n_outside_lineup += 1

    bh_outside = int(
        sum(
            not ballhandler_in_offense(o, p)
            for o, p in zip(primary["offPlayerIds"], primary["playerId"], strict=True)
        )
    )

    checks = {
        "all_touches_exactly_4": len(bad_count) == 0,
        "no_duplicate_candidates": len(dupes) == 0,
        "every_primary_touch_covered": len(uncovered) == 0,
        "no_candidates_outside_primary": len(extra) == 0,
        "no_ballhandler_as_candidate": n_self == 0,
        "no_candidate_outside_chance_lineup": n_outside_lineup == 0,
        "no_primary_ballhandler_outside_offense": bh_outside == 0,
    }
    return {
        "n_primary_touches": n_primary,
        "n_candidate_rows": len(candidates),
        "expected_rows": n_primary * EXPECTED_CANDIDATES_PER_TOUCH,
        "candidates_per_touch_counts": {int(k): int(v) for k, v in counts.items()},
        "max_candidates_per_touch": int(per_touch["size"].max()),
        "min_candidates_per_touch": int(per_touch["size"].min()),
        "n_touches_wrong_count": len(bad_count),
        "n_touches_with_duplicates": len(dupes),
        "n_primary_touches_missing_candidates": len(uncovered),
        "n_ballhandler_as_candidate": n_self,
        "n_candidates_outside_lineup": n_outside_lineup,
        "checks": checks,
        "ok": all(checks.values()),
    }


def audit_matchup_interval_join(
    candidate_states: pd.DataFrame,
    matchups: pd.DataFrame,
    *,
    min_coverage: float = 0.90,
) -> dict[str, Any]:
    """Coverage, determinism, and boundary correctness of the matchup join."""
    if candidate_states.empty:
        return {"n": 0, "ok": False, "error": "no candidate states"}

    n = len(candidate_states)
    matched = candidate_states["matchup_matched"].astype(bool)
    ambiguous = candidate_states["matchup_ambiguous"].astype(bool)

    # Every chosen interval must actually contain the frame.
    m = candidate_states[matched]
    inside = (
        (m["frameIdx"] >= m["matchup_startFrame"]) & (m["frameIdx"] <= m["matchup_endFrame"])
    )
    # Keys must be preserved and unique.
    key_cols = ["gameId", "touchId", "frameIdx", "candidateId"]
    n_dupe_keys = int(candidate_states.duplicated(subset=key_cols).sum())

    # Assigned defender must never be the offensive player and must be a real
    # defender from the matchup feed.
    n_self_assign = int((m["assigned_defenderId"] == m["candidateId"]).sum())
    known_defenders = set(matchups["defPlayerId"].astype(int)) if len(matchups) else set()
    n_unknown_def = int(
        sum(int(d) not in known_defenders for d in m["assigned_defenderId"].dropna())
    )

    eligible = candidate_states
    if "primary_frame_eligible" in candidate_states.columns:
        eligible = candidate_states[candidate_states["primary_frame_eligible"].astype(bool)]
    elig_cov = float(eligible["matchup_matched"].astype(bool).mean()) if len(eligible) else np.nan

    checks = {
        "chosen_interval_contains_frame": bool(inside.all()) if len(m) else True,
        "keys_unique": n_dupe_keys == 0,
        "no_self_assignment": n_self_assign == 0,
        "defender_from_matchup_feed": n_unknown_def == 0,
        "coverage_above_min": (elig_cov >= min_coverage) if np.isfinite(elig_cov) else False,
    }
    return {
        "n_candidate_states": n,
        "n_matched": int(matched.sum()),
        "coverage_all_states": float(matched.mean()),
        "coverage_primary_eligible_states": elig_cov,
        "n_unmatched": int((~matched).sum()),
        "unmatched_rate": float((~matched).mean()),
        "n_ambiguous_overlapping_intervals": int(ambiguous.sum()),
        "ambiguous_rate": float(ambiguous.mean()),
        "max_intervals_covering_one_frame": int(candidate_states["n_matchup_intervals"].max()),
        "n_duplicate_keys": n_dupe_keys,
        "n_self_assignments": n_self_assign,
        "resolution_rule": "latest startFrame, then shortest interval, then smallest matchupId",
        "checks": checks,
        "ok": all(checks.values()),
    }


def audit_pass_outcome_classes(passes: pd.DataFrame) -> dict[str, Any]:
    """Document the tri-state `complete` field and failed-pass evidence gaps."""
    if passes.empty or "pass_outcome_class" not in passes.columns:
        return {"n": 0, "ok": False, "error": "missing pass_outcome_class"}
    rows = {}
    for cls, g in passes.groupby("pass_outcome_class"):
        rows[cls] = {
            "n": len(g),
            "n_with_endFrame": int(g["endFrame"].notna().sum()),
            "n_with_receiverId": int(g["receiverId"].notna().sum()),
            "n_with_receiverLoc": int(g["receiverLoc"].notna().sum()),
            "n_with_distance": int(g["distance"].notna().sum()),
            "n_with_interceptor": int(g["interceptorId"].notna().sum()),
            "n_turnover": int(g["turnover"].astype(bool).sum()) if "turnover" in g.columns else None,
        }
    failed = passes[passes["pass_outcome_class"] != "complete"]
    checks = {
        "no_receiverLoc_on_failed_passes": int(failed["receiverLoc"].notna().sum()) == 0,
        "unknown_outcome_has_no_endFrame": int(
            passes.loc[passes["pass_outcome_class"] == "unknown_outcome", "endFrame"].notna().sum()
        )
        == 0,
        "no_receiverId_on_failed_passes": int(failed["receiverId"].notna().sum()) == 0,
    }
    return {
        "by_class": rows,
        "n_failed_or_unknown": len(failed),
        "note": (
            "receiverLoc / distance / receiverRegion / receiverId are absent from the "
            "raw feed for every non-complete pass; Stage 1 drops nothing. Stage 2 must "
            "infer targets from ball trajectory, pass direction, and flight-time support."
        ),
        "checks": checks,
        "ok": all(checks.values()),
    }


def audit_output_manifest(manifest_path: Path, root: Path) -> dict[str, Any]:
    """Recompute every strictly verified manifest hash."""
    if not Path(manifest_path).exists():
        return {"ok": False, "error": "manifest missing"}
    return verify_output_manifest(Path(manifest_path), Path(root))


def audit_synthetic_lane_geometry() -> dict[str, Any]:
    """Synthetic lane cases for distance / clearance / margin primitives."""
    passer = (0.0, 0.0)
    receiver = (10.0, 0.0)
    mid = (5.0, 0.0)
    off = (5.0, 3.0)
    far = (5.0, 20.0)
    d_mid = point_to_segment_distance(*mid, *passer, *receiver)
    d_off = point_to_segment_distance(*off, *passer, *receiver)
    clear_mid = lane_clearance(passer, receiver, mid, defender_radius=1.0)
    clear_off = lane_clearance(passer, receiver, off, defender_radius=1.0)
    margin_on_lane = interception_margin(
        passer, receiver, mid, pass_speed_fps=20.0, defender_speed_fps=20.0, catch_radius=0.5
    )
    margin_far = interception_margin(
        passer, receiver, far, pass_speed_fps=20.0, defender_speed_fps=20.0, catch_radius=0.5
    )
    checks = {
        "midpoint_dist_zero": abs(d_mid) < 1e-9,
        "offset_dist_3": abs(d_off - 3.0) < 1e-9,
        "clearance_mid_negative": clear_mid < 0,
        "clearance_off_positive": clear_off > 0,
        "far_margin_gt_on_lane": margin_far > margin_on_lane,
    }
    return {
        "checks": checks,
        "ok": all(checks.values()),
        "values": {
            "d_mid": d_mid,
            "d_off": d_off,
            "clear_mid": clear_mid,
            "clear_off": clear_off,
            "margin_on_lane": margin_on_lane,
            "margin_far": margin_far,
        },
    }


def audit_points_reconstruction(
    shots: pd.DataFrame,
    free_throws: pd.DataFrame,
    chances: pd.DataFrame,
    points: pd.DataFrame,
) -> dict[str, Any]:
    """Double-check points_recon vs manual sum; never trust ptsScored for outcomes."""
    # Manual total across all games
    shot_pts = 0
    if len(shots):
        shot_pts = int(
            np.where(
                shots["outcome"].astype(bool),
                np.where(shots["three"].astype(bool), 3, 2),
                0,
            ).sum()
        )
    ft_pts = int(free_throws["outcome"].astype(bool).sum()) if len(free_throws) else 0
    recon_total = int(points["points_recon"].sum()) if len(points) else 0
    pts_scored_total = int(chances["ptsScored"].fillna(0).sum()) if "ptsScored" in chances.columns else None
    return {
        "shot_pts_manual": shot_pts,
        "ft_pts_manual": ft_pts,
        "points_recon_sum": recon_total,
        "manual_equals_recon": (shot_pts + ft_pts) == recon_total,
        "ptsScored_sum_for_reference_only": pts_scored_total,
        "ok": (shot_pts + ft_pts) == recon_total,
        "note": "ptsScored must not be used as outcome source even if totals match in these 10 games",
    }


def audit_invariants_primary(touches: pd.DataFrame, passes: pd.DataFrame) -> dict[str, Any]:
    """Fail loudly if primary cohort retains known-bad rows."""
    primary = touches[touches["primary_touch_eligible"]].copy()
    bad_frame = primary[primary["endFrame"] < primary["startFrame"]]
    # Self-passes in primary
    primary_ids = set(zip(primary["gameId"], primary["id"], strict=True))
    self_in_primary = 0
    if "is_self_pass" in passes.columns:
        sp = passes[passes["is_self_pass"].astype(bool) & passes["touchId"].notna()]
        self_in_primary = sum(
            (int(g), t) in primary_ids
            for g, t in sp[["gameId", "touchId"]].itertuples(index=False, name=None)
        )
    checks = {
        "no_bad_frame_order_in_primary": len(bad_frame) == 0,
        "no_self_passes_in_primary": self_in_primary == 0,
        "n_primary_touches": len(primary),
    }
    ok = checks["no_bad_frame_order_in_primary"] and checks["no_self_passes_in_primary"]
    if not ok:
        raise AssertionError(f"Primary cohort invariant failed: {checks}")
    return {"checks": checks, "ok": ok}


def spot_check_calculations_one_game(
    game_id: int,
    touches: pd.DataFrame,
    passes: pd.DataFrame,
    chances: pd.DataFrame,
) -> dict[str, Any]:
    """Independent recomputation spot-checks on one game."""
    t = touches[touches["gameId"] == game_id]
    p = passes[passes["gameId"] == game_id]
    c = chances[chances["gameId"] == game_id]

    elapsed_ok = bool((t["elapsed_frames"] == (t["endFrame"] - t["startFrame"])).all()) if len(t) else True
    dur_corr = float(t["touchTime"].corr(t["duration_s_from_frames"])) if len(t) > 5 else np.nan
    dist_ok = True
    if "distance_abs_diff" in p.columns and p["distance_abs_diff"].notna().any():
        dist_ok = float(p["distance_abs_diff"].dropna().max()) < 1e-6
    usable_ok = bool((c["usable_flag"] == (c["qualityIndex"] >= 3)).all()) if len(c) else True
    return {
        "gameId": game_id,
        "elapsed_frames_formula_ok": elapsed_ok,
        "touchTime_vs_frames_corr": dur_corr,
        "pass_distance_recompute_ok": dist_ok,
        "usable_equals_qi_ge_3": usable_ok,
        "ok": elapsed_ok and dist_ok and usable_ok,
    }
