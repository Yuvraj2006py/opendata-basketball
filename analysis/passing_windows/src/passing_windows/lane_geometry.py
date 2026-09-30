"""Lane and interception geometry for Stage 3 candidate reconstruction.

All primitives are causal with respect to the decision frame: they use current
positions/velocities only. Missing matchups and null velocities stay missing
(NaN / False), never coerced to zero.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .coords import COURT_HALF_X, EVENT_ATTACK_SIGN, normalize_attack_xy
from .geometry import euclidean_distance, lane_clearance, point_to_segment_distance
from .target_inference import (
    COURT_HALF_LENGTH_FT,
    COURT_HALF_WIDTH_FT,
    MAX_PROJECTION_SPEED_FPS,
    project_candidate,
)

# Default kinematic assumptions for transparent geometry-only scores.
DEFAULT_PASS_SPEED_FPS = 35.0
DEFAULT_DEFENDER_SPEED_FPS = 18.0
DEFAULT_CATCH_RADIUS_FT = 2.0
DEFAULT_N_PATH_SAMPLES = 8
MAX_PLAYER_SPEED_FPS = MAX_PROJECTION_SPEED_FPS

# Court region thresholds under event convention (offense toward -x).
PAINT_X_EVENT = -28.0  # ~ft from midcourt toward hoop
PAINT_HALF_WIDTH = 8.0
CORNER_Y_ABS = 20.0
CORNER_X_EVENT = -35.0


def clip_to_court(x: float, y: float) -> tuple[float, float, bool]:
    """Clamp to court bounds; returns clipped flag."""
    if not np.isfinite(x) or not np.isfinite(y):
        return float("nan"), float("nan"), False
    cx = float(np.clip(x, -COURT_HALF_LENGTH_FT, COURT_HALF_LENGTH_FT))
    cy = float(np.clip(y, -COURT_HALF_WIDTH_FT, COURT_HALF_WIDTH_FT))
    return cx, cy, bool(cx != x or cy != y)


def project_receiver_catch(
    x: float,
    y: float,
    vx: float,
    vy: float,
    t_s: float,
    *,
    velocity_ok: bool,
    max_speed_fps: float = MAX_PLAYER_SPEED_FPS,
) -> tuple[float, float, bool, bool]:
    """Project catch point from causal velocity with speed/court caps.

    Returns ``(x, y, projection_used_velocity, clipped)``.
    When velocity is missing/invalid the receiver is held at current position
    and ``projection_used_velocity`` is False (null motion, not zero motion).
    """
    px, py, clipped = project_candidate(
        x, y, vx, vy, t_s, velocity_ok=velocity_ok, max_speed_fps=max_speed_fps
    )
    return px, py, bool(velocity_ok and np.isfinite(vx) and np.isfinite(vy)), bool(clipped)


def sample_segment_points(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    n_samples: int = DEFAULT_N_PATH_SAMPLES,
) -> np.ndarray:
    """Sample inclusive endpoints along segment AB. Shape (n, 2)."""
    n = max(int(n_samples), 2)
    if not (np.isfinite(ax) and np.isfinite(ay) and np.isfinite(bx) and np.isfinite(by)):
        return np.full((n, 2), np.nan)
    t = np.linspace(0.0, 1.0, n)
    xs = ax + t * (bx - ax)
    ys = ay + t * (by - ay)
    return np.column_stack([xs, ys])


def ball_arrival_times_along_path(
    path_xy: np.ndarray,
    *,
    pass_speed_fps: float = DEFAULT_PASS_SPEED_FPS,
) -> np.ndarray:
    """Cumulative ball arrival time (s) at each path sample from the passer."""
    if path_xy.ndim != 2 or path_xy.shape[0] == 0 or not np.isfinite(pass_speed_fps) or pass_speed_fps <= 0:
        return np.full(path_xy.shape[0] if path_xy.ndim == 2 else 0, np.nan)
    diffs = np.diff(path_xy, axis=0)
    seg = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum / float(pass_speed_fps)


def defender_time_to_point(
    def_x: float,
    def_y: float,
    def_vx: float,
    def_vy: float,
    target_x: float,
    target_y: float,
    *,
    velocity_ok: bool,
    defender_speed_fps: float = DEFAULT_DEFENDER_SPEED_FPS,
    catch_radius_ft: float = DEFAULT_CATCH_RADIUS_FT,
) -> float:
    """Estimated arrival time of a defender to a point (seconds).

    Uses current velocity when available to shrink remaining distance along the
    closing component; otherwise falls back to isotropic speed. Missing
    positions or speeds return NaN (never zero).
    """
    if not all(np.isfinite(v) for v in (def_x, def_y, target_x, target_y)):
        return float("nan")
    if not np.isfinite(defender_speed_fps) or defender_speed_fps <= 0:
        return float("nan")

    dx = target_x - def_x
    dy = target_y - def_y
    dist = float(np.hypot(dx, dy))
    remaining = max(0.0, dist - float(catch_radius_ft))
    if remaining <= 0:
        return 0.0

    speed = float(defender_speed_fps)
    if velocity_ok and np.isfinite(def_vx) and np.isfinite(def_vy) and dist > 1e-6:
        # Closing speed toward the target (causal, current only).
        closing = (def_vx * dx + def_vy * dy) / dist
        speed = max(speed, float(closing))
        speed = min(speed, MAX_PLAYER_SPEED_FPS)
    if speed <= 1e-6:
        return float("nan")
    return float(remaining / speed)


def min_lane_clearance(
    passer_xy: tuple[float, float],
    catch_xy: tuple[float, float],
    defender_xys: Sequence[tuple[float, float]],
    *,
    defender_radius: float = 0.0,
) -> float:
    """Minimum clearance of any defender to the passer→catch segment."""
    if not defender_xys:
        return float("nan")
    clearances = []
    for dxy in defender_xys:
        if not (np.isfinite(dxy[0]) and np.isfinite(dxy[1])):
            continue
        clearances.append(
            lane_clearance(passer_xy, catch_xy, (float(dxy[0]), float(dxy[1])), defender_radius)
        )
    return float(np.min(clearances)) if clearances else float("nan")


def min_temporal_interception_margin(
    passer_xy: tuple[float, float],
    catch_xy: tuple[float, float],
    defenders: Sequence[dict],
    *,
    pass_speed_fps: float = DEFAULT_PASS_SPEED_FPS,
    defender_speed_fps: float = DEFAULT_DEFENDER_SPEED_FPS,
    catch_radius_ft: float = DEFAULT_CATCH_RADIUS_FT,
    n_path_samples: int = DEFAULT_N_PATH_SAMPLES,
) -> tuple[float, int | None]:
    """Min over defenders and path samples of (t_def - t_ball).

    Positive => ball arrives first. Returns ``(margin_s, intercepting_defender_id)``.
    """
    path = sample_segment_points(
        passer_xy[0], passer_xy[1], catch_xy[0], catch_xy[1], n_path_samples
    )
    t_ball = ball_arrival_times_along_path(path, pass_speed_fps=pass_speed_fps)
    best = float("nan")
    best_id: int | None = None
    for defn in defenders:
        dx = defn.get("x", np.nan)
        dy = defn.get("y", np.nan)
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue
        did = defn.get("playerId")
        vok = bool(defn.get("velocity_ok", False))
        for i in range(path.shape[0]):
            t_def = defender_time_to_point(
                float(dx),
                float(dy),
                float(defn.get("vx", np.nan)),
                float(defn.get("vy", np.nan)),
                float(path[i, 0]),
                float(path[i, 1]),
                velocity_ok=vok,
                defender_speed_fps=defender_speed_fps,
                catch_radius_ft=catch_radius_ft,
            )
            if not np.isfinite(t_def) or not np.isfinite(t_ball[i]):
                continue
            margin = float(t_def - t_ball[i])
            if not np.isfinite(best) or margin < best:
                best = margin
                best_id = int(did) if did is not None and np.isfinite(did) else None
    return best, best_id


def receiver_separation_at_catch(
    catch_xy: tuple[float, float],
    defender_xys: Sequence[tuple[float, float]],
) -> float:
    """Distance from projected catch to nearest defender (ft)."""
    if not defender_xys or not (np.isfinite(catch_xy[0]) and np.isfinite(catch_xy[1])):
        return float("nan")
    dists = [
        euclidean_distance(catch_xy[0], catch_xy[1], float(d[0]), float(d[1]))
        for d in defender_xys
        if np.isfinite(d[0]) and np.isfinite(d[1])
    ]
    return float(np.min(dists)) if dists else float("nan")


def passer_pressure(
    passer_xy: tuple[float, float],
    defender_xys: Sequence[tuple[float, float]],
) -> float:
    """Nearest-defender distance to the passer (ft). Missing → NaN."""
    return receiver_separation_at_catch(passer_xy, defender_xys)


def target_defender_recovery_time(
    catch_xy: tuple[float, float],
    assigned_defender: dict | None,
    *,
    defender_speed_fps: float = DEFAULT_DEFENDER_SPEED_FPS,
    catch_radius_ft: float = DEFAULT_CATCH_RADIUS_FT,
) -> float:
    """Time for the matched defender to reach the projected catch.

    When ``matchup_matched`` is False / assigned defender is missing, returns NaN
    rather than inventing a nearest-defender substitute.
    """
    if assigned_defender is None:
        return float("nan")
    return defender_time_to_point(
        float(assigned_defender.get("x", np.nan)),
        float(assigned_defender.get("y", np.nan)),
        float(assigned_defender.get("vx", np.nan)),
        float(assigned_defender.get("vy", np.nan)),
        catch_xy[0],
        catch_xy[1],
        velocity_ok=bool(assigned_defender.get("velocity_ok", False)),
        defender_speed_fps=defender_speed_fps,
        catch_radius_ft=catch_radius_ft,
    )


def path_length(ax: float, ay: float, bx: float, by: float) -> float:
    if not all(np.isfinite(v) for v in (ax, ay, bx, by)):
        return float("nan")
    return euclidean_distance(ax, ay, bx, by)


def pass_angle_deg(passer_xy: tuple[float, float], catch_xy: tuple[float, float]) -> float:
    """Bearing of the pass lane in degrees (atan2), event frame."""
    if not all(np.isfinite(v) for v in (*passer_xy, *catch_xy)):
        return float("nan")
    return float(np.degrees(np.arctan2(catch_xy[1] - passer_xy[1], catch_xy[0] - passer_xy[0])))


def sideline_baseline_proximity(x: float, y: float) -> tuple[float, float]:
    """Distance to nearest sideline and baseline (ft) in the event frame."""
    if not (np.isfinite(x) and np.isfinite(y)):
        return float("nan"), float("nan")
    sideline = float(COURT_HALF_WIDTH_FT - abs(y))
    # Baselines at ±COURT_HALF_LENGTH_FT
    baseline = float(COURT_HALF_LENGTH_FT - abs(x))
    return sideline, baseline


def court_region(x_event: float, y_event: float) -> str:
    """Coarse offensive region under the event attacking-hoop convention."""
    if not (np.isfinite(x_event) and np.isfinite(y_event)):
        return "unknown"
    if x_event <= PAINT_X_EVENT and abs(y_event) <= PAINT_HALF_WIDTH:
        return "paint"
    if x_event <= CORNER_X_EVENT and abs(y_event) >= CORNER_Y_ABS:
        return "corner"
    if x_event <= 0:
        return "wing" if abs(y_event) >= 8.0 else "slot"
    return "backcourt"


def rim_relative_features(x_event: float, y_event: float) -> dict[str, float | str]:
    """Location relative to the offensive hoop (event frame, hoop at -COURT_HALF_X-ish)."""
    hoop_x = float(EVENT_ATTACK_SIGN) * float(COURT_HALF_X)  # -47
    if not (np.isfinite(x_event) and np.isfinite(y_event)):
        return {
            "dist_to_rim_ft": float("nan"),
            "angle_to_rim_deg": float("nan"),
            "x_norm": float("nan"),
            "y_norm": float("nan"),
            "court_region": "unknown",
        }
    xn, yn = normalize_attack_xy(x_event, y_event)
    return {
        "dist_to_rim_ft": euclidean_distance(x_event, y_event, hoop_x, 0.0),
        "angle_to_rim_deg": float(np.degrees(np.arctan2(y_event, x_event - hoop_x))),
        "x_norm": xn,
        "y_norm": yn,
        "court_region": court_region(x_event, y_event),
    }


def motion_toward_rim(
    x: float,
    y: float,
    vx: float,
    vy: float,
    *,
    velocity_ok: bool,
) -> float:
    """Component of causal velocity toward the offensive hoop (ft/s). Missing → NaN."""
    if not velocity_ok or not all(np.isfinite(v) for v in (x, y, vx, vy)):
        return float("nan")
    hoop_x = float(EVENT_ATTACK_SIGN) * float(COURT_HALF_X)
    dx, dy = hoop_x - x, 0.0 - y
    dist = float(np.hypot(dx, dy))
    if dist <= 1e-6:
        return 0.0
    return float((vx * dx + vy * dy) / dist)


def mean_pred_error(errors: Sequence[float]) -> float:
    vals = [float(e) for e in errors if np.isfinite(e)]
    return float(np.mean(vals)) if vals else float("nan")


def geometry_only_passability_score(
    *,
    distance_ft: float,
    min_clearance_ft: float,
    min_temporal_margin_s: float,
    receiver_sep_ft: float,
    outside_support: bool,
) -> float:
    """Transparent heuristic in [0, 1] — not a calibrated probability.

    Kept so Stage 4 models can be compared against a simple geometry baseline.
    Outside-support candidates score NaN (rejected), not zero.
    """
    if outside_support:
        return float("nan")
    if not all(
        np.isfinite(v)
        for v in (distance_ft, min_clearance_ft, min_temporal_margin_s, receiver_sep_ft)
    ):
        return float("nan")
    # Soft saturating transforms; coefficients are engineering defaults.
    z = (
        0.15 * min_clearance_ft
        + 2.0 * min_temporal_margin_s
        + 0.08 * receiver_sep_ft
        - 0.04 * distance_ft
    )
    return float(1.0 / (1.0 + np.exp(-z)))


def help_defender_density(
    catch_xy: tuple[float, float],
    defender_xys: Sequence[tuple[float, float]],
    *,
    radius_ft: float = 12.0,
) -> float:
    """Count of defenders within ``radius_ft`` of the projected catch."""
    if not (np.isfinite(catch_xy[0]) and np.isfinite(catch_xy[1])):
        return float("nan")
    n = 0
    for d in defender_xys:
        if not (np.isfinite(d[0]) and np.isfinite(d[1])):
            continue
        if euclidean_distance(catch_xy[0], catch_xy[1], float(d[0]), float(d[1])) <= radius_ft:
            n += 1
    return float(n)
