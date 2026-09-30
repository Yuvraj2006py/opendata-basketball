"""Minimal geometry primitives for Stage 1 (full lane model deferred to Stage 3)."""

from __future__ import annotations

import numpy as np


def euclidean_distance(ax: float, ay: float, bx: float, by: float) -> float:
    return float(np.hypot(ax - bx, ay - by))


def point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Distance from point P to segment AB in feet."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom <= 0:
        return euclidean_distance(px, py, ax, ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx, cy = ax + t * abx, ay + t * aby
    return euclidean_distance(px, py, cx, cy)


def lane_clearance(
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    defender_xy: tuple[float, float],
    defender_radius: float = 0.0,
) -> float:
    """Signed-ish clearance: distance of defender to pass lane minus radius.

    Positive => defender outside the lane by that many feet.
    """
    d = point_to_segment_distance(
        defender_xy[0],
        defender_xy[1],
        passer_xy[0],
        passer_xy[1],
        receiver_xy[0],
        receiver_xy[1],
    )
    return float(d - defender_radius)


def interception_margin(
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    defender_xy: tuple[float, float],
    pass_speed_fps: float,
    defender_speed_fps: float,
    catch_radius: float = 2.0,
) -> float:
    """Kinematic margin in seconds: t_defender_to_lane - t_ball_to_receiver.

    Positive => defender arrives after the ball (open). Negative => contested/blocked path.
    """
    if pass_speed_fps <= 0 or defender_speed_fps <= 0:
        return float("nan")
    pass_dist = euclidean_distance(*passer_xy, *receiver_xy)
    t_pass = pass_dist / pass_speed_fps
    d_lane = point_to_segment_distance(
        defender_xy[0],
        defender_xy[1],
        passer_xy[0],
        passer_xy[1],
        receiver_xy[0],
        receiver_xy[1],
    )
    close = max(0.0, d_lane - catch_radius)
    t_def = close / defender_speed_fps
    return float(t_def - t_pass)
